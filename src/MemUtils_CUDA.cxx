/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

#include "RAJA/config.hxx"

#if defined(RAJA_ENABLE_CUDA)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read raja/README-license.txt.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/int_datatypes.hxx"
#include "RAJA/reducers.hxx"
#include "RAJA/exec-cuda/raja_cuda.hxx"
#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"
#include "RAJA/exec-cuda/raja_cudaerrchk.hxx"

#include <iostream>
#include <string>
#include <cassert>
#include <cstring>

namespace RAJA
{

////////////////////////////////////////////////////////////////////////////////////////////////
  
namespace
{
  thread_local cudaStream_t currentStream = 0;
  
  std::unordered_map<cudaStream_t, cudaEvent_t> s_stream_events;
}

void registerStreams(cudaStream_t const* streams, size_t num_streams)
{
  {
    auto s = s_stream_events.find(getStream());
    if(s == s_stream_events.end()) {
        cudaEvent_t event;
        cudaErrchk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        s_stream_events.insert({getStream(), event});
    }
  }
  for (size_t i = 0; i < num_streams; ++s) {
    auto s = s_stream_events.find(streams[i]);
    if(s == s_stream_events.end()) {
        cudaEvent_t event;
        cudaErrchk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        s_stream_events.insert({streams[i], event});
    }
  }
}

void switchStream(cudaStream_t stream, cudaStream_t prev_stream, bool prev_event_recorded)
{
  cudaEvent_t prev_event = getEvent(prev_stream);
  
  if (!prev_event_recorded) {
    cudaErrchk(cudaEventRecord(prev_event, prev_stream));
  }
  
  cudaErrchk(cudaStreamWaitEvent(stream, prev_event, 0));
  
  currentStream = stream;
}

void cudaStream_t getStream()
{
  return currentStream;
}

void cudaEvent_t getEvent(cuda_stream_t stream)
{
  auto s = s_stream_events.find(stream);
  assert(s != s_stream_events.end());
  return s->second;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
  struct cuda_reduction_variable_t {
    CudaReductionDummyDataType host_value = {0};
    CudaReductionDummyDataType init_device_value = {0};
    int smem_offset = -1;
    int num_threads = -1;
  };
  
  struct cuda_tally_state_t {
    bool dirty = false;
    bool assigned = false;
  }
  
  struct cuda_stream_reducers_t {
    bool* reduction_id_used = nullptr;
    cuda_reduction_variable_t* reduction_variables = nullptr;
    CudaReductionDummyBlockType* mem_block = nullptr;
    cuda_tally_state_t* tally_state = nullptr;
    CudaReductionDummyTallyType* tally_block_device = nullptr;
    CudaReductionDummyTallyType* tally_block_host = nullptr;
    int smem_total = 0;
    int tally_dirty = 0;
    bool in_raja_cuda_forall = false;
    bool tally_valid = true;
    
    ~cuda_stream_reducers_t() {
      if (reduction_id_used != nullptr) {
        delete[] reduction_id_used;
      }
      if (reduction_variables != nullptr) {
        delete[] reduction_variables;
      }
      if (mem_block != nullptr) {
        cudaFree(mem_block);
      }
      if (tally_state != nullptr) {
        delete[] tally_state;
      }
      if (tally_block_device != nullptr) {
        cudaFree(tally_block_device);
      }
      if (tally_block_host != nullptr) {
        delete[] tally_block_host;
      }
    }
  };
  
  std::unordered_map< cudaStream_t, cuda_stream_reducers_t > s_cuda_stream_reducers;
}

/*
*******************************************************************************
*
* Return next available valid reduction id, or complain and exit if
* no valid id is available.
*
*******************************************************************************
*/
int getCudaReductionId_impl(void** host_val, void** init_dev_val)
{
  auto s = s_cuda_stream_reducers.find(getStream());

  // check if never seen this stream before
  if (s == s_cuda_stream_reducers.end()) {
    
    s = s_cuda_stream_reducers.insert({getStream(), cuda_stream_reducers_t()}).first;

    s->second.reduction_id_used = new bool[RAJA_MAX_REDUCE_VARS];
    
    s->second.reduction_variables = new cuda_reduction_variable_t[RAJA_MAX_REDUCE_VARS];

    for (int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
      s->second.reduction_id_used[i] = false;
    }
  }

  int id = 0;
  while (id < RAJA_MAX_REDUCE_VARS && s->second.reduction_id_used[id]) {
    id++;
  }

  if (id >= RAJA_MAX_REDUCE_VARS) {
    std::cerr << "\n Exceeded allowable RAJA CUDA reduction count, "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  s->second.reduction_id_used[id] = true;

  host_val[0] = &s->second.reduction_variables[id].host_value;
  init_dev_val[0] = &s->second.reduction_variables[id].init_device_value;

  return id;
}

/*
*******************************************************************************
*
* Release given reduction id and make inactive.
*
*******************************************************************************
*/
void releaseCudaReductionId(int id)
{
  auto s = s_cuda_stream_reducers.find(getStream());

  assert (s != s_cuda_stream_reducers.end());
  
  s->second.reduction_id_used[id] = false;
}

/*
*******************************************************************************
*
* Return pointer into RAJA-CUDA reduction device memory block
* for reducer object with given id. Allocate block if not already allocated.
*
*******************************************************************************
*/
void getCudaReductionMemBlock(int id, void** device_memblock)
{
  auto s = s_cuda_stream_reducers.find(getStream());
  
  assert (s != s_cuda_stream_reducers.end());
  
  if (s->second.mem_block == nullptr) {
    cudaErrchk(cudaMalloc((void**)&s->second.mem_block,
                          sizeof(CudaReductionDummyBlockType) *
                            RAJA_MAX_REDUCE_VARS));
  }

  device_memblock[0] = &s->second.mem_block[id];
}


/*
*******************************************************************************
*
* Return pointer into RAJA-CUDA reduction host tally block cache
* and device tally block for reducer object with given id.
* Allocate blocks if not already allocated.
*
*******************************************************************************
*/
CudaReductionDummyDataType* getCudaReductionTallyBlock_impl(
                              int id, void** host_tally, void** device_tally)
{
  CudaReductionDummyDataType* init_dev_val_ptr = nullptr;

  if (s_in_raja_cuda_forall) {
    
    auto s = s_cuda_stream_reducers.find(getStream());
    
    assert (s != s_cuda_stream_reducers.end());

    if (s->second.tally_state == nullptr) {
      
      s->second.tally_state = 
          new cuda_tally_state_t[RAJA_CUDA_REDUCE_TALLY_LENGTH];

      s->second.tally_block_host = 
          new CudaReductionDummyTallyType[RAJA_CUDA_REDUCE_TALLY_LENGTH];

      cudaErrchk(cudaMalloc((void**)&s->second.tally_block_device,
                            sizeof(CudaReductionDummyTallyType) *
                              RAJA_CUDA_REDUCE_TALLY_LENGTH));

      s->second.tally_valid = true;
      s->second.tally_dirty = 0;
    }
    
    if (!s->second.tally_state[id].assigned) {
      
      s->second.tally_dirty += 1;
      // set block dirty
      s->second.tally_state[id].dirty = true;
      s->second.tally_state[id].assigned = true;

      init_dev_val_ptr = &s->second.reduction_variables[id].init_device_value;

      memset(&s->second.tally_block_host[id], 0, 
                        sizeof(CudaReductionDummyTallyType));
    }

    host_tally[0]   = &s->second.tally_block_host[id];
    device_tally[0] = &s->second.tally_block_device[id];
  }

  return init_dev_val_ptr;
}

/*
*******************************************************************************
*
* Write back dirty tally blocks to device tally blocks.
* Can be called before tally blocks have been allocated.
*
*******************************************************************************
*/
static void writeBackCudaReductionTallyBlock()
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  if (s->second.tally_dirty > 0) {
    int first = 0;
    while (first < RAJA_CUDA_REDUCE_TALLY_LENGTH) {
      if (s->second.tally_state[first].dirty) {
        int end = first + 1;
        while (end < RAJA_CUDA_REDUCE_TALLY_LENGTH
               && s->second.tally_state[end].dirty) {
          end++;
        }
        int len = (end - first);
        cudaErrchk(cudaMemcpyAsync(&s->second.tally_block_device[first],
                                   &s->second.tally_block_host[first],
                                   sizeof(CudaReductionDummyTallyType) * len,
                                   cudaMemcpyHostToDevice));
        
        for (int i = first; i < end; ++i) {
          s->second.tally_state[i].dirty = false;
        }
        first = end + 1;
      } else {
        first++;
      }
    }
    s->second.tally_dirty = 0;
  }
}

/*
*******************************************************************************
*
* Read tally block from device if invalid on host.
* Must be called after tally blocks have been allocated.
* The Async version is synchronous on the host if 
* s_cuda_reduction_tally_block_host is allocated as pageable host memory 
* and not if allocated as pinned host memory or managed memory.
*
*******************************************************************************
*/
static void readCudaReductionTallyBlockAsync()
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  if (!s->second.tally_valid) {
    cudaErrchk(cudaMemcpyAsync( &s->second.tally_block_host[0],
                                &s->second.tally_block_device[0],
                                sizeof(CudaReductionDummyTallyType) *
                                  RAJA_CUDA_REDUCE_TALLY_LENGTH,
                                cudaMemcpyDeviceToHost));
    s->second.tally_valid = true;
  }
}
static void readCudaReductionTallyBlock()
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  if (!s->second.tally_valid) {
    cudaErrchk(cudaMemcpy(  &s->second.tally_block_host[0],
                            &s->second.tally_block_device[0],
                            sizeof(CudaReductionDummyTallyType) *
                              RAJA_CUDA_REDUCE_TALLY_LENGTH,
                            cudaMemcpyDeviceToHost));
    s->second.tally_valid = true;
  }
}

/*
*******************************************************************************
*
* Must be called before each RAJA cuda kernel, and before the copy of the 
* loop body to setup state of the dynamic shared memory variables.
* Ensures that all updates to the tally block are visible on the device by 
* writing back dirty cache lines; this invalidates the tally cache on the host.
*
*******************************************************************************
*/
void beforeCudaKernelLaunch()
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  s->second.in_raja_cuda_forall = true;
  s->second.smem_total = 0;
  for(int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
    s->second.reduction_variables[i].smem_offset = -1;
    s->second.reduction_variables[i].num_threads = -1;
  }
}



/*
*******************************************************************************
*
* Must be called after each RAJA cuda kernel.
* This resets the state of the dynamic shared memory variables.
*
*******************************************************************************
*/
void afterCudaKernelLaunch()
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  s->second.in_raja_cuda_forall = false;
  s->second.smem_total = 0;
}

/*
*******************************************************************************
*
* Must be called before reading the tally block cache on the host.
* Ensures that the host tally block cache for cuda reduction variable id can 
* be read.
* Writes any host changes to the tally block cache to the device before 
* updating the host tally blocks with the values on the GPU.
* The Async version is only asynchronous with regards to managed memory and 
* is synchronous to host code.
*
*******************************************************************************
*/
CudaReductionDummyDataType* beforeCudaReadTallyBlockAsync(int id)
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  CudaReductionDummyDataType* data = nullptr;
  if (s->second.tally_state[id].assigned) {
    if (!s->second.tally_state[id].dirty) {
      writeBackCudaReductionTallyBlock();
      readCudaReductionTallyBlockAsync();
    }
    data = &s->second.tally_block_host[id].dummy_val;
  }
  return data;
}
///
CudaReductionDummyDataType* beforeCudaReadTallyBlockSync(int id)
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  CudaReductionDummyDataType* data = nullptr;
  if (s->second.tally_state[id].assigned) {
    if (!s->second.tally_state[id].dirty) {
      writeBackCudaReductionTallyBlock();
      readCudaReductionTallyBlock();
    }
    data = &s->second.tally_block_host[id].dummy_val;
  }
  return data;
}

/*
*******************************************************************************
*
* Release tally block of reduction variable with id.
*
*******************************************************************************
*/
void releaseCudaReductionTallyBlock(int id)
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  if (s->second.tally_state[id].assigned) {
    if (s->second.tally_state[id].dirty) {
      s->second.tally_state[id].dirty = false;
      s->second.tally_dirty -= 1;
    }
    s->second.tally_state[id].assigned = false;
  }
}

/*
*******************************************************************************
*
* Earmark num_threads * size bytes of dynamic shared memory and get the byte 
* offset.
*
*******************************************************************************
*/
int getCudaSharedmemOffset(int id, dim3 reductionBlockDim, int size)
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());

  if (s->second.in_raja_cuda_forall) {
    if (s->second.reduction_variables[id].smem_offset < 0) {
      // in a forall and have not yet gotten shared memory

      s->second.reduction_variables[id].smem_offset = s_shared_memory_amount_total;

      int num_threads = 
          reductionBlockDim.x * reductionBlockDim.y * reductionBlockDim.z;

      // ignore reduction variables that don't use dynamic shared memory
      s->second.reduction_variables[id].num_threads = (size > 0) ? num_threads : 0;

      s->second.smem_total += num_threads * size;
    }
    return s->second.reduction_variables[id].smem_offset;
  } else {
    return -1;
  }
}

/*
*******************************************************************************
*
* Get size in bytes of dynamic shared memory.
* Check that the number of blocks launched is consistent with the max number of 
* blocks reduction variables can handle.
* Check that execution policy num_threads is consistent with active reduction
* policy num_threads.
*
*******************************************************************************
*/
int getCudaSharedmemAmount(dim3 launchGridDim, dim3 launchBlockDim)
{
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  int launch_num_blocks = 
      launchGridDim.x * launchGridDim.y * launchGridDim.z;

  if (launch_num_blocks > RAJA_CUDA_MAX_NUM_BLOCKS) {
    std::cerr << "\n Cuda execution error: "
              << "Can't launch " << launch_num_blocks << " blocks, " 
              << "RAJA_CUDA_MAX_NUM_BLOCKS = " << RAJA_CUDA_MAX_NUM_BLOCKS
              << ", "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  int launch_num_threads = 
      launchBlockDim.x * launchBlockDim.y * launchBlockDim.z;

  for(int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
    int reducer_num_threads = s->second.reduction_variables[i].num_threads;

    if (reducer_num_threads > 0 && launch_num_threads > reducer_num_threads) {
      std::cerr << "\n Cuda execution, reduction policy mismatch: "
                << "reduction policy with BLOCK_SIZE " << reducer_num_threads
                << " can't be used with execution policy with BLOCK_SIZE "
                << launch_num_threads << ", "
                << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
      exit(1);
    }
  }

  s->second.tally_valid = false;
  writeBackCudaReductionTallyBlock();

  return s->second.smem_total;
}

}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_ENABLE_CUDA)
