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
#include <unordered_map>

namespace RAJA
{

////////////////////////////////////////////////////////////////////////////////////////////////
  
namespace
{
  struct cuda_reduction_variable_t {
    CudaReductionDummyDataType host_value = {{0}};
    CudaReductionDummyDataType init_device_value = {{0}};
    int smem_offset = -1;
    int num_threads = -1;
  };
  
  struct cuda_tally_state_t {
    bool dirty = false;
    bool assigned = false;
  };
  
  struct cuda_stream_reducers_t {
    bool* reduction_id_used = nullptr;
    cuda_reduction_variable_t* reduction_variables = nullptr;
    CudaReductionDummyBlockType* mem_block = nullptr;
    cuda_tally_state_t* tally_state = nullptr;
    CudaReductionDummyTallyType* tally_block_device = nullptr;
    CudaReductionDummyTallyType* tally_block_host = nullptr;
    cudaEvent_t event;
    int smem_total = 0;
    int tally_dirty = 0;
    bool in_raja_cuda_forall = false;
    bool tally_valid = true;

    cuda_stream_reducers_t() {
      cudaErrchk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    cuda_stream_reducers_t(cuda_stream_reducers_t const& other)
      : reduction_id_used(other.reduction_id_used),
        reduction_variables(other.reduction_variables),
        mem_block(other.mem_block),
        tally_state(other.tally_state),
        tally_block_device(other.tally_block_device),
        tally_block_host(other.tally_block_host),
        smem_total(other.smem_total),
        tally_dirty(other.tally_dirty),
        in_raja_cuda_forall(other.in_raja_cuda_forall),
        tally_valid(other.tally_valid)
    {
      cudaErrchk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    cuda_stream_reducers_t(cuda_stream_reducers_t&& other)
      : reduction_id_used(other.reduction_id_used),
        reduction_variables(other.reduction_variables),
        mem_block(other.mem_block),
        tally_state(other.tally_state),
        tally_block_device(other.tally_block_device),
        tally_block_host(other.tally_block_host),
        smem_total(other.smem_total),
        tally_dirty(other.tally_dirty),
        in_raja_cuda_forall(other.in_raja_cuda_forall),
        tally_valid(other.tally_valid)
    {
      other.reduction_id_used = nullptr;
      other.reduction_variables = nullptr;
      other.mem_block = nullptr;
      other.tally_state = nullptr;
      other.tally_block_device = nullptr;
      other.tally_block_host = nullptr;
      other.smem_total = 0;
      other.tally_dirty = 0;
      other.in_raja_cuda_forall = false;
      other.tally_valid = true;

      cudaErrchk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    
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
      cudaErrchk(cudaEventDestroy(event));
    }

  private:
    cuda_stream_reducers_t& operator=(cuda_stream_reducers_t const& other);
  };

  bool s_in_cuda_forall_streams = false;

  thread_local cudaStream_t s_currentStream = 0;

  cudaStream_t* s_reduction_streams = nullptr;
  cudaEvent_t* s_reduction_events = nullptr;
  
  std::unordered_map< cudaStream_t, cuda_stream_reducers_t > s_cuda_stream_reducers;

  void checkStream(cudaStream_t stream)
  {
    auto s = s_cuda_stream_reducers.find(stream);

    // check if never seen this stream before
    if (s == s_cuda_stream_reducers.end()) {
      
      s = s_cuda_stream_reducers.insert({stream, cuda_stream_reducers_t()}).first;

      s->second.reduction_id_used = new bool[RAJA_MAX_REDUCE_VARS];
      
      s->second.reduction_variables = new cuda_reduction_variable_t[RAJA_MAX_REDUCE_VARS];

      for (int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
        s->second.reduction_id_used[i] = false;
      }
    }
  }

  void checkStreams(cudaStream_t const* streams, size_t len)
  {
    for (size_t i = 0; i < len; ++i) {
      checkStream(streams[i]);
    }
  }

  void allocateReductionStreams() {
    if (s_reduction_streams == nullptr) {
      s_reduction_streams = new cudaStream_t[RAJA_MAX_REDUCE_VARS];
      s_reduction_events = new cudaEvent_t[RAJA_MAX_REDUCE_VARS];
      for( int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
        cudaErrchk(cudaStreamCreateWithFlags(&s_reduction_streams[i], cudaStreamNonBlocking));
        cudaErrchk(cudaEventCreateWithFlags(&s_reduction_events[i], cudaEventDisableTiming));
      }
      atexit([](){
        if (s_reduction_streams != nullptr) {
          for( int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
            cudaErrchk(cudaStreamDestroy(s_reduction_streams[i]));
            cudaErrchk(cudaEventDestroy(s_reduction_events[i]));
          }
          delete[] s_reduction_streams;
          delete[] s_reduction_events;
          s_cuda_stream_reducers.clear();
        }
      });
    }
  }

}


void beforeCudaStreamsLaunch()
{
  assert(!s_in_cuda_forall_streams);
  s_in_cuda_forall_streams = true;
}

void afterCudaStreamsLaunch()
{
  s_in_cuda_forall_streams = false;
}

cudaStream_t getReducerStream(int id)
{
  return s_reduction_streams[id];
}

cudaEvent_t getReducerEvent(int id)
{
  return s_reduction_events[id];
}

void setStream(cudaStream_t stream)
{
  s_currentStream = stream;
}

cudaStream_t getStream()
{
  return s_currentStream;
}

cudaEvent_t getEvent(cudaStream_t stream)
{
  auto s = s_cuda_stream_reducers.find(stream);
  assert(s != s_cuda_stream_reducers.end());
  return s->second.event;
}

void splitStream(cudaStream_t const* streams, size_t len, cudaStream_t prev_stream)
{
  
  allocateReductionStreams();
  checkStream(prev_stream);
  checkStreams(streams, len);

  cudaEvent_t prev_event = getEvent(prev_stream);
  
  cudaErrchk(cudaEventRecord(prev_event, prev_stream));
  
  for(int i = 0; i < len; ++i) {
    assert(streams[i] != prev_stream);

    cudaErrchk(cudaStreamWaitEvent(streams[i], prev_event, 0));
  }
}

void joinStream(cudaStream_t const* streams, size_t len, cudaStream_t prev_stream)
{
  
  for(int i = 0; i < len; ++i) {
    cudaEvent_t event = getEvent(streams[i]);

    cudaErrchk(cudaEventRecord(event, streams[i]));

    cudaErrchk(cudaStreamWaitEvent(prev_stream, event, 0));
  }

  setStream(prev_stream);
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
  
  allocateReductionStreams();
  checkStream(getStream());

  auto s = s_cuda_stream_reducers.find(getStream());

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
bool getCudaReductionTallyBlock_impl(
        int id, void** host_tally, void** device_tally, 
        CudaReductionDummyDataType** init_device_value)
{

  auto s = s_cuda_stream_reducers.find(getStream());

  assert (s != s_cuda_stream_reducers.end());

  if (s->second.in_raja_cuda_forall) {

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
      // first time used with this stream

      // check id actually being used and this reduction wasn't passed in
      // through a forall<streams>
      assert(s->second.reduction_id_used[id]);
      
      s->second.tally_dirty += 1;
      // set block dirty
      s->second.tally_state[id].dirty = true;
      s->second.tally_state[id].assigned = true;

      init_device_value[0] = &s->second.reduction_variables[id].init_device_value;

      memset(&s->second.tally_block_host[id], 0, 
                        sizeof(CudaReductionDummyTallyType));
    }

    host_tally[0]   = &s->second.tally_block_host[id];
    device_tally[0] = &s->second.tally_block_device[id];
  }

  return s_in_cuda_forall_streams;
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
                                   cudaMemcpyHostToDevice,
                                   getStream()));
        
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
                                cudaMemcpyDeviceToHost,
                                getStream()));
    s->second.tally_valid = true;
  }
}
static void readCudaReductionTallyBlock()
{
  
  auto s = s_cuda_stream_reducers.find(getStream());
    
  assert (s != s_cuda_stream_reducers.end());
  
  if (!s->second.tally_valid) {
    cudaErrchk(cudaMemcpyAsync( &s->second.tally_block_host[0],
                                &s->second.tally_block_device[0],
                                sizeof(CudaReductionDummyTallyType) *
                                  RAJA_CUDA_REDUCE_TALLY_LENGTH,
                                cudaMemcpyDeviceToHost,
                                getStream()));
    cudaErrchk(cudaStreamSynchronize(getStream()));
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

      s->second.reduction_variables[id].smem_offset = s->second.smem_total;

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
