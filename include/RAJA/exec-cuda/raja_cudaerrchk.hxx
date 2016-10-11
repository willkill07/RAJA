/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing utility methods used in CUDA operations.
 *
 *          These methods work only on platforms that support CUDA.
 *
 ******************************************************************************
 */

#ifndef RAJA_raja_cudaerrchk_HXX
#define RAJA_raja_cudaerrchk_HXX

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

#include <iostream>
#include <string>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace RAJA
{

inline int raja_cuda_memlog_checkprt() {
   static const int do_log = (getenv("RAJA_CUDA_MEMLOG") != NULL) ? 1 : 0;
   return do_log;
}

#define raja_cuda_logf(...)                   \
   if (RAJA::raja_cuda_memlog_checkprt()) {   \
      printf(__VA_ARGS__);                    \
      printf(" %s %d\n", __FILE__, __LINE__); \
   }

///
///////////////////////////////////////////////////////////////////////
///
/// Utility assert method used in CUDA operations to report CUDA
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define raja_cudaErrchk(ans)                          \
  {                                              \
    RAJA::cudaAssert((ans), __FILE__, __LINE__); \
  }

inline void cudaAssert(cudaError_t code,
                       const char *file,
                       int line,
                       bool abort = true)
{
  if (code != cudaSuccess) {
    fprintf(
        stderr, "CUDAassert: %s %s %s %d\n", 
        cudaGetErrorName(code), cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


#define raja_cudaDeviceSynchronize()          \
  {                                           \
    raja_cudaErrchk(cudaDeviceSynchronize()); \
    raja_cuda_logf("raja_cudaDeviceSynchronize:");  \
  }

#define raja_cudaPeekAtLastError()            \
  {                                           \
    raja_cudaErrchk(cudaPeekAtLastError());   \
    raja_cuda_logf("raja_cudaPeekAtLastError:");    \
  }

#define raja_cudaMalloc(ptr, len)             \
  {                                           \
    raja_cudaErrchk(cudaMalloc((ptr), (len)));\
    raja_cuda_logf("raja_cudaMalloc: %p (%d)", *(ptr), (len));   \
  }

#define raja_cudaMallocManaged(ptr, len, flags)  \
  {                                           \
    raja_cudaErrchk(cudaMallocManaged((ptr), (len), (flags)));  \
    raja_cuda_logf("raja_cudaMallocManaged: %p (%d)", *(ptr), (len));   \
  }

#define raja_cudaFree(ptr)                    \
  {                                           \
    raja_cudaErrchk(cudaFree((ptr)));         \
    raja_cuda_logf("raja_cudaFree: %p", (ptr));   \
  }

#define raja_cudaMemcpy(dst, src, len, dir)   \
  {                                           \
    raja_cudaErrchk(cudaMemcpy((dst), (src), (len), (dir)));  \
    raja_cuda_logf("raja_cudaMemcpy: %p = %p (%d)", (dst), (src), (len));   \
  }

#define raja_cudaMemcpyAsync(dst, src, len, dir)   \
  {                                           \
    raja_cudaErrchk(cudaMemcpy((dst), (src), (len), (dir)));  \
    raja_cuda_logf("raja_cudaMemcpyAsync: %p = %p (%d)", (dst), (src), (len));   \
  }

#define raja_cudaMemset(ptr, data, len)       \
  {                                           \
    raja_cudaErrchk(cudaMemset((ptr), (data), (len)));  \
    raja_cuda_logf("raja_cudaMemset: %p = %d (%d)", (ptr), (unsigned char)(data), (len));   \
  }

#define raja_cudaMemsetAsync(ptr, data, len)  \
  {                                           \
    raja_cudaErrchk(cudaMemsetAsync((ptr), (data), (len)));  \
    raja_cuda_logf("raja_cudaMemsetAsync: %p = %d (%d)", (ptr), (unsigned char)(data), (len));   \
  }


/*!
 * \def RAJA_CUDA_CHECK_AND_SYNC(Async)
 * Macro that checks for errors and synchronizes based on paramater Async.
 */ 
#define RAJA_CUDA_CHECK_AND_SYNC(Async) \
  raja_cudaPeekAtLastError(); \
  if (!Async) { \
    raja_cudaDeviceSynchronize(); \
  }



}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
