/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#include <math.h>
#include <cfloat>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hxx"

#include "Compare.hxx"

#define NUM_STREAMS 8

#define TEST_VEC_LEN (1024 * 1024)

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

int main(int argc, char *argv[])
{
  cout << "\n Begin RAJA GPU Streams tests!!! " << endl;

  const int test_repeat = 10;

  cudaStream_t streams[NUM_STREAMS];
  for(int s = 0; s < NUM_STREAMS; s++) {
    cudaStreamCreateWithFlags ( &streams[s], cudaStreamNonBlocking );
  }

  //
  // Allocate and initialize managed data arrays
  //

  double dinit_val = 0.1;
  int iinit_val = 1;
  double *dvalues[NUM_STREAMS];
  int *ivalues[NUM_STREAMS];

  for(int s = 0; s < NUM_STREAMS; s++) {
    cudaMallocManaged((void **)&dvalues[s],
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal);
    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      dvalues[s][i] = dinit_val;
    }

    cudaMallocManaged((void **)&ivalues[s],
                      sizeof(int) * TEST_VEC_LEN,
                      cudaMemAttachGlobal);
    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      ivalues[s][i] = iinit_val;
    }
  }

  ///
  /// Define thread block size for CUDA exec policy
  ///
  const size_t block_size = 256;

  ////////////////////////////////////////////////////////////////////////////
  // Run 3 different sum reduction tests in a loop
  ////////////////////////////////////////////////////////////////////////////

  for (int tcount = 0; tcount < test_repeat; ++tcount) {
    cout << "\t tcount = " << tcount << endl;

    //
    // test 1 run a kernel with 1 reduction value in each stream synchronously.
    //
    {  // begin test 1

      double dtinit = 5.0;

      forall< cuda_stream_exec< seq_exec > >(streams, NUM_STREAMS, [=] (cudaStream_t stream, int s) {
        s_ntests_run++;

        ReduceSum<cuda_reduce<block_size>, double> dsum(dtinit);

        double* dvalue = dvalues[s];

        forall<cuda_exec_async<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dsum += dvalue[i];
        });

        double base_chk_val = dinit_val * double(TEST_VEC_LEN);

        if (!equal(double(dsum), dtinit + base_chk_val)) {
          cout << "\n TEST 1 FAILURE: tcount, s = " << tcount << " , " << s
               << endl;
          cout << "\ts = " << s
               << endl;
          cout << setprecision(20) << "\tdsum = " << static_cast<double>(dsum)
               << " (" << dtinit + base_chk_val << ") " << endl;

        } else {
          s_ntests_passed++;
        }
      } );

    }  // end test 1

    ////////////////////////////////////////////////////////////////////////////

    //
    // test 2 run a kernel with 1 reduction value in each stream synchronously.
    //
    {  // begin test 2
      s_ntests_run++;

      double dtinit = 5.0;

      ReduceSum<seq_reduce, double> dsum(dtinit);

      forall< cuda_stream_exec< seq_exec > >(streams, NUM_STREAMS, [=] (cudaStream_t stream, int s) {

        ReduceSum<cuda_reduce<block_size>, double> dsums(0.0);

        double* dvalue = dvalues[s];

        forall<cuda_exec_async<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dsums += dvalue[i];
        });

        dsum += dsums;
      });

      double base_chk_val = dinit_val * double(TEST_VEC_LEN * NUM_STREAMS);

      if (!equal(double(dsum), dtinit + base_chk_val)) {
        cout << "\n TEST 2 FAILURE: tcount = " << tcount <<
             << endl;
        cout << setprecision(20) << "\tdsum = " << static_cast<double>(dsum)
             << " (" << dtinit + base_chk_val << ") " << endl;

      } else {
        s_ntests_passed++;
      }

    }  // end test 2

    ////////////////////////////////////////////////////////////////////////////

    //
    // test 3 run kernels with 1 reduction value on data used and changed in 
    // different streams in each forall<streams> to test a possible race
    // condition between streams.
    //
    {  // begin test 3

      s_ntests_run++;

      double dtinit = 5.0;

      ReduceSum<omp_reduce, double> dsum(dtinit);

      forall< cuda_stream_exec_async< omp_parallel_for_exec > >(streams, NUM_STREAMS, [=] (cudaStream_t stream, int s) {

        ReduceSum<cuda_reduce<block_size>, double> dsums(0.0);

        double* dvalue = dvalues[s];

        forall<cuda_exec_async<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dsums += dvalue[i];
        });

        dsum += dsums;

        forall<cuda_exec_async<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dvalue[i] = -dinit_val;
        });
      } );

      forall< cuda_stream_exec< omp_parallel_for_exec > >(streams, NUM_STREAMS, [=] (int s) {

        ReduceSum<cuda_reduce<block_size>, double> dsums(0.0);

        double* dvalue = dvalues[NUM_STREAMS - 1 - s];

        forall<cuda_exec_async<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dsums += dvalue[i];
        });

        dsum += dsums;

        forall<cuda_exec_async<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dvalue[i] = dinit_val;
        });
      } );

      double base_chk_val = 0.0;

      if (!equal(double(dsum), dtinit + base_chk_val)) {
        cout << "\n TEST 3 FAILURE: tcount = " << tcount <<
             << endl;
        cout << setprecision(20) << "\tdsum = " << static_cast<double>(dsum)
             << " (" << dtinit + base_chk_val << ") " << endl;

      } else {
        s_ntests_passed++;
      }

    }  // end test 3

    ////////////////////////////////////////////////////////////////////////////

  }  // end test repeat loop

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cudaFree(dvalue);
  cudaFree(ivalue);

  free(hdvalue);
  free(hivalue);

  return !(s_ntests_passed == s_ntests_run);
}
