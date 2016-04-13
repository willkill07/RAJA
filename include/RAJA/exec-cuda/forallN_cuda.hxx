/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */
#if 0
#ifndef RAJA_forallN_cuda_HXX__
#define RAJA_forallN_cuda_HXX__

#include<RAJA/config.hxx>
#include<RAJA/int_datatypes.hxx>

namespace RAJA {



/******************************************************************
 *  ForallN OpenMP Parallel Region policies
 ******************************************************************/

// Tiling Policy
struct ForallN_OMP_Parallel_Tag {};
template<typename NEXT=ForallN_Execute>
struct OMP_Parallel {
  // Identify this policy
  typedef ForallN_OMP_Parallel_Tag PolicyTag;

  // The next nested-loop execution policy
  typedef NEXT NextPolicy;
};



/******************************************************************
 *  forallN_policy(), OpenMP Parallel Region execution
 ******************************************************************/

/*!
 * \brief Tiling policy front-end function.
 */
template<typename POLICY, typename BODY, typename ... PARGS>
RAJA_INLINE void forallN_policy(ForallN_OMP_Parallel_Tag, BODY body, PARGS ... pargs){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

#pragma omp parallel
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), body, pargs...);
  }
}



} // namespace RAJA
  
#endif

#endif
