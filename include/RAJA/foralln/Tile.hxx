/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing tiling policies and mechanics
 *          for forallN templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_forallN_tile_HXX__
#define RAJA_forallN_tile_HXX__

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

#include "RAJA/config.hxx"

#include "RAJA/int_datatypes.hxx"

namespace RAJA
{

/******************************************************************
 *  ForallN tiling policies
 ******************************************************************/

// Policy for no tiling
struct tile_none {
};

// Policy to tile by given block size
template <int TileSize>
struct tile_fixed {
};

// Struct used to create a list of tiling policies
template <typename... PLIST>
struct TileList {
  constexpr const static size_t num_loops = sizeof...(PLIST);
};

// Tiling Policy
struct ForallN_Tile_Tag {
};
template <typename TILE_LIST, typename NEXT = Execute>
struct Tile {
  // Identify this policy
  typedef ForallN_Tile_Tag PolicyTag;

  // A TileList<> that contains a list of Tile policies (tile_none, etc.)
  typedef TILE_LIST TilePolicy;

  // The next nested-loop execution policy
  typedef NEXT NextPolicy;
};

/******************************************************************
 *  Tiling mechanics
 ******************************************************************/

// Forward declaration so the apply_tile's can recurse into peel_tile
template <int TIDX, typename BODY, typename... POLICY_REST, typename TilePolicy>
RAJA_INLINE void forallN_peel_tile(TilePolicy,
                                   BODY body,
                                   POLICY_REST const &... p_rest);

/*!
 * \brief Applys the tile_none policy
 */
template <int TIDX,
          typename TilePolicy,
          typename BODY,
          typename POLICY_INIT,
          typename... POLICY_REST>
RAJA_INLINE void forallN_apply_tile(tile_none,
                                    BODY&& body,
                                    POLICY_INIT const &p_init,
                                    POLICY_REST const &... p_rest)
{
  // printf("TIDX=%d: policy=tile_none\n", (int)TIDX);

  // Pass thru, so just bind the index set
  typedef ForallN_BindFirstArg_Host<typename VarOps::remove_reference<BODY>::type, POLICY_INIT> BOUND;
  BOUND new_body(VarOps::forward<BODY>(body), p_init);

  // Recurse to the next policy
  forallN_peel_tile<TIDX + 1>(TilePolicy{}, VarOps::move(new_body), p_rest...);
}

/*!
 * \brief Applys the tile_fixed<N> policy
 */
template <int TIDX,
          typename TilePolicy,
          typename BODY,
          int TileSize,
          typename POLICY_INIT,
          typename... POLICY_REST>
RAJA_INLINE void forallN_apply_tile(tile_fixed<TileSize>,
                                    BODY&& body,
                                    POLICY_INIT const &p_init,
                                    POLICY_REST const &... p_rest)
{
  // printf("TIDX=%d: policy=tile_fixed<%d>\n", TIDX, TileSize);

  typedef ForallN_BindFirstArg_Host<typename VarOps::remove_reference<BODY>::type, POLICY_INIT> BOUND;

  // tile loop
  Index_type i_begin = p_init.getBegin();
  Index_type i_end = p_init.getEnd();
  for (Index_type i0 = i_begin; i0 < i_end; i0 += TileSize) {
    // Create a new tile
    Index_type i1 = std::min(i0 + TileSize, i_end);
    POLICY_INIT pi_tile(RangeSegment(i0, i1));

    // Pass thru, so just bind the index set
    BOUND new_body(body, pi_tile);

    // Recurse to the next policy
    forallN_peel_tile<TIDX + 1>(TilePolicy{}, VarOps::move(new_body), p_rest...);
  }
}

/*!
 * \brief Functor that wraps calling the next nested-loop execution policy.
 *
 * This is passed into the recursive tiling function forallN_peel_tile.
 */
template <typename NextPolicy, typename BODY, typename... ARGS>
struct ForallN_NextPolicyWrapper {
  BODY body;

  explicit ForallN_NextPolicyWrapper(BODY const &b) : body(b) {}

  explicit ForallN_NextPolicyWrapper(BODY&& b) : body(VarOps::move(b)) {}

  RAJA_INLINE
  void operator()(ARGS const &... args)
  {
    typedef typename NextPolicy::PolicyTag NextPolicyTag;
    forallN_policy<NextPolicy>(NextPolicyTag(), VarOps::move(body), args...);
  }
};

/*!
 * \brief Tiling policy peeling function (termination case)
 *
 * This just executes the built-up tiling function passed in by outer
 * forallN_peel_tile calls.
 */
template <int TIDX, typename BODY, typename TilePolicy>
RAJA_INLINE void forallN_peel_tile(TilePolicy, BODY&& body)
{
  // Termination case just calls the tiling function that was constructed
  body();
}

/*!
 * \brief Tiling policy peeling function, recursively peels off tiling
 * policies and applys them.
 *
 * This peels off the policy, and hands it off to the policy-overloaded
 * forallN_apply_tile function... which in turn recursively calls this function
 */
template <int TIDX,
          typename BODY,
          typename POLICY_INIT,
          typename... POLICY_REST,
          typename... TilePolicies>
RAJA_INLINE void forallN_peel_tile(TileList<TilePolicies...>,
                                   BODY&& body,
                                   POLICY_INIT const &p_init,
                                   POLICY_REST const &... p_rest)
{
  using TilePolicy = TileList<TilePolicies...>;

  // Extract the tiling policy for loop nest TIDX
  using TP = typename VarOps::get_type_at<TIDX, TilePolicies...>::type;

  // Apply this index's policy, then recurse to remaining tile policies
  forallN_apply_tile<TIDX, TilePolicy>(TP(), VarOps::forward<BODY>(body), p_init, p_rest...);
}

/******************************************************************
 *  forallN_policy(), tiling execution
 ******************************************************************/

/*!
 * \brief Tiling policy front-end function.
 */
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_INLINE void forallN_policy(ForallN_Tile_Tag, BODY&& body, PARGS... pargs)
{
  typedef typename POLICY::NextPolicy NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Extract the list of tiling policies from the policy
  using TilePolicy = typename POLICY::TilePolicy;

  // Apply the tiling policies one-by-one with a peeling approach
  typedef ForallN_NextPolicyWrapper<NextPolicy, typename VarOps::remove_reference<BODY>::type, PARGS...> WRAPPER;
  WRAPPER wrapper(VarOps::forward<BODY>(body));
  forallN_peel_tile<0>(TilePolicy{}, VarOps::move(wrapper), pargs...);
}

}  // namespace RAJA

#endif
