/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          sequential execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

#ifndef RAJA_reduce_sequential_HXX
#define RAJA_reduce_sequential_HXX

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

#include "RAJA/reducers.hxx"

#include "RAJA/MemUtils_CPU.hxx"

#include "RAJA/internal/reduction_types.hxx"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */

template <typename Reducer>
class UnaryReducer<seq_reduce, Reducer>
{
  using T = Reducer::type;

  explicit UnaryReducer(T init_val = Reducer::init())
      : m_is_copy{false},
        m_reduced_val{init_val},
        m_myID{getCPUReductionId()},
        m_blockdata{getCPUReductionMemBlock(m_myID)}
  {
    m_blockdata[0] = init_val;
  }
  //
  // Copy ctor.
  //
  UnaryReducer(const UnaryReducer<seq_reduce, Reducer>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~UnaryReducer<seq_reduce, Reducer>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  operator T()
  {
    m_reduced_val =
        Reducer::apply(m_reduced_val, static_cast<T>(m_blockdata[0]));
    return m_reduced_val;
  }

  T get() { return operator T(); }

  //
  // Method that updates min value.
  //
  UnaryReducer<seq_reduce, Reducer> apply(T val) const
  {
    m_blockdata[0] = Reducer::apply(static_cast<T>(m_blockdata[0]), val);
    return *this;
  }

private:
  bool m_is_copy;
  int m_myID;
  T m_reduced_val;
  CPUReductionBlockDataType* m_blockdata;
};

template <typename T>
class ReduceMin<seq_reduce, T>
    : public UnaryReducer<seq_reduce, RAJA::internal::Reduction::Min<T>>
{
  using Base = UnaryReducer<seq_reduce, RAJA::internal::Reduction::Min<T>>;

public:
  ReduceMin() : Base() {}
  ReduceMin(T init_val) : Base(init_val) {}
  ReduceMin(const ReduceMin<seq_reduce, T>& other) : Base(other) {}
};


template <typename T>
class ReduceMax<seq_reduce, T>
    : public UnaryReducer<seq_reduce, RAJA::internal::Reduction::Max<T>>
{
  using Base = UnaryReducer<seq_reduce, RAJA::internal::Reduction::Max<T>>;

public:
  ReduceMax() : Base() {}
  ReduceMax(T init_val) : Base(init_val) {}
  ReduceMax(const ReduceMax<seq_reduce, T>& other) : Base(other) {}
};

template <typename T>
class ReduceSum<seq_reduce, T>
    : public UnaryReducer<seq_reduce, RAJA::internal::Reduction::Sum<T>>
{
  using Base = UnaryReducer<seq_reduce, RAJA::internal::Reduction::Sum<T>>;

public:
  ReduceSum() : Base() {}
  ReduceSum(T init_val) : Base(init_val) {}
  ReduceSum(const ReduceSum<seq_reduce, T>& other) : Base(other) {}
};

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<seq_reduce, T>
{
public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMinLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;

    m_reduced_val = init_val;

    m_myID = getCPUReductionId();

    m_blockdata = getCPUReductionMemBlock(m_myID);
    m_blockdata[0] = init_val;

    m_idxdata = getCPUReductionLocBlock(m_myID);
    m_idxdata[0] = init_loc;
  }

  //
  // Copy ctor.
  //
  ReduceMinLoc(const ReduceMinLoc<seq_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMinLoc<seq_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    if (static_cast<T>(m_blockdata[0]) <= m_reduced_val) {
      m_reduced_val = m_blockdata[0];
      m_reduced_idx = m_idxdata[0];
    }
    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced min value.
  //
  Index_type getLoc()
  {
    if (static_cast<T>(m_blockdata[0]) <= m_reduced_val) {
      m_reduced_val = m_blockdata[0];
      m_reduced_idx = m_idxdata[0];
    }
    return m_reduced_idx;
  }

  //
  // Method that updates min and index values.
  //
  ReduceMinLoc<seq_reduce, T> minloc(T val, Index_type idx) const
  {
    if (val <= static_cast<T>(m_blockdata[0])) {
      m_blockdata[0] = val;
      m_idxdata[0] = idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMinLoc<seq_reduce, T>();

  bool m_is_copy;
  int m_myID;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CPUReductionBlockDataType* m_blockdata;
  Index_type* m_idxdata;
};


/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<seq_reduce, T>
{
public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMaxLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;

    m_reduced_val = init_val;

    m_myID = getCPUReductionId();

    m_blockdata = getCPUReductionMemBlock(m_myID);
    m_blockdata[0] = init_val;

    m_idxdata = getCPUReductionLocBlock(m_myID);
    m_idxdata[0] = init_loc;
  }

  //
  // Copy ctor.
  //
  ReduceMaxLoc(const ReduceMaxLoc<seq_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMaxLoc<seq_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    if (static_cast<T>(m_blockdata[0]) >= m_reduced_val) {
      m_reduced_val = m_blockdata[0];
      m_reduced_idx = m_idxdata[0];
    }
    return m_reduced_val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced max value.
  //
  Index_type getLoc()
  {
    if (static_cast<T>(m_blockdata[0]) >= m_reduced_val) {
      m_reduced_val = m_blockdata[0];
      m_reduced_idx = m_idxdata[0];
    }
    return m_reduced_idx;
  }

  //
  // Method that updates max and index values.
  //
  ReduceMaxLoc<seq_reduce, T> maxloc(T val, Index_type idx) const
  {
    if (val >= static_cast<T>(m_blockdata[0])) {
      m_blockdata[0] = val;
      m_idxdata[0] = idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMaxLoc<seq_reduce, T>();

  bool m_is_copy;
  int m_myID;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CPUReductionBlockDataType* m_blockdata;
  Index_type* m_idxdata;
};


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
