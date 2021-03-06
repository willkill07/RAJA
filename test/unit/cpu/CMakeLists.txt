###############################################################################
#
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For additional details, please also read RAJA/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

if (NOT RAJA_ENABLE_NESTED)
  add_definitions(-DRAJA_ENABLE_NESTED)
endif()

raja_add_library(NAME bis SOURCES buildIndexSet.cxx)

raja_add_test(
  NAME CPUreduce-test
  SOURCES main-reduce.cxx
  DEPENDS_ON bis)

raja_add_test(
  NAME CPUnested-test
  SOURCES main-nested.cxx)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME CPUnested_reduce-test
    SOURCES main-nested-reduce.cxx)
endif()

raja_add_test(
  NAME indexset-tests
  SOURCES test-indexsets.cxx
  DEPENDS_ON bis gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

if (NOT MSVC)
  raja_add_test(
    NAME reduction-tests
    SOURCES test-reductions.cxx
    DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

  raja_add_test(
    NAME scan-tests
    SOURCES test-scan.cxx
    DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

  raja_add_test(
    NAME forall-tests
    SOURCES test-forall.cxx
    DEPENDS_ON bis gtest gtest_main ${CMAKE_THREAD_LIBS_INIT}) 
endif ()

