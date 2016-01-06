set(RAJA_PLATFORM RAJA_PLATFORM_X86_AVX)
set(RAJA_COMPILER RAJA_COMPILER_CLANG)

if(CMAKE_BUILD_TYPE MATCHES Opt)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x -fopenmp")
endif()
endif()
