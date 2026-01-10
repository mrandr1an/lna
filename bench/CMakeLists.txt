cmake_minimum_required(VERSION 3.20)
project(lna_bench LANGUAGES CXX)

include(FetchContent)

set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.0  
  GIT_SUBMODULES_RECURSE TRUE
)

FetchContent_MakeAvailable(benchmark)

# Executables go here
add_executable(bench_test bench_test.cpp)
target_compile_features(bench_test PRIVATE cxx_std_17)

target_link_libraries(bench_test PRIVATE lna::lna benchmark::benchmark benchmark::benchmark_main)

add_custom_target(benchmarks DEPENDS bench_math)
