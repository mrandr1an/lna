cmake_minimum_required(VERSION 3.20)

project(lna
  VERSION 0.0.1
  LANGUAGES C
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LNA_BUILD_TESTS "Build desktop tests" OFF)
option(LNA_BUILD_EXAMPLES "Build desktop examples" OFF)
option(LNA_BUILD_BENCHMARKS "Build desktop benchmarks" OFF)
option(LNA_BUILD_DOCS "Build library documentation" OFF)
option(LNA_ENABLE_WARNINGS "Enable warning flags for this library" ON)

include(cmake/lna_sources.cmake)
include(cmake/lna_warnings.cmake)

message(STATUS "LNA_SOURCES = '${LNA_SOURCES}'")

add_library(lna STATIC ${LNA_SOURCES})
add_library(lna::lna ALIAS lna)

target_include_directories(lna
  PUBLIC
    $<BUILD_INTERFACE:${LNA_PUBLIC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lna PUBLIC c_std_11)

if(LNA_ENABLE_WARNINGS)
  lna_apply_warnings(lna)
endif()

if(LNA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(LNA_BUILD_EXAMPLES)
  add_subdirectory(examples/desktop)
endif()

if(LNA_BUILD_BENCHMARKS)
  add_subdirectory(bench EXCLUDE_FROM_ALL)
endif()

if(LNA_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md")
  set(DOXYGEN_EXTRACT_ALL NO)
  set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
  set(DOXYGEN_WARN_IF_DOC_ERROR YES)
  set(DOXYGEN_WARNINGS YES)

  doxygen_add_docs(docs
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    COMMENT "Generate API documentation with Doxygen"
  )
endif()
