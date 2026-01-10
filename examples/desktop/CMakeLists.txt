add_library(lna_examples_common STATIC
  common/test_common.c
)

target_include_directories(lna_examples_common PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/common
)

target_link_libraries(lna_examples_common PUBLIC lna::lna)
target_compile_features(lna_examples_common PUBLIC c_std_11)

# add_executable(lna_example_basic hello_test.c)
# target_link_libraries(lna_example_basic PRIVATE lna_examples_common)

