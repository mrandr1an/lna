set(LNA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../")  # points to repo root

idf_component_register(
  SRCS
    "${LNA_ROOT}/src/hello.c"
  INCLUDE_DIRS
    "${LNA_ROOT}/include"
)
