# Must have criterion installed on your OS
find_package(PkgConfig REQUIRED)
pkg_check_modules(CRITERION REQUIRED criterion)

add_executable(unit_tests
  test.c
)
target_include_directories(unit_tests PRIVATE ${CRITERION_INCLUDE_DIRS})

target_link_libraries(unit_tests PRIVATE ${CRITERION_LIBRARIES})
target_link_libraries(unit_tests PUBLIC lna::lna)

# add_test(NAME unit COMMAND unit_tests)
