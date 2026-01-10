function(lna_apply_warnings target)
  option(LNA_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wshadow
      -Wconversion -Wsign-conversion
      -Wstrict-prototypes -Wmissing-prototypes
    )
    if(LNA_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()
