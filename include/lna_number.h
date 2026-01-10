#ifndef LNA_NUMBER_H
#define LNA_NUMBER_H

#include <stdint.h>

_Static_assert(sizeof(float) == 4, "Expected 32-bit float");
_Static_assert(sizeof(double) == 8, "Expected 64-bit double");

#define euler (f32) 2.71828
#define eps 1e-12f
#define pi 3.14159

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef u8 b8;
#endif //LNA_NUMBER_H
