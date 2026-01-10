#ifndef LNA_ALLOC_H
#define LNA_ALLOC_H

#include <stdint.h>

#include "lna_number.h"

#define KiB(n) ( (u64)(n) << 10 )
#define MiB(n) ( (u64)(n) << 20 )
#define GiB(n) ( (u64)(n) << 30 )

#define ARENA_ALIGN (sizeof(void*))
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

typedef struct {
  void* base;
  u64 capacity; //bytes
  u64 pos; //bytes used
} ml_arena;

void create_ml_arena(ml_arena* arena,u64 capacity,void* mem);

void* push_ml_arena(ml_arena* arena,u64 size);
void  pop_ml_arena(ml_arena* arena,u64 size);

u64 get_free_mem(ml_arena* arena);

#endif //LNA_ALLOC_H
