#include "lna_arena.h"
#include "lna_number.h"

#include <stdio.h>

void create_ml_arena(ml_arena *arena, u64 capacity, void *mem) {
  arena->base = (u8*) mem;
  arena->capacity = capacity;
  arena->pos = 0;
}

void* push_ml_arena(ml_arena *arena, u64 size) {
  u64 cur = arena->pos; 
  u64 aligned = ALIGN_UP_POW2(cur,ARENA_ALIGN);
  if (aligned > arena->capacity) return NULL;
  if (size > (arena->capacity - aligned)) return NULL;
  void* out = (u8*) arena->base + aligned;
  arena->pos = aligned + size;
  return out;
}

void pop_ml_arena(ml_arena *arena,u64 size) {
  if (size >= arena->pos)
    arena->pos = 0;
  else
    arena->pos -= size;
}


u64 get_free_mem(ml_arena *arena) {
  return arena->capacity - arena->pos; 
}

