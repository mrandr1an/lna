#ifndef LNA_MATRIX_H
#define LNA_MATRIX_H

#include "lna_number.h"
#include "lna_arena.h"

typedef enum {
  INVALID_DIMENSIONS,
  INVALID_ELEMENT_TYPES,
  INVALID_ALLOC,
  LNA_TODO,
  LNA_OK,
} LnaOpStatus;

typedef struct {
  u64 rows, cols;
  f32* data;      
} Matrix;

/* Matrix Initialization Operations */
void create_Matrix(ml_arena* arena,Matrix* dest,u64 rows,u64 cols);
void fill_row_Matrix(Matrix* mat,u64 row,f32 col_vals[]);

/* Matrix Memory Operations */
float mget(Matrix matrix, u64 row, u64 col);
void mset(Matrix *matrix, u64 row, u64 col , f32 val);

/* Matrix x Matrix Operations */
LnaOpStatus Matrix_mut_Matrix(ml_arena* arena,Matrix* out,Matrix lhs,Matrix rhs);
LnaOpStatus Matrix_plus_Matrix(ml_arena* arena,Matrix* out,Matrix lhs,Matrix rhs);
LnaOpStatus Matrix_minus_Matrix(ml_arena* arena,Matrix* out,Matrix lhs,Matrix rhs);

LnaOpStatus Matrix_mut_Matrix_inplace(Matrix* lhs,Matrix rhs);
LnaOpStatus Matrix_plus_Matrix_inplace(Matrix* lhs,Matrix rhs);
LnaOpStatus Matrix_minus_Matrix_inplace(Matrix* lhs,Matrix rhs);

/* Matrix x Scalar Operations */
LnaOpStatus Matrix_mut_Scalar(Matrix* lhs,f32 rhs);
LnaOpStatus Matrix_plus_Scalar(Matrix* lhs,f32 rhs);
LnaOpStatus Matrix_minus_Scalar(Matrix* lhs,f32 rhs);

/* Matrix Unary Operations */
LnaOpStatus Matrix_transpose(ml_arena* arena,Matrix* out,Matrix target);

/* Matrix Non Linear Operations */
LnaOpStatus Matrix_plus_Matrix_rowwise(Matrix* lhs, Matrix rhs);
#endif //LNA_MATRIX_H
