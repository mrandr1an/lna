#include "lna_matrix.h"
#include "lna_arena.h"

void create_Matrix(ml_arena *arena,Matrix* dest, u64 rows, u64 cols) {
  void* data_ptr = push_ml_arena(arena,(rows*cols)*sizeof(f32));
  dest->rows = rows;
  dest->cols = cols;
  dest->data = data_ptr;
}

void fill_row_Matrix(Matrix *mat, u64 row, f32 col_vals[]) {
    f32* data = (f32*) mat->data;
    u64 base = row * mat->cols;
    for (u64 cur_col = 0; cur_col < mat->cols; ++cur_col) {
	data[base + cur_col] = col_vals[cur_col];
    }
}

LnaOpStatus Matrix_mut_Matrix(ml_arena *arena, Matrix *out, Matrix lhs, Matrix rhs) {
  if(lhs.cols != rhs.rows) return INVALID_DIMENSIONS;
  if(!lhs.data || !rhs.data) return INVALID_ALLOC;

  const u64 lhs_rows = lhs.rows;
  const u64 lhs_cols = lhs.cols;
  const u64 rhs_cols = rhs.cols;

  create_Matrix(arena,out,lhs_rows,rhs_cols);

  const f32* lhs_data = (const f32*)lhs.data;
  const f32* rhs_data = (const f32*)rhs.data;
  f32* out_data = (f32*)out->data;
  for (u64 cur_row = 0; cur_row < out->rows; ++cur_row) {
    const u64 out_row_index = cur_row * out->cols;
    const u64 lhs_row_index = cur_row * lhs_cols;
    for (u64 cur_col = 0; cur_col < out->cols; ++cur_col) {
      f32 acc = 0.0f;
        for (u64 i = 0; i < lhs.cols; ++i) {
	  acc += lhs_data[lhs_row_index + i] * rhs_data[i*rhs_cols + cur_col];
        }
      out_data[out_row_index + cur_col] = acc;
    }
  }

  return LNA_OK;
}

LnaOpStatus Matrix_plus_Matrix(ml_arena *arena, Matrix *out, Matrix lhs,
                               Matrix rhs) {
  if(lhs.cols != rhs.cols) return INVALID_DIMENSIONS;
  if(lhs.rows != rhs.rows) return INVALID_DIMENSIONS;
  if(!lhs.data || !rhs.data) return INVALID_ALLOC;

  
  const u64 lhs_rows = lhs.rows;
  const u64 lhs_cols = lhs.cols;

  create_Matrix(arena,out,lhs_rows,lhs_cols);

  const f32* lhs_data = (const f32*) lhs.data;
  const f32* rhs_data = (const f32*) rhs.data;
  f32* out_data = (f32*) out->data;

  for (u64 cur_row = 0; cur_row < out->rows; cur_row++) {
    //row index is the same for all three matrices
    u64 row_index = cur_row * lhs.cols;
    for (u64 cur_col = 0; cur_col < out->cols; cur_col++) {
      out_data[row_index + cur_col] =
	lhs_data[row_index + cur_col] + rhs_data[row_index + cur_col];
    }
  }
  
  return LNA_OK;  
}

LnaOpStatus Matrix_minus_Matrix(ml_arena *arena, Matrix *out, Matrix lhs,
                               Matrix rhs) {
  if(lhs.cols != rhs.cols) return INVALID_DIMENSIONS;
  if(lhs.rows != rhs.rows) return INVALID_DIMENSIONS;
  if(!lhs.data || !rhs.data) return INVALID_ALLOC;

  
  const u64 lhs_rows = lhs.rows;
  const u64 lhs_cols = lhs.cols;

  create_Matrix(arena,out,lhs_rows,lhs_cols);

  const f32* lhs_data = (const f32*) lhs.data;
  const f32* rhs_data = (const f32*) rhs.data;
  f32* out_data = (f32*) out->data;

  for (u64 cur_row = 0; cur_row < out->rows; cur_row++) {
    //row index is the same for all three matrices
    u64 row_index = cur_row * lhs.cols;
    for (u64 cur_col = 0; cur_col < out->cols; cur_col++) {
      out_data[row_index + cur_col] =
	lhs_data[row_index + cur_col] - rhs_data[row_index + cur_col];
    }
  }
  
  return LNA_OK;  
}

LnaOpStatus Matrix_mut_Scalar(Matrix *lhs, f32 rhs) {
  if(!lhs->data) return INVALID_ALLOC;

  for (u64 cur_row = 0; cur_row < lhs->rows; ++cur_row) {
    u64 row_index = cur_row * lhs->cols;
    for (u64 cur_col = 0; cur_col < lhs->cols; ++cur_col) {
      lhs->data[row_index + cur_col] = lhs->data[row_index + cur_col] * rhs;
    }
  }
  return LNA_OK;
}

LnaOpStatus Matrix_plus_Scalar(Matrix *lhs, f32 rhs) {
  if(!lhs->data) return INVALID_ALLOC;

  for (u64 cur_row = 0; cur_row < lhs->rows; ++cur_row) {
    u64 row_index = cur_row * lhs->cols;
    for (u64 cur_col = 0; cur_col < lhs->cols; ++cur_col) {
      lhs->data[row_index + cur_col] = lhs->data[row_index + cur_col] + rhs;
    }
  }
  return LNA_OK;
}

LnaOpStatus Matrix_minus_Scalar(Matrix *lhs, f32 rhs) {
  if(!lhs->data) return INVALID_ALLOC;

  for (u64 cur_row = 0; cur_row < lhs->rows; ++cur_row) {
    u64 row_index = cur_row * lhs->cols;
    for (u64 cur_col = 0; cur_col < lhs->cols; ++cur_col){
      lhs->data[row_index + cur_col] = lhs->data[row_index + cur_col] - rhs;
    }
  }
  return LNA_OK;
}

LnaOpStatus Matrix_transpose(ml_arena* arena,Matrix* out,Matrix target) { 
  if (!arena) return INVALID_ALLOC;
  if (!target.data) return INVALID_ALLOC;

  create_Matrix(arena,out,target.cols,target.rows);
  f32* out_data = out->data;
  f32* target_data = target.data;

  for (u64 cur_row = 0; cur_row < target.rows; ++cur_row) {
    u64 target_row_index = cur_row * target.cols;
    for (u64 cur_col= 0; cur_col < target.cols; ++cur_col) {
      out_data[cur_col * out->cols + cur_row] =
	target_data[target_row_index + cur_col];
    }
  }

  return LNA_OK;
}

LnaOpStatus Matrix_minus_Matrix_inplace(Matrix* lhs, Matrix rhs) { 
  if(lhs->cols != rhs.cols) return INVALID_DIMENSIONS;
  if(lhs->rows != rhs.rows) return INVALID_DIMENSIONS;
  if(!lhs->data || !rhs.data) return INVALID_ALLOC;

  
  const u64 lhs_rows = lhs->rows;
  const u64 lhs_cols = lhs->cols;


  f32* lhs_data = (f32*) lhs->data;
  const f32* rhs_data = (const f32*) rhs.data;

  for (u64 cur_row = 0; cur_row < lhs_rows; cur_row++) {
    //row index is the same for all three matrices
    u64 row_index = cur_row * lhs->cols;
    for (u64 cur_col = 0; cur_col < lhs_cols; cur_col++) {
      lhs_data[row_index + cur_col] =
	lhs_data[row_index + cur_col] - rhs_data[row_index + cur_col];
    }
  }
  
  return LNA_OK;  
}

float mget(Matrix matrix, u64 row, u64 col) {
  return matrix.data[row * matrix.cols + col];
}
void mset(Matrix *matrix, u64 row, u64 col , f32 val) {
  matrix->data[row * matrix->cols + col] = val;
}
