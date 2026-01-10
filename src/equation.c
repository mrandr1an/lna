#include "equation.h"
#include "lna_primitive.h"

#include <math.h>

f32 cross_entropy(Matrix probabilities, const u64* labels) {
  f32 loss = 0.0f;
  for (u64 cur_class = 0; cur_class < probabilities.rows; cur_class++) {
    u64 correct_class = labels[cur_class];
    f32 correct_prob = mget(probabilities,cur_class,correct_class);
    if (correct_prob < eps) correct_prob = eps;
    loss += -logf(correct_prob);
  }

  return loss / (f32) probabilities.rows;
}

LnaOpStatus softmax(ml_arena* arena, Matrix* out, Matrix logits) {
  create_Matrix(arena,out,logits.rows,logits.cols);

  for (u64 cur_sample = 0; cur_sample < logits.rows; cur_sample++) {
    f32 row_max = mget(logits, cur_sample, 0);
    for (u64 cur_score = 0; cur_score < logits.cols; cur_score++) {
      f32 v = mget(logits,cur_sample,cur_score);
      if (v > row_max) row_max = v;
    }

    f32 row_sum = 0.0f;
    for (u64 cur_score = 0; cur_score < logits.cols; cur_score++) {
      f32 e =
	expf(mget(logits, cur_sample, cur_score) - row_max);
      mset(out,cur_sample,cur_score,e);
      row_sum += e;
    }

    f32 inv_sum = 1.0f / row_sum;
    for (u64 cur_score = 0; cur_score < logits.cols; cur_score++) {
      f32 val = mget(*out,cur_sample,cur_score) * inv_sum;
      mset(out,cur_sample,cur_score,val);
    }
  }

  return LNA_OK;
}

LnaOpStatus add_bias_rowwise_inplace(Matrix* logits, Matrix bias) {
  if (bias.rows != 1 || bias.cols != logits->cols) return INVALID_DIMENSIONS;
  for (u64 i = 0; i < logits->rows; i++) {
    for (u64 j = 0; j < logits->cols; j++) {
      f32 v = mget(*logits, i, j) + mget(bias, 0, j);
      mset(logits, i, j, v);
    }
  }
  return LNA_OK;
}

LnaOpStatus sum_rows(ml_arena* arena, Matrix* out_1xC, Matrix m) {
  create_Matrix(arena, out_1xC, 1, m.cols);

  // zero
  for (u64 j = 0; j < m.cols; j++) mset(out_1xC, 0, j, 0.0f);

  for (u64 i = 0; i < m.rows; i++) {
    for (u64 j = 0; j < m.cols; j++) {
      f32 v = mget(*out_1xC, 0, j) + mget(m, i, j);
      mset(out_1xC, 0, j, v);
    }
  }
  return LNA_OK;
}

// Compute gradient of the loss w.r.t. logits Z for
// softmax + categorical cross-entropy.
//
// On entry:
//   probs contains P = softmax(Z), shape (N x C)
//
// On exit:
//   probs is overwritten with dZ = DL/DZ = (P - Y) / N
//   where Y is the one-hot label matrix.
// with the notation DX/DY I mean the classic partial derivative.
LnaOpStatus softmax_xent_backward_inplace(Matrix* probs, const u64* labels) {
  // N batch size (probs->rows)
  // C number of classes (probs->cols)
  // P_{i,j} predicted probability for sample i, class j (mget(probs,i,j))
  // y_i label index for sample i (label[i])
  
  // 1/N
  f32 invN = 1.0f / (f32)probs->rows;

  // Subtract one-hot label: P_{i,y_i} -= 1
  for (u64 i = 0; i < probs->rows; i++) {
    u64 y = labels[i];
    if (y >= probs->cols) return INVALID_ELEMENT_TYPES; // or a better status
    mset(probs, i, y, mget(*probs, i, y) - 1.0f);
  }

  // (P_{i,j} - Y)/N
  for (u64 i = 0; i < probs->rows; i++) {
    for (u64 j = 0; j < probs->cols; j++) {
      //Probs -> dZ
      mset(probs, i, j, mget(*probs, i, j) * invN);
    }
  }

  return LNA_OK;
}

f32 softmax_regression_train_step(ml_arena* arena,
                                  Matrix X,
                                  const u64* labels,
                                  Matrix* W,
                                  Matrix* b,
                                  f32 learning_rate) {

  // X : independed variables, feature vector of height N.
  // W : N x C weights matrix.
  // C : Classes.
  // b : 1 x C bias row vector.
  // Z = X*W + b logits.
  // P = softmax(Z) : N x C, probabilities computed by softmax.
  // loss L = mean cross-entropy.
  
  //Start computing Z

  // logits = X*W
  Matrix logits;
  create_Matrix(arena, &logits, X.rows, W->cols);
  Matrix_mut_Matrix(arena, &logits, X, *W);

  // logits += b 
  add_bias_rowwise_inplace(&logits, *b);

  // Z = X*W + b is done.

  // P = softmax(Z)
  Matrix probs;
  softmax(arena, &probs, logits);
  
  // loss
  f32 loss = cross_entropy(probs, labels);

  // P is now dZ (gradient of the loss w.r.t Z)
  softmax_xent_backward_inplace(&probs, labels);

  // dW = (X^T)*dZ
  Matrix Xt;
  Matrix_transpose(arena, &Xt, X);
  Matrix dW;
  create_Matrix(arena, &dW, Xt.rows, probs.cols); // (D×C)
  Matrix_mut_Matrix(arena, &dW, Xt, probs);

  // db = sum_rows(dZ)
  Matrix db;
  sum_rows(arena, &db, probs);

  // SGD update: W -= lr*dW
  Matrix_mut_Scalar(&dW, learning_rate);
  Matrix_minus_Matrix_inplace(W, dW);

  // SGD update: b -= lr*db
  Matrix_mut_Scalar(&db, learning_rate);
  Matrix_minus_Matrix_inplace(b, db);

  return loss;
}
