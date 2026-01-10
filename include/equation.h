#ifndef LNA_EQUATION_H
#define LNA_EQUATION_H

#include "lna_primitive.h"

f32 cross_entropy(Matrix probabilities, const u64* labels);

LnaOpStatus softmax(ml_arena* arena,Matrix* out,Matrix logits);

LnaOpStatus add_bias_rowwise_inplace(Matrix* logits, Matrix bias);
LnaOpStatus sum_rows(ml_arena* arena, Matrix* out_1xC, Matrix m);
LnaOpStatus softmax_xent_backward_inplace(Matrix* probs, const u64* labels);
f32 softmax_regression_train_step(ml_arena* arena,
                                  Matrix X,
                                  const u64* labels,
                                  Matrix* W,
                                  Matrix* b,
                                  f32 learning_rate);
#endif //LNA_EQUATION_H
