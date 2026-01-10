#ifndef LNA_ML_MODELS_H
#define LNA_ML_MODELS_H

#include "lna_arena.h"
#include "lna_number.h"
#define LNA_SOFTMAX_REGRESSION

#include "lna_primitive.h"

typedef struct {
  u64 number_of_samples;
  b8 labeled;
  Matrix batch;
  Matrix labels;
} Dataset;

#ifdef LNA_SOFTMAX_REGRESSION

 typedef struct {
   u64 number_of_features;
   u64 number_of_classes; 
   ml_arena* arena;
 } SoftmaxRegressionConfig;

 LnaOpStatus create_SR_conf(u64 feats,u64 classes);

 typedef struct {
   Matrix weights;
   Matrix biases;
   Matrix features;
   SoftmaxRegressionConfig conf;
 } SoftmaxRegressionModel;

 LnaOpStatus create_SR_Model(SoftmaxRegressionConfig conf);
 LnaOpStatus train_SR_Model(SoftmaxRegressionModel* model, Dataset ds, u64 steps,f32 learning_rate);
 LnaOpStatus infer_SR_Model(SoftmaxRegressionModel* model, Matrix in_batch,Matrix* out_batch);
#endif

#endif //LNA_ML_MODELS_H
