#include <stdlib.h>

typedef struct MSEloss {
  float* output;
  float* gradient;
  float err_sum;
  int width;
} mseloss;

void init_mse(mseloss* mse, int width) {
  mse->width = width;
  mse->err_sum = 0;
  mse->output = (float*)malloc(width * sizeof(float));
  mse->gradient = (float*)malloc(width * sizeof(float));
}

float mse_square_err(float out, float expected) {
  return (((out - expected) * (out - expected))) * 0.5;
}

float se_distance(float out, float expected) {
  return out - expected;
}

void mse_loss_calc(mseloss* in, float* vect_in, float* vect_expected) {
  in->err_sum = 0;
  for (int i = 0; i < in->width; i++) {
    in->output[i] = mse_square_err(vect_in[i], vect_expected[i]);
    in->gradient[i] = se_distance(vect_in[i], vect_expected[i]);
    in->err_sum += in->output[i];
  }
}