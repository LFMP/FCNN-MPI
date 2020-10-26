#include <stdlib.h>

typedef struct Relu {
  float* inputRef;
  float* output;
  float* gradient;
  int width;
} relu;

void init_relu(relu* r, int width) {
  r->width = width;
  r->output = (float*)malloc(width * sizeof(float*));
  r->gradient = (float*)malloc(width * sizeof(float*));
}

float calc_relu(float in) {
  if (in >= 0) {
    return in;
  }
  return 0.1 * in;
}

float relu_distance(float in) {
  if (in >= 0) {
    return 1;
  }
  return 0.1;
}

void relu_forward(relu* in, float* vec_in) {
  for (int i = 0; i < in->width; i++) {
    in->output[i] = calc_relu(vec_in[i]);
  }
  in->inputRef = vec_in;
}

void relu_backward(relu* in, float* in_gradient) {
  for (int i = 0; i < in->width; i++) {
    in->gradient[i] = relu_distance(in->inputRef[i]) * in_gradient[i];
  }
}