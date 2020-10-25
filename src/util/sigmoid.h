#include <math.h>
#include <stdlib.h>

typedef struct Sigmoid {
  float* inputRef;
  float* output;
  float* gradient;
  int width;
} sigmoid;

void init_sigmoid(sigmoid* s, int width) {
  s->width = width;
  s->output = (float*)malloc(width * sizeof(float));
  s->gradient = (float*)malloc(width * sizeof(float));
}

float calc_sigmoid(float in) {
  return 1.0 / (1.0 + exp(-1 * in));
}

float sigmoid_distance(float in) {
  return calc_sigmoid(in) * (1.0 - calc_sigmoid(in));
}

void sigmoid_forward(sigmoid* in, float* vec_in) {
  for (int i = 0; i < in->width; i++) {
    in->output[i] = calc_sigmoid(vec_in[i]);
  }
  in->inputRef = vec_in;
}

void sigmoid_backward(sigmoid* in, float* in_gradient) {
  for (int i = 0; i < in->width; i++) {
    in->gradient[i] = sigmoid_distance(in->inputRef[i]) * in_gradient[i];
  }
}