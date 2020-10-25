#include <stdlib.h>

typedef struct Layer {
  int input_width;
  int output_width;
  float* output;
  float* gradient;
  float* inputRef;

  float* weights;
  float* weights_errs;
  float* biases;
  float* biases_err;
  float learningRate;
} layer;

void layer_create(layer* l, int input_len, int output_len) {
  l->input_width = input_len;
  l->output_width = output_len;
  l->learningRate = 0.01;
  l->output = (float*)malloc(input_len * sizeof(float));
  l->gradient = (float*)malloc(input_len * sizeof(float));
}

void layer_initialize(layer* l, float mult, float offset) {
  int count = l->input_width * l->output_width;
  l->weights = (float*)malloc(count * sizeof(float));
  l->weights_errs = (float*)malloc(count * sizeof(float));
  l->biases = (float*)malloc(l->output_width * sizeof(float));
  l->biases_err = (float*)malloc(l->output_width * sizeof(float));

  for (int i = 0; i < count; i++) {
    l->weights[i] = ((float)rand() / RAND_MAX) * mult + offset;
    l->weights_errs[i] = 0;
  }

  for (int i = 0; i < l->output_width; i++) {
    l->biases[i] = ((float)rand() / RAND_MAX) * 0.01;
    l->biases_err[i] = 0;
  }
}

void layer_forward(layer* l, float* in) {
  for (int i = 0; i < l->output_width; i++) {
    float sum = 0;
    for (int k = 0; k < l->input_width; k++) {
      sum += in[k] * l->weights[l->input_width * i + k];
    }
    l->output[i] = sum + l->biases[i];
  }
  l->inputRef = in;
}

void layer_backward(layer* l, float* in_gradient) {
  for (int i = 0; i < l->input_width; i++) {
    float sum = 0;
    for (int k = 0; k < l->output_width; k++) {
      l->weights_errs[l->input_width * k + i] += l->inputRef[i] * in_gradient[k];
      sum += l->weights[l->input_width * k + i] * in_gradient[k];
    }
    l->gradient[i] = sum;
  }
  for (int i = 0; i < l->output_width; i++) {
    l->biases_err[i] += in_gradient[i];
  }
}

void layer_update_w(layer* l) {
  for (int i = 0; i < l->input_width * l->output_width; i++) {
    l->weights[i] -= l->weights_errs[i] * l->learningRate;
    l->weights_errs[i] = 0;
  }
  for (int i = 0; i < l->output_width; i++) {
    l->biases[i] -= l->biases_err[i] * l->learningRate;
    l->biases[i] = 0;
  }
}