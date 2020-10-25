#include <stdlib.h>

typedef struct Network {
  int input_width = 0;
  int output_width = 0;
  float* output = NULL;
  float* gradient = NULL;
  float* inputRef = NULL;

  float* weights = NULL;
  float* weights_errs = NULL;
  float* biases = NULL;
  float* biases_err = NULL;
  float learningRate = 0.01;
} network;

void network_create(network* net, int input_len, int output_len) {
  net->input_width = input_len;
  net->output_width = output_len;
  net->output = (float*)malloc(input_len * sizeof(float));
  net->gradient = (float*)malloc(input_len * sizeof(float));
}

void network_initialize(network* net, float mult, float offset) {
  int count = net->input_width * net->output_width;
  net->weights = (float*)malloc(count * sizeof(float));
  net->weights_errs = (float*)malloc(count * sizeof(float));
  net->biases = (float*)malloc(net->output_width * sizeof(float));
  net->biases_err = (float*)malloc(net->output_width * sizeof(float));

  for (int i = 0; i < count; i++) {
    net->weights[i] = ((float)rand() / RAND_MAX) * mult + offset;
    net->weights_errs[i] = 0;
  }

  for (int i = 0; i < net->output_width; i++) {
    net->biases[i] = ((float)rand() / RAND_MAX) * 0.01;
    net->biases_err[i] = 0;
  }
}

void network_forward(network* net, float* in) {
  for (int i = 0; i < net->output_width; i++) {
    float sum = 0;
    for (int k = 0; k < net->input_width; k++) {
      sum += in[k] * net->weights[net->input_width * i + k];
    }
    net->output[i] = sum + net->biases[i];
  }
  net->inputRef = in;
}

void network_backward(network* net, float* in_gradient) {
  for (int i = 0; i < net->input_width; i++) {
    float sum = 0;
    for (int k = 0; k < net->output_width; k++) {
      net->weights_errs[net->input_width * k + i] += net->inputRef[i] * in_gradient[k];
      sum += net->weights[net->input_width * k + i] * in_gradient[k];
    }
    net->gradient[i] = sum;
  }
  for (int i = 0; i < net->output_width; i++) {
    net->biases_err[i] += in_gradient[i];
  }
}

void network_update_w(network* net) {
  for (int i = 0; i < net->input_width * net->output_width; i++) {
    net->weights[i] -= net->weights_errs[i] * net->learningRate;
    net->weights_errs[i] = 0;
  }
  for (int i = 0; i < net->output_width; i++) {
    net->biases[i] -= net->biases_err[i] * net->learningRate;
    net->biases[i] = 0;
  }
}