#include <stdio.h>
#include <string.h>

#include "../util/layer.h"
#include "../util/mse-loss.h"
#include "../util/relu.h"
#include "../util/sigmoid.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../util/stb_image.h"

void load_images(char* path, int width, int height, int count, float** data) {
  int channels = 1;
  for (int i = 0; i < count; i++) {
    stbi_ldr_to_hdr_scale(1.0f);
    char* aux = strcat(path, (char)i);
    char* filename = strcat(aux, ".png");
    float* image = stbi_loadf(filename, &width, &height, &channels, STBI_grey);
    data[i] = (float*)malloc(width * height * sizeof(float));
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        data[i][y * width + x] = image[y * width + x];
      }
    }
    stbi_image_free(image);
  }
}

void load_labels(char* path, int count, int qtd_class, float** data) {
  FILE* fp = fopen(strcat(path, "labels.txt"), 'r');
  char buffer[5];
  int label = atoi(fgets(buffer, sizeof(buffer), fp)), element = 0;
  while (label != NULL && element < count) {
    data[element] = (float*)malloc(qtd_class * sizeof(float));
    for (int i = 0; i < qtd_class; i++) {
      data[element][i] = 0;
    }
    data[element][label] = 1.0;
    char aux[5] = fgets(buffer, sizeof(buffer), fp);
    if (aux != NULL) {
      label = atoi(aux);
    } else {
      label = NULL;
    }

    element++;
    free(aux);
  }
  fclose(fp);
}

int find_max(float* vect_in, int len) {
  int max = 0;
  for (int i = 0; i < len; i++) {
    if (vect_in[max] < vect_in[i]) {
      max = i;
    }
  }
  return max;
}

void train(int width, int hight, int train_size, int test_size, int qtd_class) {
  // load images
  float** train_images = (float**)malloc(train_size * sizeof(float*));
  float** train_labels = (float**)malloc(train_size * sizeof(float*));
  float** test_images = (float**)malloc(test_size * sizeof(float*));
  float** test_labels = (float**)malloc(test_size * sizeof(float*));
  load_images("../../Datasets/dataset1/train/", width, hight, train_size, train_images);
  load_labels("../../Datasets/dataset1/train/", train_size, qtd_class, train_labels);
  load_images("../../Datasets/dataset1/test/", width, hight, train_size, test_images);
  load_labels("../../Datasets/dataset1/test/", test_size, qtd_class, test_labels);
  // create and init layers, relu, sigmoid and loss function
  layer* l1 = (layer*)malloc(sizeof(layer));
  layer* l2 = (layer*)malloc(sizeof(layer));
  layer* l3 = (layer*)malloc(sizeof(layer));
  layer* l4 = (layer*)malloc(sizeof(layer));
  layer* l5 = (layer*)malloc(sizeof(layer));
  relu* r1 = (relu*)malloc(sizeof(relu));
  relu* r2 = (relu*)malloc(sizeof(relu));
  relu* r3 = (relu*)malloc(sizeof(relu));
  relu* r4 = (relu*)malloc(sizeof(relu));
  sigmoid* s1 = (sigmoid*)malloc(sizeof(sigmoid));
  mseloss* mse = (mseloss*)malloc(sizeof(mseloss));
  layer_create(l1, width * hight, 1024);
  layer_create(l2, 512, 512);
  layer_create(l3, 256, 256);
  layer_create(l4, qtd_class, qtd_class);
  layer_create(l5, qtd_class, qtd_class);
  layer_initialize(l1, 0.01, 0);
  layer_initialize(l2, 0.01, 0);
  layer_initialize(l3, 0.01, 0);
  layer_initialize(l4, 0.01, 0);
  layer_initialize(l5, 0.01, 0);
  init_relu(r1, 1024);
  init_relu(r2, 256);
  init_relu(r3, 256);
  init_relu(r4, 256);
  init_sigmoid(s1, qtd_class);
  init_mse(mse, qtd_class);
  // aux vars
  int sample = 0;
  double mse_sum = 0;
  // train and test process
  for (int i = 0; i < 20; i++) {
    mse_sum = 0;
    for (int j = 0; j < train_size; j++) {
      // chose sample
      sample = rand() % train_size;
      // foward information
      layer_forward(l1, train_images[sample]);
      relu_forward(r1, l1->output);
      layer_forward(l2, l1->output);
      relu_forward(r2, l2->output);
      layer_forward(l3, l2->output);
      relu_forward(r3, l3->output);
      layer_forward(l4, l3->output);
      relu_forward(r4, l4->output);
      layer_forward(l5, l4->output);
      sigmoid_forward(s1, l5->output);
      // calculate mse
      mse_loss_calc(mse, s1->output, train_labels[sample]);
      // backward information
      sigmoid_backward(s1, mse->gradient);
      layer_backward(l5, s1->gradient);
      relu_backward(r4, l5->gradient);
      layer_backward(l4, r4->gradient);
      relu_backward(r3, l4->gradient);
      layer_backward(l3, r3->gradient);
      relu_backward(r2, l3->gradient);
      layer_backward(l2, r2->gradient);
      relu_backward(r1, l2->gradient);
      layer_backward(l1, r1->gradient);
      // update weigths
      layer_update_w(l1);
      layer_update_w(l2);
      layer_update_w(l3);
      layer_update_w(l4);
      layer_update_w(l5);
      mse_sum += mse->err_sum;
    }
    printf("Epoch: %d\n", i);
    printf("Train error: %lf", mse_sum / train_size);
    mse_sum = 0;
    for (int j = 0; j < test_size; j++) {
      layer_forward(l1, test_images[sample]);
      relu_forward(r1, l1->output);
      layer_forward(l2, l1->output);
      relu_forward(r2, l2->output);
      layer_forward(l3, l2->output);
      relu_forward(r3, l3->output);
      layer_forward(l4, l3->output);
      relu_forward(r4, l4->output);
      layer_forward(l5, l4->output);
      sigmoid_forward(s1, l5->output);
      if (find_max(test_labels[sample], qtd_class) != find_max(s1->output, qtd_class)) {
        mse_sum += 1.0;
      }
    }
    printf("Test error: %.2lf", mse_sum * 100 / test_size);
  }
}