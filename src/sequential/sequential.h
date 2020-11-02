#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "../util/layer.h"
#include "../util/loads.h"
#include "../util/maxs.h"
#include "../util/mse-loss.h"
#include "../util/relu.h"
#include "../util/sigmoid.h"

void train(int width, int hight, int train_size, int test_size, int qtd_class, int epochs, char* train_folder, char* test_folder) {
  // load images
  float** train_images = (float**)malloc(train_size * sizeof(float*));
  float** train_labels = (float**)malloc(train_size * sizeof(float*));
  float** test_images = (float**)malloc(test_size * sizeof(float*));
  float** test_labels = (float**)malloc(test_size * sizeof(float*));
  load_images(train_folder, width, hight, train_size, train_images);
  load_labels(train_folder, train_size, qtd_class, train_labels);
  load_images(test_folder, width, hight, test_size, test_images);
  load_labels(test_folder, test_size, qtd_class, test_labels);
  // create and init layers, relu, sigmoid and loss function
  //printf("Initializing network\n");
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
  layer_create(l2, 1024, 512);
  layer_create(l3, 512, 256);
  layer_create(l4, qtd_class, qtd_class);
  layer_create(l5, qtd_class, qtd_class);
  layer_initialize(l1, 0.01, 0);
  layer_initialize(l2, 0.01, 0);
  layer_initialize(l3, 0.01, 0);
  layer_initialize(l4, 0.01, 0);
  layer_initialize(l5, 2, -1);
  init_relu(r1, 1024);
  init_relu(r2, 512);
  init_relu(r3, 256);
  init_relu(r4, qtd_class);
  init_sigmoid(s1, qtd_class);
  init_mse(mse, qtd_class);
  // aux vars
  int sample = 0;
  double mse_sum = 0;
  // train and test process
  for (int i = 0; i < epochs; i++) {
    mse_sum = 0;
    //printf("Training...\n");
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
    //printf("Epoch: %d\n", i);
    //printf("Train error: %lf", mse_sum / train_size);
    mse_sum = 0;
    //printf("Testing...\n");
    for (int j = 0; j < test_size; j++) {
      sample = i;
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
    // printf("Test error: %.2lf", mse_sum * 100 / test_size);
  }
}
