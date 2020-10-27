#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../util/layer.h"
#include "../util/loads.h"
#include "../util/maxs.h"
#include "../util/mse-loss.h"
#include "../util/relu.h"
#include "../util/sigmoid.h"

void train(int width, int hight, int train_size, int test_size, int qtd_class, int argc, char* argv[]) {
  int pcount, pid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &pcount);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // load images
  char train_folder[PATH_MAX], test_folder[PATH_MAX];
  strcpy(train_folder, "/home/luiz/GitProjects/FCNN-MPI/Datasets/dataset1/train/");
  strcpy(test_folder, "/home/luiz/GitProjects/FCNN-MPI/Datasets/dataset1/test/");
  float** train_images = (float**)malloc(train_size * sizeof(float*));
  float** train_labels = (float**)malloc(train_size * sizeof(float*));
  float** test_images = (float**)malloc(test_size * sizeof(float*));
  float** test_labels = (float**)malloc(test_size * sizeof(float*));
  if (pid == 0) {
    load_images(train_folder, width, hight, train_size, train_images);
    load_labels(train_folder, train_size, qtd_class, train_labels);
    load_images(test_folder, width, hight, test_size, test_images);
    load_labels(test_folder, test_size, qtd_class, test_labels);
  } else {
    for (int i = 0; i < train_size; i++) {
      train_images[i] = (float*)malloc(width * hight * sizeof(float*));
      train_labels[i] = (float*)malloc(qtd_class * sizeof(float*));
      if (i < test_size) {
        test_images[i] = (float*)malloc(width * hight * sizeof(float*));
        test_labels[i] = (float*)malloc(qtd_class * sizeof(float*));
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // create and init layers, relu, sigmoid and loss function
  for (int i = 0; i < train_size; i++) {
    MPI_Bcast(train_images[i], width * hight, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(train_labels[i], qtd_class, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  printf("Initializing network\n");
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
    printf("Training...\n");
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
    MPI_Barrier(MPI_COMM_WORLD);
    // printf("Epoch: %d\n", i);
    // printf("Train error: %lf", mse_sum / train_size);
    if (pid == 0) {
      mse_sum = 0;
      printf("Testing...\n");
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
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // create aux buffers
    int l1_size = l1->input_width * l1->output_width;
    int l2_size = l2->input_width * l2->output_width;
    int l3_size = l3->input_width * l3->output_width;
    float* aux_weights_l1 = (float*)malloc(l1_size * sizeof(float*));
    float* aux_weights_l2 = (float*)malloc(l2_size * sizeof(float*));
    float* aux_weights_l3 = (float*)malloc(l3_size * sizeof(float*));
    float* aux_biases_l1 = (float*)malloc(l1->output_width * sizeof(float*));
    float* aux_biases_l2 = (float*)malloc(l2->output_width * sizeof(float*));
    float* aux_biases_l3 = (float*)malloc(l3->output_width * sizeof(float*));
    // reduce to join results
    MPI_Allreduce(l1->weights, aux_weights_l1, l1_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(l1->biases, aux_biases_l1, l1->output_width, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(l2->weights, aux_weights_l2, l2_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(l2->biases, aux_biases_l2, l2->output_width, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(l3->weights, aux_weights_l3, l3_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(l3->biases, aux_biases_l3, l3->output_width, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // average
    for (int k = 0; k < l1_size; k++) {
      l1->weights[k] = aux_weights_l1[k] / (float)pcount;
    }
    for (int k = 0; k < l1->output_width; k++) {
      l1->biases[k] = aux_biases_l1[k] / (float)pcount;
    }
    for (int k = 0; k < l2_size; k++) {
      l2->weights[k] = aux_weights_l2[k] / (float)pcount;
    }
    for (int k = 0; k < l2->output_width; k++) {
      l2->biases[k] = aux_biases_l2[k] / (float)pcount;
    }
    for (int k = 0; k < l3_size; k++) {
      l3->weights[k] = aux_weights_l3[k] / (float)pcount;
    }
    for (int k = 0; k < l3->output_width; k++) {
      l3->biases[k] = aux_biases_l3[k] / (float)pcount;
    }
    free(aux_weights_l1);
    free(aux_weights_l2);
    free(aux_weights_l3);
    free(aux_biases_l1);
    free(aux_biases_l2);
    free(aux_biases_l3);
    // printf("Test error: %.2lf", mse_sum * 100 / test_size);
  }
  MPI_Finalize();
}