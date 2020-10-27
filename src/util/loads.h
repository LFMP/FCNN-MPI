#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../util/stb_image.h"

void load_images(char* path, int width, int height, int count, float** data) {
  printf("Reading images\n");
  int channels = 1;
  for (int i = 0; i < count; i++) {
    char str[10];
    char aux[PATH_MAX];
    char filename[PATH_MAX];
    sprintf(str, "%d", i);
    strcpy(aux, path);
    strcat(aux, str);
    strcpy(filename, aux);
    strcat(filename, ".bmp");
    stbi_ldr_to_hdr_scale(1.0f);
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
  printf("Reading labels\n");
  char file_path[PATH_MAX];
  char filename[12];
  strcpy(filename, "labels.txt");
  strcpy(file_path, path);
  strcat(file_path, filename);
  FILE* fp = fopen(file_path, "r");
  char buffer[5];
  fgets(buffer, sizeof(buffer), fp);
  int label = 0, element = 0;
  label = atoi(buffer);
  while (label != -1 && element < count) {
    data[element] = (float*)malloc(qtd_class * sizeof(float));
    for (int i = 0; i < qtd_class; i++) {
      data[element][i] = 0;
    }
    data[element][label] = 1.0;
    char aux[5];
    fgets(aux, sizeof(aux), fp);
    if (aux != NULL) {
      label = atoi(aux);
    } else {
      label = -1;
    }
    element++;
  }
  fclose(fp);
}