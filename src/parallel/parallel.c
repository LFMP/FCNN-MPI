#include "parallel.h"

int main(int argc, char *argv[]) {
  train(28, 28, 60000, 10000, 10, argc, argv);
  return 0;
}
