#include "parallel.h"

int main(int argc, char *argv[]) {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  train(28, 28, 3000, 1000, 10, 10, argc, argv);
  gettimeofday(&end, NULL);
  printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
  return 0;
}
