#include "sequential.h"

int main(int argc, char const *argv[]) {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  train(28, 28, 30000, 10000, 10);
  gettimeofday(&end, NULL);
  printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
  return 0;
}
