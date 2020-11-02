#include "sequential.h"

int main(int argc, char *argv[]) {
  struct timeval start, end;
  char train_path[PATH_MAX];
  char test_path[PATH_MAX];
  int opt = -1, height = 28, width = 28, train_size = 60000, test_size = 10000, epochs = 16, num_class = 10;
  while ((opt = getopt(argc, argv, "w:n:m:e:f:c:d:h")) != -1) {
    switch (opt) {
      case 'w':
        width = strtoul(optarg, NULL, 0);
        height = width;
        break;
      case 'n':
        train_size = strtoul(optarg, NULL, 0);
        break;
      case 'm':
        test_size = strtoul(optarg, NULL, 0);
        break;
      case 'e':
        epochs = strtoul(optarg, NULL, 0);
        break;
      case 'c':
        num_class = strtoul(optarg, NULL, 0);
        break;
      case 'd':
        strcpy(train_path, optarg);
        strcat(train_path, "train/");
        strcpy(test_path, optarg);
        strcat(test_path, "test/");
        break;
      case 'h':
        printf("Para este programa estao disponiveis as seguintes opcoes:\n");
        printf("-w <parameter>\t: Tamanho das imagens (28x28 por padrao)\n");
        printf("-n <parameter>\t: Quantidade de amostras para treino (60000 por padrao)");
        printf("-m <parameter>\t: Quantidade de amostras para teste (10000 por padrao)");
        printf("-e <parameter>\t: Quantidade de epochs (16 por padrao)");
        printf("-c <parameter>\t: Quantidade de classes (10 por padrao)");
        printf("-d <parameter>\t: Path do dataset");
        printf("-h <parameter>\t: Esta ajuda");
        exit(0);
    }
  }
  if ((argc == 0) && (opt == (-1))) {
    printf("Para este programa estao disponiveis as seguintes opcoes:\n");
    printf("-w <parameter>\t: Tamanho das imagens (28x28 por padrao)\n");
    printf("-n <parameter>\t: Quantidade de amostras para treino (60000 por padrao)");
    printf("-m <parameter>\t: Quantidade de amostras para teste (10000 por padrao)");
    printf("-e <parameter>\t: Quantidade de epochs (16 por padrao)");
    printf("-c <parameter>\t: Quantidade de classes (10 por padrao)");
    printf("-d <parameter>\t: Path do dataset");
    printf("-h <parameter>\t: Esta ajuda");
    exit(0);
  }
  gettimeofday(&start, NULL);
  train(width, height, train_size, test_size, num_class, epochs, train_path, test_path);
  gettimeofday(&end, NULL);
  printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
  return 0;
}
