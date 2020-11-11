#!/bin/bash

DIR=$(pwd)
N=$1
parallel="src/parallel/fcnn-mpi.out"
sequential="src/sequential/fcnn.out"
DATASET="$DIR/Datasets/dataset1/"
if [ -z "$N" ]; then
  echo "Eh necessario informar a quantidade de execucoes"
  exit 1
fi

# executa sequencial
SUMS1=0
SUMS2=0
SUMS3=0
SUMS4=0
echo "executando sequencial"
for ((i = 0; i < $N; ++i)); do
  SUMS1=$(bc <<<$SUMS1 + $(./$sequential -n 1000 -m 300 -d $DATASET))
done

for ((i = 0; i < $N; ++i)); do
  SUMS2=$(bc <<<$SUMS2 + $(./$sequential -n 9000 -m 3000 -d $DATASET))
done

for ((i = 0; i < $N; ++i)); do
  SUMS3=$(bc <<<$SUMS3 + $(./$sequential -n 30000 -m 6000 -d $DATASET))
done

for ((i = 0; i < $N; ++i)); do
  SUMS4=$(bc <<<$SUMS4 + $(./$sequential -n 60000 -m 10000 -d $DATASET))
done

# executa paralelo
echo "executando paralelo"
for k in 2 3 4; do
  ((SUMP = 0))
  for ((i = 0; i < $N; ++i)); do
    SUMP=$(bc <<<$SUMP + $(mpirun -np $k $parallel -n 1000 -m 300 -d $DATASET))
  done
  echo "print('Speedup para small e $k processos: ',($SUMS1/$N)/($SUMP/$N))" | python3

  ((SUMP = 0))
  for ((i = 0; i < $N; ++i)); do
    SUMP=$(bc <<<$SUMP + $(mpirun -np $k $parallel -n 9000 -m 3000 -d $DATASET))
  done
  echo "print('Speedup para medium e $k processos: ',($SUMS2/$N)/($SUMP/$N))" | python3

  ((SUMP = 0))
  for ((i = 0; i < $N; ++i)); do
    SUMP=$(bc <<<$SUMP + $(mpirun -np $k $parallel -n 30000 -m 6000 -d $DATASET))
  done
  echo "print('Speedup para large e $k processos: ',($SUMS3/$N)/($SUMP/$N))" | python3

  ((SUMP = 0))
  for ((i = 0; i < $N; ++i)); do
    SUMP=$(bc <<<$SUMP + $(mpirun -np $k $parallel -n 60000 -m 10000 -d $DATASET))
  done
  echo "print('Speedup para extra large e $k processos: ',($SUMS4/$N)/($SUMP/$N))" | python3
done
