#!/bin/bash

DIR=$(pwd)
N=$1
parallel="src/parallel/fcnn-mpi.out"
sequential="src/sequential/fcnn.out"
PERF="sudo perf stat"
RUNF="-a -B -e cache-misses,cache-references,cycles,instructions,mem-loads,mem-stores,context-switches,power/energy-cores/,power/energy-ram/,power/energy-pkg/"
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
$PERF -r $N $RUNF ./$sequential -n 1000 -m 300 -d $DATASET >>sequential.1000.300.txt

$PERF -r $N $RUNF ./$sequential -n 9000 -m 3000 -d $DATASET >>sequential.9000.3000.txt

$PERF -r $N $RUNF ./$sequential -n 30000 -m 6000 -d $DATASET >>sequential.30000.6000.txt

$PERF -r $N $RUNF ./$sequential -n 60000 -m 10000 -d $DATASET >>sequential.60000.10000.txt

# executa paralelo
echo "executando paralelo"
for k in 2 4; do

  for ((i = 0; i < $N; ++i)); do
    mpirun -N $k $parallel -n 1000 -m 300 -d $DATASET >>parallel.$k.1000.300.txt
  done

  for ((i = 0; i < $N; ++i)); do
    mpirun -N $k $parallel -n 9000 -m 3000 -d $DATASET >>parallel.$k.9000.3000.txt
  done

  for ((i = 0; i < $N; ++i)); do
    mpirun -N $k $parallel -n 30000 -m 6000 -d $DATASET >>parallel.$k.30000.6000.txt
  done

  for ((i = 0; i < $N; ++i)); do
    mpirun -N $k $parallel -n 60000 -m 10000 -d $DATASET >>parallel.$k.60000.10000.txt
  done
done
