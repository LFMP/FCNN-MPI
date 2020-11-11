# FCNN-MPI

Sequential and parallel C version, with MPI, of Fully Connected Neural Network (FCNN).

First download MNIST database [here](https://drive.google.com/file/d/1alUM0B0PtYLTw4TWUNMyByIxyZn81FYO/view?usp=sharing "MNIST Database").
Folders called "train" and "test" must be put in directory called ```Dataset/dataset1```, or you can change the variable ```DATASET``` in run.sh.

Just type `make` to see the magic happen.

To run sequential and parallel version and get speedup, run the command:

```bash
./run.sh N
```

Where `N` is the number of executions,.

The options for sequential and parallel programs can be found in the executables themselves.

```bash
cd src/sequential
./fcnn.out -h
```

For parallel:

```bash
cd src/parallel
./fcnn-mpi.out -h
```
