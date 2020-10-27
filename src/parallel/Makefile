# In order to execute this "Makefile" just type "make"

OBJS	= parallel.o
SOURCE	= parallel.c
HEADER	= ../util/mse-loss.h ../util/relu.h ../util/sigmoid.h ../util/network.h parallel.h
OUT	= fcnn-mpi.out
CC	 = mpicc
FLAGS	 = -c
LFLAGS	 = -lm
PERF = sudo systemd-inhibit perf stat
RUNF =  -r 11 -a -B -e cache-misses,cache-references,cycles,instructions,mem-loads,mem-stores,context-switches,power/energy-cores/,power/energy-ram/,power/energy-pkg/
# -g option enables debugging mode 
# -c flag generates object code for separate files

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

# create/compile the individual files >>separately<<
parallel.o: parallel.c
	$(CC) $(FLAGS) parallel.c 

# clean house
clean:
	rm -f $(OBJS) $(OUT)
	rm -f vgcore.*
	rm -f *.stats

# run the program
run: $(OUT)
	mpirun -np 2 ./$(OUT)
	# mpirun -np 4 ./$(OUT)
	# mpirun -np 6 ./$(OUT)
	# mpirun -np 8 ./$(OUT)

# compile program with debugging information
debug: $(OUT)
	valgrind $(OUT)

# run program with valgrind for errors
valgrind: $(OUT)
	valgrind ./$(OUT)

# run program with valgrind for leak checks
valgrind_leakcheck: $(OUT)
	valgrind --leak-check=full ./$(OUT)

# run program with valgrind for leak checks (extreme)
valgrind_extreme: $(OUT)
	valgrind --leak-check=full --show-leak-kinds=all --leak-resolution=high --track-origins=yes --vgdb=yes ./$(OUT)
