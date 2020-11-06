# In order to execute this "Makefile" just type "make"

OBJS	= parallel.o
SOURCE	= parallel.c
HEADER	= ../util/mse-loss.h ../util/relu.h ../util/sigmoid.h ../util/network.h parallel.h
OUT	= fcnn-mpi.out
CC	 = mpicc
FLAGS	 = -c -L/opt/mpiP-3.5/libmpiP.so -lmpi
LFLAGS	 = -lm
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