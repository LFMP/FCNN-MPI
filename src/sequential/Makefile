# In order to execute this "Makefile" just type "make"

OBJS	= sequential.o
SOURCE	= sequential.c
HEADER	= ../util/mse-loss.h ../util/relu.h ../util/sigmoid.h ../util/network.h sequential.h
OUT	= fcnn.out
CC	 = gcc
FLAGS	 = -O3 -g3 -c
LFLAGS	 = -lm
PERF = sudo systemd-inhibit perf stat
RUNF =  -r 11 -a -B -e cache-misses,cache-references,cycles,instructions,mem-loads,mem-stores,context-switches,power/energy-cores/,power/energy-ram/,power/energy-pkg/
# -g option enables debugging mode 
# -c flag generates object code for separate files

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

# create/compile the individual files >>separately<<
sequential.o: sequential.c
	$(CC) $(FLAGS) sequential.c 

# clean house
clean:
	rm -f $(OBJS) $(OUT)
	rm -f vgcore.*
	rm -f *.stats