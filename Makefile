all:
	# make -C docs/ all
	make -C src/ all 

sequencial:
	make -C src/sequential/ all

paralelo:
	make -C src/parallel/ all

clean:
	make -C docs/ clean
	make -C src/ clean