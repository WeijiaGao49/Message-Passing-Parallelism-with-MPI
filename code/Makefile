CC=gcc

MPI=-DMPI
MPICC = mpicc


DEBUG=0
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS= -lm
DDIR = ./data

CFILES = crun.c graph.c simutil.c sim.c rutil.c cycletimer.c
SEQ_CFILES = crun_seq.c graph_seq.c simutil_seq.c sim_seq.c rutil.c cycletimer.c
HFILES = crun.h rutil.h cycletimer.h
SEQ_HFILES = crun_seq.h rutil.h cycletimer.h

all: crun-seq crun-mpi


crun-seq: $(SEQ_CFILES) $(HFILES) 
	$(CC) $(CFLAGS) -o crun-seq $(SEQ_CFILES) $(LDFLAGS)

crun-mpi: $(CFILES) $(SEQ_HFILES)
	$(MPICC) $(CFLAGS) $(MPI) -o crun-mpi $(CFILES) $(LDFLAGS)

demo1: grun.py
	@echo "Running Python simulator with text visualization.  Synchronous mode."
	./grun.py -g data/g-t012x012.gph -r data/r-012x012-r4.rats -n 20 -u s -v a -p 0.3

demo2: grun.py
	@echo "Running Python simulator with heat-map visualization.  Synchronous mode"
	@echo "Note the oscillatory behavior"
	./grun.py -g data/g-t036x036.gph -r data/r-036x036-r10.rats -u s -n 20 -v h -p 0.3

demo3: grun.py
	@echo "Running Python simulator with heat-map visualization.  Rat-order mode"
	@echo "Note the smooth transitions"
	./grun.py -g data/g-t036x036.gph -r data/r-036x036-r10.rats -u r -n 20 -v h -p 0.3

demo4: grun.py
	@echo "Running Python simulator with heat-map visualization.  Batch mode"
	@echo "Note the smooth transitions"
	./grun.py -g data/g-t036x036.gph -r data/r-036x036-r10.rats -u b -n 20 -v h -p 0.3

demo5: crun-seq grun.py
	@echo "Running C simulator using Python simulator as visualizer"
	./crun-seq -g data/g-t036x036.gph -r data/r-036x036-r10.rats -n 20 | ./grun.py -d -v h -p 0.3

demo6: crun-seq grun.py
	@echo "Running on a 180x180 tiled graph"
	./crun-seq -g data/g-t180x180.gph -r data/r-180x180-r32.rats -n 75 | ./grun.py -d -v h -p 0.02

demo7: crun-seq grun.py
	@echo "Running on a 180x180 horizontal graph"
	./crun-seq -g data/g-h180x180.gph -r data/r-180x180-r32.rats -n 75 | ./grun.py -d -v h -p 0.02

demo8: crun-seq grun.py
	@echo "Running on a 180x180 parquet graph"
	./crun-seq -g data/g-p180x180.gph -r data/r-180x180-r32.rats -n 75 | ./grun.py -d -v h -p 0.02

demo9: crun-seq grun.py
	@echo "Running on a 180x180 parquet graph, with rats initially along diagonal"
	./crun-seq -g data/g-p180x180.gph -r data/r-180x180-d32.rats -n 75 | ./grun.py -d -v h -p 0.02

demo10: crun-seq grun.py
	@echo "Running on a 180x180 parquet graph, with rats initially distributed uniformly"
	./crun-seq -g data/g-p180x180.gph -r data/r-180x180-u32.rats -n 75 | ./grun.py -d -v h -p 0.02

clean:
	rm -f *~ *.pyc
	rm -rf *.dSYM
	rm -rf regression-cache check
	rm -f crun crun-seq crun-mpi
