CC = g++
MPICXX = mpicxx

SPECDEP=makefile
all: omp mpi hybrid

omp: svd_omp.cc $(SPECDEP)
	$(CC) -fopenmp svd_omp.cc -o omp  

mpi: svd_mpi.cc $(SPECDEP)
	$(MPICXX) svd_mpi.cc -o mpi

hybrid: svd_hybrid.cc $(SPECDEP)
	$(MPICXX) -fopenmp svd_hybrid.cc -o hybrid
