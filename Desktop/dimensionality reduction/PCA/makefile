CC = g++
MPICXX = mpicxx

SPECDEP=makefile
all: omp mpi hybrid

omp: pca_omp.cc $(SPECDEP)
	$(CC) -fopenmp pca_omp.cc -o omp  

mpi: pca_mpi.cc $(SPECDEP)
	$(MPICXX) pca_mpi.cc -o mpi

hybrid: pca_hybrid.cc $(SPECDEP)
	$(MPICXX) -fopenmp pca_hybrid.cc -o hybrid
