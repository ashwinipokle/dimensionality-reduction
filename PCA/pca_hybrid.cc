/**
PCA hybrid implementation for tall and skinny matrices (rows >> columns)
uses MPI and OpenMP for utilizing multi-node + multi-core clusters 
*/
#include<iostream>
#include<mpi.h>
#include<fstream>
#include<omp.h>
#include<Eigen/Dense>
#include<Eigen/Cholesky>
#include<Eigen/SVD>
#include<Eigen/QR>

using namespace std;
using namespace Eigen;

typedef Matrix<double,Dynamic,Dynamic,RowMajor> RMatrixXd;

int main(int argc, char *argv[])
{
	int npes, myrank;
	MPI::Init(argc, argv);
	myrank = MPI::COMM_WORLD.Get_rank();
	npes = MPI::COMM_WORLD.Get_size();
	
	unsigned long rows = atol(argv[1]);
	int cols = atoi(argv[2]);

	int N = ceil(rows/npes);
	int nthreads = atoi(argv[3]);

	double *recvbuf = new double[N*cols]; 
	double *stack   = new double[nthreads*npes*cols*cols]; 	
	double *mat_q2  = new double[npes*cols*cols]; 
	double *mat_v   = new double[cols*cols]; 	
	double *q2buf   = new double[cols*cols];
	
	RMatrixXd a (N,cols), q1 (N,cols);
	RMatrixXd q2 (cols,cols);
	RMatrixXd res (cols,cols), res2 (cols,cols), v(cols,cols);
	VectorXd singular;
	
	double *data = NULL;
	 
	if(myrank == 0) {
	    unsigned long long position = 0;	
	    data = new double[rows*cols];
	    ifstream fin (argv[4]);
        if(fin.is_open()) {
            while(!fin.eof() && position < (rows*cols)) {
                fin >> data[position];
                position++;
            }
        } else {
            cout<<"File not found at path given. Enter correct file name "<<endl;
            exit(0);
        }
        fin.close();
	}
	//Scatter N rows to each slave from master
	MPI::COMM_WORLD.Scatter(data,N*cols,MPI::DOUBLE,recvbuf,N*cols,MPI::DOUBLE,0);
		
	a = Map<RMatrixXd>(recvbuf,N,cols);	
	
	MPI::COMM_WORLD.Barrier();
	double start = MPI::Wtime();

	int rows2 = ceil(N/nthreads);

	RMatrixXd a_local (rows2,cols);
	RMatrixXd st (nthreads*cols,cols);
    
    
	//mean adjust data for pca
	a = a.rowwise() - a.colwise().mean();		
	a = a/sqrt(rows-1);
		
	#pragma omp parallel num_threads(nthreads) shared(a,res,st) private(a_local)
	{ 
		a_local = a.block(omp_get_thread_num()*rows2,0,rows2,cols);
		LLT<RMatrixXd> lltOfA(a_local.transpose()*a_local);
		res = lltOfA.matrixL().transpose(); 
		st.block(omp_get_thread_num()*cols,0,cols,cols) = res;
	}

	double* local_stack = st.data();
	MPI::COMM_WORLD.Gather(local_stack,nthreads*cols*cols,MPI::DOUBLE,stack,nthreads*cols*cols,MPI::DOUBLE,0);
	
	if(myrank == 0)
	{
		RMatrixXd s (nthreads*npes*cols,cols);
		s = Map<RMatrixXd>(stack,nthreads*npes*cols,cols);
		
		LLT<RMatrixXd> lltOfS(s.transpose()*s);
		res2 = lltOfS.matrixL().transpose(); //fast - works on positive semidefinite matrix
	   
		//Calculate small svd
		JacobiSVD<RMatrixXd> svd(res2,ComputeThinV); //Compute only V and singular values
		v = svd.matrixV();
		singular = svd.singularValues();
		singular = singular.array().square();	    //variance associated with each eigen value				
		
		mat_v = v.data();
	}
	
	MPI::COMM_WORLD.Bcast(mat_v,cols*cols,MPI::DOUBLE,0);	
		
	v =  Map<RMatrixXd>(mat_v,cols,cols);
	RMatrixXd proj = a * v.transpose();		//transformed data ; is this valid every time?
		
	MPI::COMM_WORLD.Barrier();
	double end = MPI::Wtime();	
	double time = end - start;
	double max;
	MPI::COMM_WORLD.Reduce(&time, &max,1,MPI::DOUBLE,MPI::MAX,0);
	
	if(myrank == 0) { 
	    ofstream fout;
        fout.open("hybrid_PCA_output.txt", ofstream::app | ofstream::out);		
	    fout<<"Time (in sec) "<<rows<<" "<<cols<<" "<<nthreads<<" : "<<max<<endl;
	}
	
	delete[] recvbuf; 
	delete[] stack;  
		
	MPI::Finalize();
	return 0;
}
