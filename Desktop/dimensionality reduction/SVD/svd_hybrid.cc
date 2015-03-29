/**
Computes SVD for Long & skinny matrices (rows >> cols) using QR decomposition
This is a hybrid implementation which uses MPI in combination with OpenMP
Uses Eigen C++ library for linear algebra computations
Note : if a matrix with (rows <= cols) is given as input then result is indeterminate

Orthogonality checks can be done by computing L2 norm - |(U^T * U) - I_n |_2 of the resultant matrices
to check the correcness of the results 

To see sample output matrices in stdout or in file, simply do cout<<v; or fout<<v;
*/

#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

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
    unsigned long long position = 0;
    
	double *recvbuf = new double[N*cols]; 
	double *stack   = new double[nthreads*npes*cols*cols]; 
	double *mat_q2  = new double[npes*cols*cols]; 
	double *mat_u   = new double[cols*cols]; 
	double *q2buf   = new double[cols*cols];
	
	RMatrixXd q1 (N,cols);
	RMatrixXd q2 (cols,cols);
	RMatrixXd res (cols,cols), res2 (cols,cols), u(cols,cols);
	
	int rows2 = ceil(N/nthreads);

	RMatrixXd a_local (rows2,cols);
	RMatrixXd st (nthreads*cols,cols);
	
	double *data = NULL;
	 
	if(myrank == 0) {	
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
	//Scatter N rows to each worker from master
	MPI::COMM_WORLD.Scatter(data,N*cols,MPI::DOUBLE,recvbuf,N*cols,MPI::DOUBLE,0);
	
	RMatrixXd a = Map<RMatrixXd>(recvbuf,N,cols);	

	MPI::COMM_WORLD.Barrier();
	double start = MPI::Wtime();	
	
	#pragma omp parallel num_threads(nthreads) shared(a,q1,res,st) private(a_local)
	{ 
		a_local = a.block(omp_get_thread_num()*rows2,0,rows2,cols);
		LLT<RMatrixXd> lltOfA(a_local.transpose()*a_local);
		res = lltOfA.matrixL().transpose(); 
		q1.block(omp_get_thread_num()*rows2,0,rows2,cols) = a_local * res.inverse(); 
		st.block(omp_get_thread_num()*cols,0,cols,cols) = res;
	}

	double* local_stack = st.data();
	MPI::COMM_WORLD.Gather(local_stack,nthreads*cols*cols,MPI::DOUBLE,stack,nthreads*cols*cols,MPI::DOUBLE,0);
	//task parallelism - matrix q2 and svd being calculated parallely
	if(myrank == 0)
	{
		RMatrixXd s (nthreads*npes*cols,cols);
		s = Map<RMatrixXd>(stack,nthreads*npes*cols,cols);
		
		LLT<RMatrixXd> lltOfS(s.transpose()*s);
		res2 = lltOfS.matrixL().transpose(); //fast - works on positive definite matrix
	   
		double *residual2 = res2.data();
		MPI::COMM_WORLD.Send(residual2,cols*cols,MPI::DOUBLE,1,1); //tag = 1

		s = s*res2.inverse(); // q2temp is of size N*cols*cols

		mat_q2 = s.data();
	}
	
	if(myrank == 1)
	{	
		double *residual2 = (double*)malloc(sizeof(double)*cols*cols);

		MPI::Status status;
		MPI::COMM_WORLD.Recv(residual2,cols*cols,MPI::DOUBLE,0,1,status);
		res2 = Map<RMatrixXd>(residual2,cols,cols);	
		//Calculate svd
		JacobiSVD<RMatrixXd> svd(res2,ComputeThinU | ComputeThinV);
        
		u = svd.matrixU();
		mat_u = u.data();
	}
	MPI::COMM_WORLD.Scatter(mat_q2,cols*cols,MPI::DOUBLE,q2buf,cols*cols,MPI::DOUBLE,0); //Needn't scatter entire matrix q2	
	
	MPI::COMM_WORLD.Bcast(mat_u,cols*cols,MPI::DOUBLE,1);		
	
	q2 = Map<RMatrixXd>(q2buf,cols,cols);
	u =  Map<RMatrixXd>(mat_u,cols,cols);
 
	#pragma omp parallel num_threads(nthreads) shared(q2,q1,u) private(a_local)
	{
		q1.block(omp_get_thread_num()*rows2,0,rows2,cols) = q1.block(omp_get_thread_num()*rows2,0,rows2,cols)*q2;	
		a_local = q1.block(omp_get_thread_num()*rows2,0,rows2,cols)*u;  //reduced matrix a - distributed over all processors
	}

	MPI::COMM_WORLD.Barrier();
	double end = MPI::Wtime();	
	double time = end - start;
	
	double max;
	
	MPI::COMM_WORLD.Reduce(&time, &max,1,MPI::DOUBLE,MPI::MAX,0);
	
	if(myrank == 0) { 
	    ofstream fout;
        fout.open("hybrid_SVD_output.txt", ofstream::app | ofstream::out);		
	    fout<<"Time (in sec) "<<rows<<" "<<cols<<" "<<nthreads<<" : "<<max<<endl;
	}
	
	delete[] recvbuf; 
	delete[] stack;  
	delete[] q2buf;
	
	MPI::Finalize();
	return 0;
}
