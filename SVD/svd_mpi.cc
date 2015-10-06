/**
Computes SVD for tall and skinny matrices ( rows >>> cols) using MPI (QR decomposition method)
the final matrices U would be distributed over all nodes, 
Matrices V and sigma would be avaiable on root (rank == 0)

Orthogonality checks can be done by computing L2 norm - |(U^T * U) - I_n |_2 of the resultant matrices
to check the correcness of the results 

To see sample output matrices in stdout or in file, simply do cout<<v; or fout<<v;
*/

#include<iostream>
#include<mpi.h>
#include<fstream>
#include<Eigen/Dense>
#include<Eigen/SVD>
#include<Eigen/Cholesky>

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
    unsigned long long position = 0;
     
	int N = ceil(rows/npes);
	
	double *recvbuf = new double[N*cols];           
	double *stack   = new double[npes*cols*cols];   	
	double *mat_q2  = new double[npes*cols*cols];   
	double *mat_u   = new double[cols*cols];    
	double *q2buf   = new double[cols*cols];
	
	RMatrixXd q1 (N, cols);
	RMatrixXd q2 (cols, cols);
	RMatrixXd res (cols, cols), res2 (cols, cols), u(cols, cols), v(cols, cols);
	
	double *data = NULL;
	
	if(myrank == 0) {	
	    data = new double[rows*cols];
	    ifstream fin (argv[3]);
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

	LLT<RMatrixXd> lltOfA(a.transpose()*a);
	res = lltOfA.matrixL().transpose(); 
	q1 = a * res.inverse(); 
	
	double* residual = res.data();

	MPI::COMM_WORLD.Gather(residual,cols*cols,MPI::DOUBLE,stack,cols*cols,MPI::DOUBLE,0);
	
	//task parallelism - matrix q2 and jacobi svd for small matrices being calculated simultaneously
	if(myrank == 0)
	{
		RMatrixXd s = Map<RMatrixXd>(stack,npes*cols,cols);
		LLT<RMatrixXd> lltOfS(s.transpose()*s);
		res2 = lltOfS.matrixL().transpose();       //fast - works on positive definite matrix
		cout<<"res : "<<res<<endl;
		double *residual2 = res2.data();
		MPI::COMM_WORLD.Send(residual2,cols*cols,MPI::DOUBLE,1,1);
		s = s*res2.inverse();
		mat_q2 = s.data();
	}
	if(myrank == 1)
	{	
		double *residual2 = new double[cols*cols];
		MPI::Status status;
		
		MPI::COMM_WORLD.Recv(residual2,cols*cols,MPI::DOUBLE,0,1,status);
		
		res2 = Map<RMatrixXd>(residual2,cols,cols);	
		
		JacobiSVD<RMatrixXd> svd(res2, ComputeThinU | ComputeThinV);
		v = svd.matrixV();
		u = svd.matrixU();
	
		delete[] residual2;		
		mat_u = u.data();	
	}
	
	MPI::COMM_WORLD.Scatter(mat_q2,cols*cols,MPI::DOUBLE,q2buf,cols*cols,MPI::DOUBLE,0);
	
	MPI::COMM_WORLD.Bcast(mat_u,cols*cols,MPI::DOUBLE,1);	
	
	q2 = Map<RMatrixXd>(q2buf,cols,cols);
 	q1 = q1*q2;
	u =  Map<RMatrixXd>(mat_u,cols,cols);	
	a = q1*u;  		//matrix U - distributed over all processors - compact SVD

	MPI::COMM_WORLD.Barrier();
	double end = MPI::Wtime();	

	double time = end - start;
	double max;

	MPI::COMM_WORLD.Reduce(&time, &max,1,MPI::DOUBLE,MPI::MAX,0);

	if(myrank == 0) { 
	    ofstream fout;
        fout.open("mpi_SVD_output.txt", ofstream::app | ofstream::out);		
	    fout<<"Time (in seconds) : "<<rows<<" "<<cols<<" : "<<max<<endl;
	}
	
	delete[] recvbuf;
	delete[] stack;
	delete[] q2buf;
	
	MPI::Finalize();
	return 0;
}
