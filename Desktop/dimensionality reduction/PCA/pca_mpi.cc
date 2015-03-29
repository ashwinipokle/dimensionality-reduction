/**
Computes PCA for tall and skinny matrices ( rows >>> cols) using MPI (QR decomposition method)
 
V and sigma (or variance) would be avaiable on root (rank == 0)

The principal components are columns of matrix v where the first column is the principal vector and 
captures the maximum variance.

Orthogonality checks can be done by computing L2 norm - |(V^T * V) - I_n |_2 of the resultant matrices
to check the correctness of the results 

To see sample output matrices in stdout or in file, simply do cout<<v; or fout<<v;
*/

#include<iostream>
#include<mpi.h>
#include<fstream>
#include<Eigen/Dense>
#include<Eigen/SVD>
#include<Eigen/Cholesky>
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
	
	double *recvbuf = new double[N*cols]; 
	double *stack   = new double[npes*cols*cols];
	double *mat_v   = new double[cols*cols];
	
	RMatrixXd s (npes*cols, cols);
	RMatrixXd res (cols,cols), res2 (cols,cols),v (cols,cols);
	VectorXd singular;
	
	double *data = NULL;
	
	if(myrank == 0) {
	    unsigned long long position = 0;
    
        ifstream fin (argv[3]);
        if(fin.is_open()) {
            data = new double[rows*cols];
            while(!fin.eof() && position < (rows*cols)) {
                fin >> data[position];
                position++;
            }
        } else {
            cout<<"Input file name should be included in arguments"<<endl;
            exit(0);
        }
        fin.close();
    }
	//Scatter N rows to each slave from master
	MPI::COMM_WORLD.Scatter(data,N*cols,MPI::DOUBLE,recvbuf,N*cols,MPI::DOUBLE,0);
	
	RMatrixXd a = Map<RMatrixXd>(recvbuf,N,cols);	

	MPI::COMM_WORLD.Barrier();
	double start = MPI::Wtime();

	//mean adjust data for pca
	a = a.rowwise() - a.colwise().mean();		
	a = a/sqrt(rows-1);
	
	LLT<RMatrixXd> lltOfA(a.transpose()*a);
	res = lltOfA.matrixL().transpose(); 
	
	double* residual = res.data();	
	MPI::COMM_WORLD.Gather(residual,cols*cols,MPI::DOUBLE,stack,cols*cols,MPI::DOUBLE,0);
	if(myrank == 0)
	{
		s = Map<RMatrixXd>(stack,npes*cols,cols); 
		LLT<RMatrixXd> lltOfS(s.transpose()*s);
		
		res2 = lltOfS.matrixL().transpose();        //fast - works on positive definite matrix
			
		JacobiSVD<RMatrixXd> svd(res2,ComputeThinV); //Computing only thin V, res2 is small so this step is fast
		v = svd.matrixV();
		
		singular = svd.singularValues();
		singular = singular.array().square();	    //variance associated with each eigen value				
		
		mat_v = v.data();
	}
	MPI::COMM_WORLD.Bcast(mat_v,cols*cols,MPI::DOUBLE,0);		
	
	MPI::COMM_WORLD.Barrier();
	v =  Map<RMatrixXd>(mat_v,cols,cols);	
	RMatrixXd proj = a * v.transpose();		        //transformed data 
		
	MPI::COMM_WORLD.Barrier();
	double end = MPI::Wtime();	
	double time = end - start;
	double max;

	MPI::COMM_WORLD.Reduce(&time, &max,1,MPI::DOUBLE,MPI::MAX,0);

	if(myrank == 0) {     
        ofstream fout;
        fout.open("mpi_PCA_output.txt", ofstream::app | ofstream::out);	
	    fout<<"rows : "<<rows<<"cols : "<<cols;
	    fout<<rows<<" "<<cols<<" : "<<max<<endl;
        fout.close();
    }
    
    delete[] recvbuf;
	delete[] stack;
	 
	MPI::Finalize();
	return 0;
}
