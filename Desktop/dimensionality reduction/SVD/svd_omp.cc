#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>

using namespace std;
using namespace Eigen;

typedef Matrix<double,Dynamic,Dynamic,RowMajor> RMatrixXd;

int main(int argc, char *argv[])
{
	struct timeval start,end;
	unsigned long rows;
	int cols, nthreads;
	unsigned long long position = 0;
	
	rows = atol(argv[1]);
	cols = atoi(argv[2]);
	nthreads = atoi(argv[3]);
	
	int N = ceil(rows/nthreads); //rows for each partition of data matrix

	MatrixXd q1 (N,cols);
	MatrixXd q2 (nthreads*cols,cols); 
	MatrixXd s (nthreads*cols,cols);
	MatrixXd res (cols,cols); MatrixXd res2 (cols,cols); MatrixXd u (cols,cols);
	MatrixXd v;
	VectorXd sigma;
	
	double* input = new double[rows*cols];
	ifstream fin (argv[4]);
    if(fin.is_open()) {
        while(!fin.eof() && position < (rows*cols)) {
            fin >> input[position];
            position++;
        }
    } else {
        cout<<"File not found at path given. Enter correct file name "<<endl;
        exit(0);
    }
    
    RMatrixXd data = Map<RMatrixXd>(input, rows, cols);
	gettimeofday(&start, NULL);
	#pragma omp parallel num_threads(nthreads) shared(data,s,q2,u,v,sigma) private(res,q1) 
	{
		MatrixXd a = data.block(N*omp_get_thread_num(),0,N,cols);
		LLT<MatrixXd> lltOfA(a.transpose()*a);
		res = lltOfA.matrixL().transpose(); 
		q1 = a * res.inverse();  
		s.block(cols*omp_get_thread_num(),0,cols,cols) = res;
		#pragma omp barrier
		#pragma omp single
		{
			LLT<MatrixXd> lltOfS(s.transpose()*s);
			res2 = lltOfS.matrixL().transpose(); //fast - works on positive definite matrix
			q2 = s*res2.inverse();			
			JacobiSVD<MatrixXd> svd(res2, ComputeThinU | ComputeThinV);
			u = svd.matrixU();
			v = svd.matrixV();
			sigma = svd.singularValues();
		} 
		q1 = q1*q2.block(omp_get_thread_num()*cols,0,cols,cols);
		a = q1*u;	                            //The matrix U - compact svd 
		data.block(N*omp_get_thread_num(),0,N,cols) = a;	
	}
	gettimeofday(&end, NULL);
   
	ofstream fout;
    fout.open("omp_SVD_output.txt", ofstream::app | ofstream::out);		
	fout<<rows<<" "<<cols<<" : ";
	fout<< (end.tv_sec - start.tv_sec) * 1000u + (end.tv_usec - start.tv_usec)/1.e3<<endl;
	
	return 0;
}

