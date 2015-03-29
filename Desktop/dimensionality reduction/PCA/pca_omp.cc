#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
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
	rows = atoi(argv[1]);
	cols = atoi(argv[2]);
	nthreads = atoi(argv[3]);
	
	int N = ceil(rows/nthreads); //rows for each partition of data matrix

	MatrixXd s (nthreads*cols,cols);
	MatrixXd res (cols,cols), res2 (cols,cols);
	VectorXd singular;
	MatrixXd v;
	
	unsigned long long position = 0;
	
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
    fin.close();
    RMatrixXd data = Map<RMatrixXd>(input, rows, cols);	

	gettimeofday(&start, NULL);
	#pragma omp parallel num_threads(nthreads) shared(data,s,v,singular) private(res) 
	{
		MatrixXd a = data.block(N*omp_get_thread_num(),0,N,cols);
		//mean adjust data for pca
		a = a.rowwise() - a.colwise().mean();	
		a = a/sqrt(rows-1);
		//start svd calculations				
		LLT<MatrixXd> lltOfA(a.transpose()*a);
		res = lltOfA.matrixL().transpose(); 
		  
		s.block(cols*omp_get_thread_num(),0,cols,cols) = res;
		#pragma omp barrier
		#pragma omp single
		{
			LLT<MatrixXd> lltOfS(s.transpose()*s);
			res2 = lltOfS.matrixL().transpose(); //fast - works on positive semidefinite matrix
						
			JacobiSVD<MatrixXd> svd(res2,ComputeThinV); //computation of U not needed really
			singular = svd.singularValues();
			singular = singular.array().square();						
			v = svd.matrixV();
		} 		
		//final U : not really needed for calculating PCA
		MatrixXd proj = a * v.transpose();		//transformed data to lower dimension 	
	}
	
	gettimeofday(&end, NULL);
	
	ofstream fout;
    fout.open("omp_PCA_output.txt", ofstream::app | ofstream::out);		
	fout<<rows<<" "<<cols<<" : ";
	fout<< (end.tv_sec - start.tv_sec) * 1000u + (end.tv_usec - start.tv_usec)/1.e3<<endl;
	
	return 0;
}

