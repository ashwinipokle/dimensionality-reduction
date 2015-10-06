/**
Serial SVD using QR decomposition
Final matrices of SVD are stored as below - 
U in MatrixXd u
variance in VectorXd singular
v in MatrixXd v
*/

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
	int cols;
	
	if(argv[1] == NULL || argv[2] == NULL) {
	    cout<<"Mention rows and columns as arguments"<<endl;
	    exit(0);
	}
	
	rows = atol(argv[1]);
	cols = atoi(argv[2]);
	
    unsigned long long position = 0;
    double* data = new double[rows*cols];
    
    ifstream fin (argv[3]);
    if(fin.is_open()) {
        while(!fin.eof() && position < (rows*cols)) {
            fin >> data[position];
            position++;
        }
    } else {
        cout<<"Input file name should be included in arguments"<<endl;
        exit(0);
    }
    
    RMatrixXd a = Map<RMatrixXd>(data, rows, cols);
    cout<<a<<endl;
	MatrixXd res (cols,cols);
	MatrixXd q1 (rows,cols);

	gettimeofday(&start, NULL);
	
	LLT<MatrixXd> lltOfA(a.transpose()*a);
	res = lltOfA.matrixL().transpose(); 
	q1 = a * res.inverse();
	
	JacobiSVD<MatrixXd> svd(res, ComputeThinU | ComputeThinV);
    
	MatrixXd v = svd.matrixV();             //Matrix V of SVD
	VectorXd singular = svd.singularValues(); //Vector storing sigma
	
	q1 = q1*svd.matrixU();	                //final matrix U

    gettimeofday(&end, NULL);
   
    ofstream fout;
    fout.open("serial_SVD_output.txt", ofstream::app | ofstream::out);	
	fout<<"rows : "<<rows<<"cols : "<<cols;
	fout<<" Time (in ms) : "<<(end.tv_sec - start.tv_sec) * 1000u + (end.tv_usec - start.tv_usec)/1.e3<<endl;
    
    delete[] data;
}
