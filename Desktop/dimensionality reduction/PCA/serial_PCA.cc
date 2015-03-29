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
	
	rows = atoi(argv[1]);
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
	
	MatrixXd res (cols,cols);

	gettimeofday(&start, NULL);
	
	//mean adjust data for pca
	a = a.rowwise() - a.colwise().mean();		
	a = a/sqrt(rows-1);
	
	LLT<MatrixXd> lltOfA(a.transpose()*a);
	res = lltOfA.matrixL().transpose(); 

	JacobiSVD<MatrixXd> svd(res,ComputeThinV);
	res.resize(0,0);
	
	VectorXd singular = svd.singularValues();
	singular = singular.array().square();
	
	MatrixXd proj = a * svd.matrixV().transpose();	

	gettimeofday(&end, NULL);
	
	ofstream fout;
    fout.open("serial_PCA_output.txt", ofstream::app | ofstream::out);	
	fout<<"rows : "<<rows<<"cols : "<<cols;
	fout<<" Time (in ms) : "<<(end.tv_sec - start.tv_sec) * 1000u + (end.tv_usec - start.tv_usec)/1.e3<<endl;
    
    delete[] data;
}
