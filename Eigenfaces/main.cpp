#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "eigenfaces.hpp"

using namespace Eigen;
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	string dir;
	if (argc >= 2) {
		dir = argv[1];
	}
	else {
		cout << "Wrong arguments\n";
		return 0;
	}

	Eigenfaces eigenfaces(dir);
	eigenfaces.setMethod(Eigenfaces::FISHER);
	eigenfaces.accuracyTest();

	system("PAUSE");
	return 0;
}