#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "eigenfaces.hpp"

using namespace Eigen;
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	string dir, filename;
	if (argc >= 3) {
		dir = argv[1], filename = argv[2];
	}
	else {
		cout << "Wrong arguments\n";
		return 0;
	}

	Eigenfaces eigenfaces(dir, filename);

	system("PAUSE");
	return 0;
}