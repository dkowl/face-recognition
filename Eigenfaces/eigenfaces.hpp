#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;
using namespace Eigen;

class Eigenfaces {

	//Input processing
	static const string TEST_FOLDER, TRAINING_FOLDER, LABEL_FILE;
	string dir_;
	vector<string> filenames_;
	vector<int> labels_;
	vector<int> trainingIds_, testIds_;

	//OpenCV images
	vector<Mat> cvMats_;

	//Image vectors
	typedef vector<uchar> Image;
	vector<Image> images_;
	Image mean_;

	//Eigen matrices
	MatrixXd A_, C_; // A - training images, C - covariance matrix

	void processLabelFile(string path, bool isTraining);
	void vectorize();
	void computeMean();
	void computeCovariance();

	int imageSize();

public:

	Eigenfaces(string dir);
};