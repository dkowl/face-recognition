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
	static const int EIGENFACE_NO;
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
	MatrixXd eigenfaces_;

	void processLabelFile(string path, bool isTraining);
	void vectorize();
	void computeMean();
	void computeEigenfaces();

	void displayEigenfaces(int amount = EIGENFACE_NO);

	int imageSize();
	int imageRows();
	int imageCols();
	int trainingSize();

public:

	Eigenfaces(string dir);
};