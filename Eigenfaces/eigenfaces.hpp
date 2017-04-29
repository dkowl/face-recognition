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
	static const int EIGENFACE_NO = 20;
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

	//Weight vectors
	struct WeightV {
		double w[EIGENFACE_NO];
		double& operator[](int i) { return w[i]; }
	};
	vector<WeightV> weights_;

	//training
	void processLabelFile(string path, bool isTraining);
	void vectorize();
	void computeMean();
	void computeEigenfaces();
	void computeWeights();

	//classification
	double weightDist(int id1, int id2);

	//displaying
	void displayEigenfaces(int amount = EIGENFACE_NO);

	//utils
	int imageSize();
	int imageRows();
	int imageCols();
	int trainingSize();
	int testSize();
	int datasetSize();

public:

	Eigenfaces(string dir);
};