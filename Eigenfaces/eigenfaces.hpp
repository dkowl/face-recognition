#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace Eigen;

class Eigenfaces {

	//Input processing
	static const string TEST_FOLDER, TRAINING_FOLDER, LABEL_FILE;
	static const int EIGENFACE_NO = 40;
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
	int startFace_, endFace_;
	double weightDist(int id1, int id2);
	vector<int> faceEngine(int id, int n);

	//testing
	int classify(int id, bool verbose = false, int imagesToDisplayNo = 1);
	double test(vector<int> testIds, bool verbose = false);
	double test(bool verbose = false);
	Image reconstruct(int id, int n);
	void accuracyTest(bool verbose = false);
	void reconstructionTest();


	//displaying
	void displayEigenfaces(int amount = EIGENFACE_NO);
	void displayImages(vector<int> ids, string winName = "Images");
	Mat displayImage(Image &im);
	Mat displayImage(int i);

	//utils
	int imageSize();
	int imageRows();
	int imageCols();
	int trainingSize();
	int testSize();
	int datasetSize();
	Image normalize(vector<double>& v);

public:

	Eigenfaces(string dir);
};