#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

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

	void processLabelFile(string path, bool isTraining);
	void vectorize();


public:

	Eigenfaces(string dir);
};