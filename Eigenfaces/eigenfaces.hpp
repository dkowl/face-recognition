#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

class Eigenfaces {

	//Input processing
	vector<string> filenames_;
	vector<int> labels_;

	void vectorize();

public:

	Eigenfaces(string dir, string filename);
};