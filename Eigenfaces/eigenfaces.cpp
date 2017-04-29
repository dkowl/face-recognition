#include "eigenfaces.hpp"

const string Eigenfaces::TEST_FOLDER = "test\\";
const string Eigenfaces::TRAINING_FOLDER = "training\\";
const string Eigenfaces::LABEL_FILE = "classes.csv";

Eigenfaces::Eigenfaces(string dir) :
	dir_(dir),
	startFace_(1),
	endFace_(EIGENFACE_NO)
{
	string path = dir_ + TRAINING_FOLDER + LABEL_FILE;
	processLabelFile(path, true);	

	path = dir_ + TEST_FOLDER + LABEL_FILE;
	processLabelFile(path, false);

	vectorize();
	computeMean();
	computeEigenfaces();
	displayEigenfaces();
	computeWeights();

	for (int i = 1; i < EIGENFACE_NO; i*=2) {
		for (int j = i; j < EIGENFACE_NO; j *= 2) {
			startFace_ = i;
			endFace_ = j;
			cout << test() << " ";
		}
		cout << endl;
	}
}

void Eigenfaces::processLabelFile(string path, bool isTraining)
{
	try {
		ifstream input(path);

		//csv processing
		char delim = ';';
		string line;

		while (getline(input, line)) {
			string s;
			vector<string> values;

			for (int i = 0; i < line.size(); i++) {
				if (line[i] != delim) {
					s += line[i];
				}
				else {
					values.push_back(s);
					s.clear();
				}
			}
			if (!s.empty())
				values.push_back(s);

			if (values.size() < 2) {
				throw (std::exception("Invalid csv line"));
			}
			filenames_.push_back(values[0]);
			labels_.push_back(stoi(values[1]));
			if (isTraining)
				trainingIds_.push_back(labels_.size()-1);
			else
				testIds_.push_back(labels_.size() - 1);
		}
	}
	catch (exception &e) {
		cerr << e.what();
	}
}

void Eigenfaces::vectorize()
{
	vector<string> paths(filenames_.size());

	for (auto&& i : trainingIds_) {
		paths[i] = dir_ + TRAINING_FOLDER + filenames_[i];
	}
	for (auto&& i : testIds_) {
		paths[i] = dir_ + TEST_FOLDER + filenames_[i];
	}

	cvMats_.reserve(labels_.size());
	images_.reserve(labels_.size());
	for (int i = 0; i < paths.size(); i++) {
		cvMats_.push_back(imread(paths[i], IMREAD_GRAYSCALE));

		Image image;
		for (int r = 0; r < cvMats_[i].rows; r++) {
			image.insert(image.end(), cvMats_[i].ptr<uchar>(r), cvMats_[i].ptr<uchar>(r) + cvMats_[i].cols);
		}
		images_.push_back(image);
	}
}

void Eigenfaces::computeMean()
{
	vector<int> sum(images_[0].size());
	for (auto&& image : images_) {
		for (int i = 0; i < image.size(); i++) {
			sum[i] += image[i];
		}
	}
	for (auto&& i : sum) {
		i /= images_.size();
	}
	mean_.insert(mean_.end(), sum.begin(), sum.end());

	/*Mat mat(cvMats_[0].size(), CV_8U);
	memcpy(mat.data, mean_.data(), mean_.size() * sizeof(uchar));
	imshow("Mean", mat);
	waitKey();*/
}

void Eigenfaces::computeEigenfaces()
{
	//Constructing array of training images
	MatrixXd A(trainingSize(), imageSize());
	for (int i = 0; i < trainingIds_.size(); i++) {
		for (int j = 0; j < images_[i].size(); j++) {
			A(i, j) = double(images_[i][j] - mean_[j]);
		}
	}

	//computing eigenvectors
	MatrixXd covariance = (A*A.transpose())/imageSize();
	EigenSolver<MatrixXd> es(covariance);
	MatrixXcd ev = es.eigenvectors();
	MatrixXd evReal(trainingSize(), trainingSize());
	for (int i = 0; i < trainingSize(); i++) {
		for (int j = 0; j < trainingSize(); j++) {
			evReal(i, j) = ev(i, j).real();
		}
	}

	//sorting eigenvectors by eigenvalues
	VectorXcd evals = es.eigenvalues();
	vector<pair<double, int>> evalsReal;
	for (int i = 0; i < trainingSize(); i++) {
		evalsReal.push_back(pair<double, int>(evals(i).real(), i));
	}
	sort(evalsReal.begin(), evalsReal.end(), greater<pair<double, int>>());
	MatrixXd evRealSorted(trainingSize(), trainingSize());
	for (int i = 0; i < trainingSize(); i++) {
		evRealSorted.col(i) = evReal.col(evalsReal[i].second);
	}

	eigenfaces_ = A.transpose()*evRealSorted;
}

void Eigenfaces::computeWeights()
{
	for (int i = 0; i < datasetSize(); i++) {
		//Image vector
		VectorXd I(imageSize());
		for (int j = 0; j < imageSize(); j++) {
			I(j) = double(images_[i][j] - mean_[j]);
		}
		//Weight vector
		WeightV w;

		for (int j = 0; j < EIGENFACE_NO; j++) {
			//Eigenface vector
			VectorXd E = eigenfaces_.col(j);
			w[j] = E.transpose()*I;
			w[j] /= imageSize() * 128;
		}

		weights_.push_back(w);
	}
}

double Eigenfaces::weightDist(int id1, int id2)
{
	double result = 0.0;
	for (int i = startFace_ - 1; i < endFace_; i++) {
		double diff = weights_[id1][i] - weights_[id2][i];
		result += diff*diff;
	}
	return result / EIGENFACE_NO;
}

vector<int> Eigenfaces::faceEngine(int id, int n)
{
	vector<pair<double, int>> distV;
	for (auto i: trainingIds_) {
		if (i != id) distV.push_back(pair<double, int>(weightDist(id, i), i));
	}
	sort(distV.begin(), distV.end());
	vector<int> result;
	for (int i = 0; i < n; i++) {
		result.push_back(distV[i].second);
	}
	return result;
}

int Eigenfaces::classify(int id, bool verbose, int imagesToDisplayNo)
{
	vector<int> v = faceEngine(id, imagesToDisplayNo);

	if (verbose) {
		displayImages(vector<int>{id}, "Test case");
		displayImages(v, "Results");
	}

	return labels_[v[0]];
}

double Eigenfaces::test(vector<int> testIds, bool verbose)
{
	double correct = 0.0;
	for (auto testId : testIds) {
		int label = classify(testId, verbose, 10);
		if (label == labels_[testId]) correct++;
	}
	if(verbose) cout << "Accuracy: " << correct / testIds.size() << endl;
	return correct / testIds.size();
}

double Eigenfaces::test(bool verbose)
{
	return test(testIds_, verbose);
}

void Eigenfaces::displayEigenfaces(int amount)
{
	if (amount > trainingSize()) amount = trainingSize();

	Mat eigenfacesMat(imageRows(), imageCols()*amount, CV_8U);
	for (int i = 0; i < imageRows(); i++) {
		for (int j = 0; j < imageCols()*amount; j++) {
			eigenfacesMat.at<uchar>(i, j) = uchar(eigenfaces_(j%imageCols() + i*imageCols(), j / imageCols()) / 4) + 127;
		}
	}
	imshow("Eigenfaces", eigenfacesMat);
	waitKey();
}

void Eigenfaces::displayImages(vector<int> ids, string winName)
{
	vector<Mat> mats;
	for (auto i : ids) mats.push_back(cvMats_[i]);
	Mat result;
	hconcat(mats,result);
	imshow(winName, result);
	waitKey();
}

int Eigenfaces::imageSize()
{
	return images_[0].size();
}

int Eigenfaces::imageRows()
{
	return cvMats_[0].rows;
}

int Eigenfaces::imageCols()
{
	return cvMats_[0].cols;
}

int Eigenfaces::trainingSize()
{
	return trainingIds_.size();
}

int Eigenfaces::testSize()
{
	return testIds_.size();
}

int Eigenfaces::datasetSize()
{
	return images_.size();
}