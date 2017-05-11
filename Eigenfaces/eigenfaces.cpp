#include "eigenfaces.hpp"

const string Eigenfaces::TEST_FOLDER = "test\\";
const string Eigenfaces::TRAINING_FOLDER = "training\\";
const string Eigenfaces::LABEL_FILE = "classes.csv";

Eigenfaces::Eigenfaces(string dir) :
	dir_(dir),
	startFace_(1),
	endFace_(EIGENFACE_NO),
	kNeighbours_(1)
{
	string path = dir_ + TRAINING_FOLDER + LABEL_FILE;
	processLabelFile(path, true);	

	path = dir_ + TEST_FOLDER + LABEL_FILE;
	processLabelFile(path, false);

	computePaths();
	vectorize();
	train();
	
	//reconstructionTest();
	/*vector<int> kV{ 1, 3, 7, 15 };
	for (auto k : kV) {
		kNeighbours_ = k;
		cout << " k-neighbours: " << k << endl;
		accuracyTest();
	}*/
	//test(true);
}

int Eigenfaces::addFace(string path, int label, bool training)
{
	Mat mat = imread(path);
	cvtColor(mat, mat, COLOR_RGB2GRAY);
	resize(mat, mat, Size(imageCols(), imageRows()));
	cvMats_.push_back(mat);
	images_.push_back(cvMatToImage(mat));

	paths_.push_back(path);
	labels_.push_back(label);
	int id = labels_.size() - 1;
	if (training) {
		trainingIds_.push_back(id);
	}
	else {
		testIds_.push_back(id);
	}
	classIds_[label].push_back(id);

	if (training) {
		train();
	}
	else computeWeights();

	return id;
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
			int label = stoi(values[1]);
			filenames_.push_back(values[0]);
			labels_.push_back(label);
			int id = labels_.size() - 1;
			if (isTraining)
				trainingIds_.push_back(id);
			else
				testIds_.push_back(id);
			classIds_[label].push_back(id);
		}
	}
	catch (exception &e) {
		cerr << e.what();
	}
}

void Eigenfaces::computePaths()
{
	paths_ = vector<string>(filenames_.size());

	for (auto&& i : trainingIds_) {
		paths_[i] = dir_ + TRAINING_FOLDER + filenames_[i];
	}
	for (auto&& i : testIds_) {
		paths_[i] = dir_ + TEST_FOLDER + filenames_[i];
	}
}

void Eigenfaces::vectorize()
{
	cvMats_.reserve(labels_.size());
	images_.reserve(labels_.size());
	for (int i = 0; i < paths_.size(); i++) {
		cvMats_.push_back(imread(paths_[i], IMREAD_GRAYSCALE));		
		images_.push_back(cvMatToImage(cvMats_[i]));
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
}

void Eigenfaces::computeClassMeans()
{
	
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

void Eigenfaces::computeFisherfaces()
{
	
}

void Eigenfaces::computeWeights()
{
	weights_ = vector<WeightV>(datasetSize());
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

		weights_[i] = w;
	}
}

void Eigenfaces::train()
{
	computeMean();
	computeEigenfaces();
	computeWeights();
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

	map<int, int> labelCount;
	for (int i = 0; i < v.size(); i++) {
		labelCount[labels_[v[i]]]++;
	}
	int maxCount = 0, bestLabel;
	for (auto p : labelCount) {
		if (p.second > maxCount) {
			maxCount = p.second;
			bestLabel = p.first;
		}
	}
	return bestLabel;
}

double Eigenfaces::test(vector<int> testIds, bool verbose)
{
	double correct = 0.0;
	for (auto testId : testIds) {
		int label = classify(testId, verbose, kNeighbours_);
		if (label == labels_[testId]) correct++;
		if (verbose) {
			cout << "Input: " << labels_[testId] << " --- Output: " << label << endl;
		}
	}
	if(verbose) cout << "Accuracy: " << correct / testIds.size() << endl;
	return correct / testIds.size();
}

double Eigenfaces::test(bool verbose)
{
	return test(testIds_, verbose);
}

Eigenfaces::Image Eigenfaces::reconstruct(int id, int n)
{
	vector<double> im(imageSize());
	if (n > EIGENFACE_NO) n = EIGENFACE_NO;

	for (int i = 0; i < imageSize(); i++) {
		for (int j = 0; j < n; j++) {
			im[i] += eigenfaces_(i, j) * weights_[id][j];
		}
	}

	Image imN = normalize(im);
	for (int i = 0; i < im.size(); i++) {
		im[i] = mean_[i] + imN[i];
	}
	return normalize(im);
}

void Eigenfaces::accuracyTest(bool verbose)
{
	for (int i = 1; i < EIGENFACE_NO; i *= 2) {
		for (int j = i; j < EIGENFACE_NO; j *= 2) {
			startFace_ = i;
			endFace_ = j;
			cout << test(verbose) << " ";
		}
		cout << endl;
	}
}

void Eigenfaces::reconstructionTest(vector<int> faceIds)
{
	if (faceIds.empty()) {
		for (int i = 0; i < datasetSize(); i++) faceIds.push_back(i);
	}
	vector<int> faceNo{ 5, 10, 20, EIGENFACE_NO };
	Mat original, reconstructed, error(imageRows(), imageCols(), CV_8U);
	for (auto i: faceIds) {
		for (auto f : faceNo) {
			original = displayImage(i);
			reconstructed = displayImage(reconstruct(i, f));
			absdiff(original, reconstructed, error);
			vector<Mat> mats{ original, reconstructed, error };
			Mat out;
			hconcat(mats, out);
			imshow("Reconstruction", out);
			waitKey();
		}
	}
}

void Eigenfaces::displayEigenfaces(int amount)
{
	if (amount > trainingSize()) amount = trainingSize();
	static const int facesPerRow = 15;
	int rows = amount / facesPerRow, cols = min(facesPerRow, amount);

	Mat eigenfacesMat(imageRows()*rows, imageCols()*cols, CV_8U);
	vector<Mat> matRows;
	for (int i = 0; i < rows; i++) {
		Mat matRow;
		vector<Mat> rowMats;
		for (int j = 0; j < cols; j++) {
			vector<double>vTemp(imageSize());
			for (int k = 0; k < imageSize(); k++) {
				vTemp[k] = eigenfaces_(k, i*cols + j);
			}
			rowMats.push_back(displayImage(normalize(vTemp)));
		}
		hconcat(rowMats, matRow);
		matRows.push_back(matRow);
	}
	vconcat(matRows, eigenfacesMat);
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

Mat Eigenfaces::displayImage(Image &im)
{
	Mat m(imageRows(), imageCols(), CV_8U);

	for (int i = 0; i < imageRows(); i++) {
		for (int j = 0; j < imageCols(); j++) {
			m.at<uchar>(i, j) = im[i*imageCols() + j];
		}
	}

	return m;
}

Mat Eigenfaces::displayImage(int i)
{
	return displayImage(images_[i]);
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

Eigenfaces::Image Eigenfaces::normalize(vector<double>& v)
{
	double minVal = v[0], maxVal = v[0];
	for (int i = 1; i < v.size(); i++) {
		minVal = min(minVal, v[i]);
		maxVal = max(maxVal, v[i]);
	}
	Image im(v.size());
	for (int i = 0; i < v.size(); i++) {
		im[i] = (v[i] - minVal) * (255 / (maxVal - minVal));
	}

	return im;
}

Eigenfaces::Image Eigenfaces::cvMatToImage(Mat mat)
{
	Image image;
	for (int r = 0; r < mat.rows; r++) {
		image.insert(image.end(), mat.ptr<uchar>(r), mat.ptr<uchar>(r) + mat.cols);
	}
	return image;
}

void Eigenfaces::testCustomFace(string path)
{
	int id = addFace(path, datasetSize()+1, false);
	classify(id, true, 10);
	reconstructionTest(vector<int>{id});
}