#include "eigenfaces.hpp"

const string Eigenfaces::TEST_FOLDER = "test\\";
const string Eigenfaces::TRAINING_FOLDER = "training\\";
const string Eigenfaces::LABEL_FILE = "classes.csv";

Eigenfaces::Eigenfaces(string dir):
	dir_(dir)
{
	string path = dir_ + TRAINING_FOLDER + LABEL_FILE;
	processLabelFile(path, true);	

	path = dir_ + TEST_FOLDER + LABEL_FILE;
	processLabelFile(path, false);

	vectorize();
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