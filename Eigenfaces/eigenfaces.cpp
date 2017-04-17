#include "eigenfaces.hpp"

Eigenfaces::Eigenfaces(string dir, string filename)
{
	try {
		string path = dir + filename;
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
			if(!s.empty())
				values.push_back(s);

			if (values.size() < 2) {
				throw (std::exception("Invalid csv line"));
			}
			filenames_.push_back(values[0]);
			labels_.push_back(stoi(values[1]));
		}

		for (int i = 0; i < filenames_.size(); i++) {
			cout << filenames_[i] << endl;
			cout << labels_[i] << endl;
		}
	}
	catch (exception &e) {
		cerr << e.what();
	}
}

void Eigenfaces::vectorize()
{

}