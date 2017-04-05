#include "eigenfaces.hpp"

Eigenfaces::Eigenfaces(string dir, string filename)
{
	try {
		string path = dir + filename;
		ifstream input(path);

		cout << input.rdbuf();
	}
	catch (exception &e) {
		cerr << e.what();
	}
}