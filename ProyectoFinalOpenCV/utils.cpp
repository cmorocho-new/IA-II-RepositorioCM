#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

// for string delimiter
vector<string> split(string s, string delimiter) {
	size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	string token;
	vector<string> res;

	while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}

	res.push_back(s.substr(pos_start));
	return res;
}

double euclideanDistance(double m1[7], double m2[7]) {
	double suma = 0.0;
	for (int i = 0; i < 7; i++) {
		suma += (m1[i] - m2[i]) * (m1[i] - m2[i]);
	}
	return sqrt(suma);
}

string matchMoments(double huMomentsNew[7], string fileName, double valueAcept) {
	double huMomentsOld[7];
	string line, gesture;
	vector<string> atributes;
	ifstream infile(fileName);
	if (infile.is_open()) {
		while (getline(infile, line))
		{
			// get all atributes form line
			atributes = split(line, ";");
			for (int i = 0; i < 7; i++)
			{
				huMomentsOld[i] = stod(atributes[i]);
			}
			// compare two moments
			if (euclideanDistance(huMomentsNew, huMomentsOld) < valueAcept) {
				gesture = atributes[7];
			}
		}
		// close file stream
		infile.close();
	}
	return gesture;
}

bool saveMoments(double huMomentsNew[7], string fileName, double valueAcept) {
	double huMomentsOld[7];
	int next = 1;
	string line, gesture;
	vector<string> atributes;
	ifstream infile(fileName);
	if (infile.is_open()) {
		gesture = "";
		while (getline(infile, line))
		{
			next++;
			// get all atributes form line
			gesture += line + "\n";
			atributes = split(line, ";");
			for (int i = 0; i < 7; i++)
			{
				huMomentsOld[i] = stod(atributes[i]);
			}
			// compare two moments
			if (euclideanDistance(huMomentsNew, huMomentsOld) < valueAcept) {
				return false;
			}
		}
		infile.close();
		ofstream oufile(fileName);
		if (oufile.is_open()) {
			for (int i = 0; i < 7; i++)
			{
				gesture += to_string(huMomentsNew[i]) + ";";
			}
			gesture += "Gesture(" + to_string(next) + ")";
			oufile << gesture << endl;
			oufile.close();
		}
	}
	return true;
}


Mat pintarContenidoContorno(Mat image1, Mat image2) {

	Vec3b pixel;
	Mat imageR = image2.clone();
	for (int i = 0; i < image1.rows; i++) {
		for (int j = 0; j < image1.cols; j++) {
			pixel = image1.at<Vec3b>(i, j);
			if (pixel[0] == 0 && pixel[1] == 0 && pixel[1] == 0) {
				imageR.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else {
				break;
			}
		}

		for (int j = image1.cols - 1; j > 0; j--) {
			pixel = image1.at<Vec3b>(i, j);
			if (pixel[0] == 0 && pixel[1] == 0 && pixel[1] == 0) {
				imageR.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else
				break;
		}

	}

	return imageR;

}