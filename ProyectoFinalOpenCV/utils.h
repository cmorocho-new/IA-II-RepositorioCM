#pragma once
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;

// for string delimiter
vector<string> split(string s, string delimiter);

// for calc euclidean distance
double euclideanDistance(double m1[7], double m2[7]);

Mat pintarContenidoContorno(Mat image1, Mat image2);

bool saveMoments(double huMomentsNew[7], string fileName = "MomentosDB.txt", double valueAcept = 0.02);

string matchMoments(double huMomentsNew[7], string fileName = "MomentosDB.txt", double valueAcept = 0.02);