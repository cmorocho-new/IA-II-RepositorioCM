#pragma once
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

enum TypeProcess {
	DETECT_MODE = 1,
	RECORD_MODE = 2,

	MAX_RANGE = 1,
	MIN_RANGE = 0,

	HUMOMENTS = 0,
	SIGNATURE = 1,
};

Mat converInRange(Mat image);

Mat detectContourVideo(Mat image);

Mat morphOpenCloseVideo(Mat image);

void drawGesture(Mat& image, vector<Point> contour);

void processVerificacion(Mat& image);

void processGesture(Mat imgFrame);

void processVideoCamera();