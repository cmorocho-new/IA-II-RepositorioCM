// Include estandar c++ libraries
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>

// Include opencv libraries
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>


// Include header
#include "reconocedor.h";
#include "utils.h"

using namespace std;
using namespace cv;

// Types of process active
int ACTIVE_MODE = RECORD_MODE;
int ACTIVE_PROCES = HUMOMENTS;
int BRIGHT_MODE = 0;

// Names of windows
const string TITLE_WIN_MAIN = "Mode";
const string TITLE_WIN_ROI = "Video (ROI)";
const string TITLE_WIN_DIL = "Video (Open + Close)";
const string TITLE_WIN_BOR = "Video (Borde)";
const string TITLE_WIN_GES = "Video (Gesture)";

// Names of trackbars
const string NAME_TRACKBAR_H = "Rango (H)";
const string NAME_TRACKBAR_S = "Rango (S)";
const string NAME_TRACKBAR_V = "Rango (V)";

// GUI variables
Mat imageResult;
double huMoments[7];
double fourierDescriptor[7];
vector<vector<Point>> pointsContours;

int valBright = 50;
int minMaxHSV = 0;
int valMorph[2] = { 2, 5};

int valHSV[3] = {0, 60, 19};
int valMinHSV[3] = { 0, 60, 19 }; // { 0, 0, 144};
int valMaxHSV[3] = { 180, 255, 255 }; // { 179, 144, 255};

/**
	Process that performs video gesture recording and detector. 
**/
void processGesture(Mat imgFrame) {
	// Add img title 
	Mat imagenLabel;
	Mat imagenLabel1(24, imgFrame.cols, CV_8UC3, Scalar::all(66));
	Mat imagenLabel2(36, imgFrame.cols, CV_8UC3, Scalar::all(128));
	if (ACTIVE_PROCES == HUMOMENTS) {
		putText(imagenLabel1, "HuMoments technique", Point(10, 14), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(255));
	}
	else {
		putText(imagenLabel1, "Fourier descriptor technique", Point(10, 14), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(255));
	}
	if (ACTIVE_MODE == DETECT_MODE) {
		putText(imagenLabel2, "PRES (ENTER) >> Gesture detector mode", Point(10, 14), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(0));
	}
	else {
		putText(imagenLabel2, "PRES (ENTER) >> Gesture recorder mode", Point(10, 14), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(0));
		putText(imagenLabel2, "PRES (DOUBLE CLICK) >> Save gesture", Point(10, 28), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(0));
	}	
	vconcat(imagenLabel1, imagenLabel2, imagenLabel);
	// Conver img to HSV
	Mat imgFrameROI = converInRange(imgFrame);
	imshow(TITLE_WIN_ROI, imgFrameROI);
	// Apply close morphology
	Mat imgMorpho = morphOpenCloseVideo(imgFrameROI);
	imshow(TITLE_WIN_DIL, imgMorpho);
	// Apply detecion countornos
	Mat imgContour = detectContourVideo(imgMorpho);
	imshow(TITLE_WIN_BOR, imgContour);
	// Apply verification technique
	processVerificacion(imgFrame);
	// Concat label and original image
	vconcat(imagenLabel, imgFrame, imgFrame);
	imshow(TITLE_WIN_MAIN, imgFrame);
}


void drawMainPointsGesture(Mat& image, vector<Point> contour) {
	// get external points
	vector<Point> convexhull;
	vector<int> convexhullsI;
	convexHull(contour, convexhull);
	convexHull(contour, convexhullsI);
	for (Point punto : convexhull)
		circle(image, punto, 4, CV_RGB(0, 0, 255), 2);
	drawContours(image, vector<vector<Point>>(1, convexhull), 0, CV_RGB(0, 0, 255));
	// get center points
	try {
		vector<Vec4i> convexityDef;
		convexityDefects(contour, convexhullsI, convexityDef);
		for (const Vec4i& defecto : convexityDef)
		{
			float depth = defecto[3] / 256;
			if (depth > 10)
			{
				int faridx = defecto[2]; Point ptFar(contour[faridx]);
				circle(image, ptFar, 4, Scalar(0, 255, 0), 2);
			}
		}
	}
	catch (Exception) {
		cout << "No defects found" << endl;
	}
}

void calcFourierDescriptor(vector<Point> contour, Point center) {

	// Find equidistant points along the contour
	int N = 14; // N point DFT
	size_t dim = contour.size();
	int shift = static_cast<int>(floor((dim / (N)) + 0.5));
	vector<Point> K;
	for (int ii = 0; ii < N; ii++) {
		if (shift * ii > dim) {
			break;
		}
		else {
			K.push_back(contour[shift * ii]);
		}
	}

	// Euclidean distance vector
	vector<double> rt;
	for (int ii = 0; ii < K.size(); ii++) {
		K[ii] -= center;
		rt.push_back(norm(K[ii]));
	}

	// take the magnitude of the N point FFT
	Mat img = Mat(rt);
	Mat planes[] = { Mat_<double>(img), Mat::zeros(img.size(), CV_64F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_COMPLEX_OUTPUT);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);

	// Normalize by the f0 (DC value)
	Mat fAll = planes[0].t();
	fAll /= fAll.at<double>(0);
	int offset = static_cast<int>(floor((N / 2) + 0.5) + 1);
	Mat ft = fAll.colRange(Range(1, offset));

	// fill descriptor array
	for (int i = 0; i < ft.cols; i++) {
		fourierDescriptor[i] = ft.at<double>(i);
	}
}

void calacHuMoments(Mat image) {
	// get hu moments from img
	Moments momentos = moments(image);
	HuMoments(momentos, huMoments);
	for (int i = 0; i < 7; i++)
		huMoments[i] = logTransform(huMoments[i]);
}

void processVerificacion(Mat& image) {
	// get contour with max area
	vector<Point> contoursPoly;
	double maxArea = 0.0f;
	int index = 0;
	for (int i = 0; i < pointsContours.size(); i++)
	{
		approxPolyDP(Mat(pointsContours[i]), contoursPoly, 3, true);
		double currArea = contourArea(contoursPoly);
		if (currArea > maxArea) {
			index = i;
			maxArea = currArea;
		}
	}
	// verify max contour area > 100
	if (maxArea > 100) {
		string gesture;
		vector<Point> pointsContour = pointsContours[index];
		// get center poit from contour
		Rect area = boundingRect(pointsContour);
		Point center(area.x + (area.width / 2), area.y + (area.height / 2));
		rectangle(image, area.tl(), area.br(), CV_RGB(255, 0, 0), 2);
		drawContours(image, pointsContours, index, CV_RGB(255, 0, 0));
		// draw countor filled
		Mat imageR = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);
		drawContours(imageR, pointsContours, index, Scalar(255), FILLED);
		imshow(TITLE_WIN_GES, imageR);
		if (ACTIVE_PROCES == HUMOMENTS) {
			calacHuMoments(imageR);
			gesture = matchDescriptores(huMoments, "MomentosDB.txt", 2);
		}
		else {
			calcFourierDescriptor(pointsContour, center);
			gesture = matchDescriptores(fourierDescriptor, "FourierDescriptoresDB.txt", 0.1);
		}
		if (ACTIVE_MODE == DETECT_MODE) {
			if (gesture != NEW_GESTURE) {
				drawMainPointsGesture(image, pointsContour);
				putText(image, gesture, Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 2.5);
			}
		}
		else {
			if (gesture == NEW_GESTURE) {
				drawMainPointsGesture(image, pointsContour);
				putText(image, gesture, Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 2.5);
			}
		}
	}
}

Mat converInRange(Mat image) {
	Mat imageR;
	cvtColor(image, imageR, COLOR_BGR2HSV);
	inRange(imageR, Scalar(valMinHSV[0], valMinHSV[1], valMinHSV[2]),
		Scalar(valMaxHSV[0], valMaxHSV[1], valMaxHSV[2]), imageR);
	return imageR;
}

Mat morphOpenCloseVideo(Mat image) {
	Mat imageR, imagenR1, imagenR2;
	Mat elemento = getStructuringElement(valMorph[0], Size(valMorph[1] + 1, valMorph[1] + 1));
	morphologyEx(image, imagenR1, MORPH_OPEN, elemento);
	morphologyEx(image, imagenR2, MORPH_CLOSE, elemento);
	bitwise_or(imagenR1, imagenR2, imageR);
	return imageR;
}

Mat detectContourVideo(Mat image) {
	Mat imageR = Mat(Size(image.cols, image.rows), CV_8UC3, Scalar::all(0));
	findContours(image, pointsContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(imageR, pointsContours, -1, Scalar(0, 0, 255), 2);
	return imageR;
}

void changeMinMaxHSV(int val, void* p) {
	if (val == MIN_RANGE) {
		setWindowTitle(TITLE_WIN_ROI, TITLE_WIN_ROI + "( MIN )");
		copy(begin(valMinHSV), end(valMinHSV), valHSV);
	}
	else {
		setWindowTitle(TITLE_WIN_ROI, TITLE_WIN_ROI + "( MAX )");
		copy(begin(valMaxHSV), end(valMaxHSV), valHSV);
	}
	// change the value 
	setTrackbarPos(NAME_TRACKBAR_H, TITLE_WIN_ROI, valHSV[0]);
	setTrackbarPos(NAME_TRACKBAR_S, TITLE_WIN_ROI, valHSV[1]);
	setTrackbarPos(NAME_TRACKBAR_V, TITLE_WIN_ROI, valHSV[2]);
}

void changeRangeMinMaxHSV(int val, void* p) {
	int tipo = *(int*) p;
	if (minMaxHSV == MIN_RANGE) {
		valMinHSV[tipo] = val;
	}
	else if (minMaxHSV == MAX_RANGE) {
		valMaxHSV[tipo] = val;
	}
}

void clickMouseWindown(int event, int x, int y, int flags, void* param) {
	if (event == EVENT_LBUTTONDBLCLK) {
		if (ACTIVE_MODE == DETECT_MODE) {
			// save the image result
			imwrite("image-result-gesture-CM.png", imageResult);
		}
		else {
			if (ACTIVE_PROCES == HUMOMENTS) {
				if (saveDescriptores(huMoments, "MomentosDB.txt", 1)) {
					cout << "Momentos del gesto guardado" << endl;
				}
				else {
					cout << "Momentos del gesto existente" << endl;
				}
			}
			else {
				if (saveDescriptores(fourierDescriptor, "FourierDescriptoresDB.txt", 0.1)) {
					cout << "Descriptor del gesto guardado" << endl;
				}
				else {
					cout << "Descriptor del gesto existente" << endl;
				}
			}
		}
	}
}

void processVideoCamera() {
	// initialize the camera
	VideoCapture video(0);
	if (video.isOpened()) {
		int opcion;

		// IMG vars
		Mat imgFrame;
		Mat imgFrameD;

		// Add windown
		namedWindow(TITLE_WIN_MAIN, WINDOW_AUTOSIZE);
		namedWindow(TITLE_WIN_ROI, WINDOW_AUTOSIZE);
		namedWindow(TITLE_WIN_DIL, WINDOW_AUTOSIZE);
		namedWindow(TITLE_WIN_BOR, WINDOW_AUTOSIZE);

		// Add event to main windown
		setWindowTitle(TITLE_WIN_ROI, TITLE_WIN_ROI + "( MIN )");
		if (ACTIVE_MODE == DETECT_MODE) {
			setWindowTitle(TITLE_WIN_MAIN, "Gesture detector (ACTIVATED)");
		}
		else {
			setWindowTitle(TITLE_WIN_MAIN, "Gesture recorder (ACTIVATED)");
		}
		setMouseCallback(TITLE_WIN_MAIN, clickMouseWindown);

		// Add trackbars for Brightness
		createTrackbar("Technique", TITLE_WIN_MAIN, &ACTIVE_PROCES, 1);
		createTrackbar("Bright(*)", TITLE_WIN_MAIN, &BRIGHT_MODE, 1);
		createTrackbar("Brightness", TITLE_WIN_MAIN, &valBright, 100);

		// Add trackbars for Dilation
		createTrackbar("Element", TITLE_WIN_DIL, &valMorph[0], 2);
		createTrackbar("Kernel", TITLE_WIN_DIL, &valMorph[1], 50);

		// Add trackbars for InRange HSV
		int tipos[3] = { 0, 1, 2 };
		createTrackbar("Min | Max", TITLE_WIN_ROI, &minMaxHSV, 1, changeMinMaxHSV);
		createTrackbar(NAME_TRACKBAR_H, TITLE_WIN_ROI, &valHSV[0], 180, changeRangeMinMaxHSV, &tipos[0]);
		createTrackbar(NAME_TRACKBAR_S, TITLE_WIN_ROI, &valHSV[1], 255, changeRangeMinMaxHSV, &tipos[1]);
		createTrackbar(NAME_TRACKBAR_V, TITLE_WIN_ROI, &valHSV[2], 255, changeRangeMinMaxHSV, &tipos[2]);

		// get video frames
		while (true) {
			video >> imgFrame;
			resize(imgFrame, imgFrame, Size(), 0.6, 0.6);
			imgFrame.convertTo(imgFrame, -1, 1, valBright * (BRIGHT_MODE == 0 ? -1 : 1));

			// wait key
			if (waitKey(25) == 13) {
				if (ACTIVE_MODE == DETECT_MODE) {
					ACTIVE_MODE = RECORD_MODE;
					setWindowTitle(TITLE_WIN_MAIN, "Gesture recorder (ACTIVATED)");
				}
				else {
					ACTIVE_MODE = DETECT_MODE;
					setWindowTitle(TITLE_WIN_MAIN, "Gesture detector (ACTIVATED)");
				}
			}
			// call the process
			processGesture(imgFrame);

		}
	}else{
		cout << "The video camera couldn't be open";
	}
}