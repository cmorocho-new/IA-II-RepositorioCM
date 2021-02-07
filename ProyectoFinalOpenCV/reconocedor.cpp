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

// Names of windows
const string TITLE_WIN_MAIN = "Mode";
const string TITLE_WIN_ROI = "Video (ROI)";
const string TITLE_WIN_DIL = "Video (Close)";
const string TITLE_WIN_BOR = "Video (Bordes)";

// Names of trackbars
const string NAME_TRACKBAR_H = "Rango (H)";
const string NAME_TRACKBAR_S = "Rango (S)";
const string NAME_TRACKBAR_V = "Rango (V)";

// GUI variables
Mat imageResult;
double huMoments[7];
vector<vector<Point>> pointsContour;

int valBright = 50;
int valProces = 0;
int minMaxHSV = 0;
int valMorph[2] = { 2, 5};

int valHSV[3] = {0, 0, 144};
int valMinHSV[3] = { 0, 0, 144};
int valMaxHSV[3] = { 179, 144, 255};

/**
	Process that performs video gesture recording and detector. 
**/
void processGesture(Mat imgFrame) {
	// Add img title 
	Mat imagenLabel;
	Mat imagenLabel1(24, imgFrame.cols, CV_8UC3, Scalar::all(66));
	Mat imagenLabel2(36, imgFrame.cols, CV_8UC3, Scalar::all(128));
	putText(imagenLabel1, "HuMoments verification technique", Point(10, 14), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(255));
	if (ACTIVE_MODE == DETECT_MODE) {
		putText(imagenLabel2, "PRES (ENTER) >> Gesture detector mode", Point(10, 14), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(0));
	}
	else {
		putText(imagenLabel2, "PRES (ENTER) >> Gesture recorder mode", Point(10, 14), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(0));

	}	
	putText(imagenLabel2, "PRES (DOUBLE CLICK) >> Save gesture", Point(10, 28), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar::all(0));
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
	imageResult = imgFrame.clone();
	// Concat label and original image
	vconcat(imagenLabel, imgFrame, imgFrame);
	imshow(TITLE_WIN_MAIN, imgFrame);
}


void drawGesture(Mat& image, vector<Point> contour) {
	// get all main points
	vector<Point> contourHull;
	convexHull(contour, contourHull);
	for (int i = 0; i < contourHull.size(); i++)
	{
		Point punto = contourHull[i];
		circle(image, contourHull[i], 4, CV_RGB(0, 10, 255), 1.5);
		if ((i + 1) % 2 == 0) {
			line(image, contourHull[(i - 1)], contourHull[i], CV_RGB(0, 0, 255), 1.5);
		}
	}
}

void processVerificacion(Mat& image) {
	string gesture;
	int index = -1, max = 0;
	for (int i = 0; i < pointsContour.size(); i++)
	{
		if (max < pointsContour[i].size()) {
			index = i;
			max = pointsContour[i].size();
		}
	}
	if (index != -1) {
		vector<Point> lastContour = pointsContour[index];
		// get center point
		Rect r = boundingRect(lastContour);
		rectangle(image, r.tl(), r.br(), CV_RGB(255, 0, 0), 2);
		Point center(r.x + (r.width / 2), r.y + (r.height / 2));
		if (ACTIVE_PROCES == HUMOMENTS) {
			// get all moments
			// Mat imageR = Mat(Size(image.cols, image.rows), CV_8UC3, Scalar::all(0));
			// rectangle(imageR, r.tl(), r.br(), Scalar(255), 2);
			// imageR = pintarContenidoContorno(imageR, image);
			// imshow("IMG COUNTOR", imageR);
			Moments momentos = moments(lastContour);
			HuMoments(momentos, huMoments);
			gesture = matchMoments(huMoments);
			if (ACTIVE_MODE == DETECT_MODE){ 
				if (gesture != "") {
					// draw gesture
					drawContours(image, pointsContour, index, Scalar(0, 255, 0), 1.5);
					drawGesture(image, lastContour);
					putText(image, gesture, Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 2.5);
				}
			}
			else {
				if (gesture == "") {
					gesture = "New Gesture (+)";
					drawContours(image, pointsContour, index, Scalar(0, 255, 0), 1.5);
					drawGesture(image, lastContour);
					putText(image, gesture, Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 2.5);
				}
			}
		}
		else {
			// gesture = "None";
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
	Mat imagenR;
	Mat elemento = getStructuringElement(valMorph[0], Size(valMorph[1] + 1, valMorph[1] + 1));
	// morphologyEx(image, imagenR, MORPH_OPEN, elemento);
	morphologyEx(image, imagenR, MORPH_CLOSE, elemento);
	return imagenR;
}

Mat detectContourVideo(Mat image) {
	Mat imageR = Mat(Size(image.cols, image.rows), CV_8UC3, Scalar::all(0));
	findContours(image, pointsContour, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(imageR, pointsContour, -1, Scalar(0, 0, 255), 2);
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
				if (saveMoments(huMoments)) {
					cout << "Momentos del gesto guardado" << endl;
				}
				else {
					cout << "Momentos del gesto existente" << endl;
				}
			}
			else {

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
		createTrackbar("Technique", TITLE_WIN_MAIN, &valProces, 1);
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
			imgFrame.convertTo(imgFrame, -1, 1, valBright * -1);

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