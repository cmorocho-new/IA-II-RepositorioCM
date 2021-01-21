#include <iostream>

#include <opencv2/core/core.hpp> 
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "MorfologiaHeader.h"

using namespace cv;
using namespace std;

Mat imagen, imagenDst;
int morphElem = 0;
int morphSize = 0;
int morphOperator = 0;
int const maxOperator = 4;
int const maxElem = 2;
int const maxKernelSize = 40;

void morphologyOperations(int, void*)
{
    Mat elemento;
    int morphType;
    string titulo;

    if (morphElem == 0) {
        titulo = "Rect";
        morphType = MORPH_RECT;
    }
    else if (morphElem == 1) {
        titulo = "Cross";
        morphType = MORPH_CROSS;
    }
    else if (morphElem == 2) {
        titulo = "Ellipse";
        morphType = MORPH_ELLIPSE;
    }

    switch (morphOperator) {
        case 0:
            titulo = "EROSION -> " + titulo;
            elemento = getStructuringElement(morphType,
                Size(morphSize + 1, morphSize + 1),
                Point(morphSize, morphSize));
            erode(imagen, imagenDst, elemento);
        break;
        case 1:
            titulo = "DILATION -> " + titulo;
            elemento = getStructuringElement(morphType,
                Size(2 * morphSize + 1, 2 * morphSize + 1),
                Point(morphSize, morphSize));
            dilate(imagen, imagenDst, elemento);
        break;
        case 2:
            titulo = "TOP HAT -> " + titulo;
            elemento = getStructuringElement(morphElem,
                Size(morphSize + 1, morphSize + 1), Point(morphSize, morphSize));
            morphologyEx(imagen, imagenDst, 5, elemento);
        break;
        case 3:
            titulo = "BLACK HAT -> " + titulo;
            elemento = getStructuringElement(morphElem,
                Size(morphSize + 1, morphSize + 1), Point(morphSize, morphSize));
            morphologyEx(imagen, imagenDst, 6, elemento);
        break;
        case 4:
            titulo = "Original + (Top Hat – Black Hat) -> " + titulo;
            Mat imagenResta, imagenTopHat, imagenBlackHat;
            elemento = getStructuringElement(morphElem,
                Size(morphSize + 1, morphSize + 1), Point(morphSize, morphSize));
            morphologyEx(imagen, imagenTopHat, 5, elemento);
            morphologyEx(imagen, imagenBlackHat, 6, elemento);
            absdiff(imagenTopHat, imagenBlackHat, imagenResta); // realiza la resta
            add(imagen, imagenResta, imagenDst); // realiza la suma
        break;
    }
    // Agrega el titulo
    Mat imagenLabel(30, imagenDst.cols, CV_8U, Scalar(128, 128, 128));
    putText(imagenLabel, titulo, Point(10, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
    vconcat(imagenLabel, imagenDst, imagenDst);
    // muestra la imagen
    imshow("Morfologia", imagenDst);
}

void procesarImagen(string tituloW, Mat imagenN) {
    // Agrega los ventanas imagen 1
    imagen = imagenN.clone();
    namedWindow(tituloW, WINDOW_AUTOSIZE);
    moveWindow(tituloW, imagen.cols, 0);
    createTrackbar("Operacion:", tituloW, &morphOperator, maxOperator, morphologyOperations);
    createTrackbar("Element:", tituloW, &morphElem, maxElem, morphologyOperations);
    createTrackbar("Kernel", tituloW, &morphSize, maxKernelSize, morphologyOperations);
    morphologyOperations(0, 0);
}

void procesoMorfologia() {
    Mat imagen1 = imread(".\\data\\radiografia-abdomen.jpg", IMREAD_GRAYSCALE);
    Mat imagen2 = imread(".\\data\\fotonoticia.jpg", IMREAD_GRAYSCALE);
    Mat imagen3 = imread(".\\data\\radiografias-dentales.png", IMREAD_GRAYSCALE);
    resize(imagen1, imagen1, Size(), 0.5, 0.5);
    resize(imagen2, imagen2, Size(), 0.5, 0.5);
    resize(imagen3, imagen3, Size(), 0.5, 0.5);
    
    // Agrega los ventanas imagen
    procesarImagen("Morfologia", imagen2);
    waitKey(0);
}