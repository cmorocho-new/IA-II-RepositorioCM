# pragma once
// Librerias de opencv
#include <opencv2/core/core.hpp> 

// Espacio de nombres
using namespace std;
using namespace cv;

/**
	Genera el ruido de sal o pimienta de una imagen
**/
Mat generarRuido(Mat imagen, int rango=30, bool esSal=true);

/**
	Aplica los filtros Median y Gaussian a una imagen
**/
void aplicarFiltros(Mat imagen, Mat& imagenFiltro, int k[2]);

/**
	Aplica los detectores de bordes Laplace y Canny a una imagen
**/
void aplicarDetectorBordes(Mat imagen, Mat& imagenBordes);

/**
	Procesa el video aplicacnod los filtros y bordes
**/
void procesarVideo();