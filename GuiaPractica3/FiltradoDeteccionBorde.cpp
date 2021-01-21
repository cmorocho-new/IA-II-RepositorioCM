// Librerias estandar
#include <iostream>
#include <cstdlib>
#include <cmath>

// Librerias de opencv
#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio/videoio.hpp>

// Incluyo la cabecera
#include "FiltradoHeader.h"

// Espacio de nombres
using namespace std;
using namespace cv;

Mat generarRuido(Mat imagen, int rango, bool esSal) {
    Mat imagenRuido = imagen.clone();
    int total = (int)(rango * imagenRuido.rows * imagenRuido.cols) / 100;
    int cont = 0, row = 0, col = 0;
    srand(time(0));
    while (cont < total) {
        cont++;
        row = rand() % imagenRuido.rows;
        col = rand() % imagenRuido.cols;
        imagenRuido.at<uchar>(row, col) = esSal ? 255 : 0;
    }
    return imagenRuido;
}

void aplicarFiltros(Mat imagen, Mat& imagenFiltro, int k[2]) {
    Mat imagenFiltroM;
    Mat imagenFiltroG;
    if (k[0] > 0) {
        int valorkM = k[0] % 2 == 1 ? k[0] : k[0] - 1;
        medianBlur(imagen, imagenFiltroM, valorkM);
    } 
    else {
        imagenFiltroM = imagen.clone();
    }
    if (k[1] > 0) {
        int valorkG = k[1] % 2 == 1 ? k[1] : k[1] - 1;
        GaussianBlur(imagen, imagenFiltroG, Size(valorkG, valorkG), 0, 0);
    }
    else {
        imagenFiltroG = imagen.clone();
    }
    Mat separador(imagen.rows, 3, CV_8U, Scalar(243, 156, 18));
    hconcat(imagenFiltroM, separador, imagenFiltroM);
    hconcat(imagenFiltroM, imagenFiltroG, imagenFiltro);
}

void aplicarDetectorBordes(Mat imagen, Mat& imagenBordes) {
    Mat imagenBordeL;
    Mat imagenBordeC;
    int umbral = 70;
    double ratio = 3.;

    // Aplicamos el filto de canny
    Canny(imagen, imagenBordeC, umbral, umbral * ratio, 3);

    // Aplicamos el filto de laplace
    Laplacian(imagen, imagenBordeL, CV_16S, 3);
    convertScaleAbs(imagenBordeL, imagenBordeL);

    // Insertamos los titulos
    Mat imagenLabel1(30, imagenBordeC.cols, CV_8U, Scalar(128, 128, 128));
    putText(imagenLabel1, "Borde Canny con suavisado ( Median y Gaussian )",
        Point(10, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
    vconcat(imagenLabel1, imagenBordeC, imagenBordeC);
    Mat imagenLabel2(30, imagenBordeL.cols, CV_8U, Scalar(128, 128, 128));
    putText(imagenLabel2, "Borde Laplace con suavisado ( Median y Gaussian )", 
        Point(10, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
    vconcat(imagenLabel2, imagenBordeL, imagenBordeL);

    // Concatena las dos imagenes resultantes
    vconcat(imagenBordeC, imagenBordeL, imagenBordes);
}

void procesarVideo() {
    VideoCapture video(".\\data\\OneMinute.mkv");
    if (video.isOpened()) {
        // imagenes
        Mat imagenframe;
        Mat imagenRuido;
        // Imagenes para el ruido
        Mat imagenRuidoSP;
        // Imagenes para el filtro
        Mat imagenFiltroSP;
        // Imagenes para e¿los bordes
        Mat imagenBordesSP;

        // Titulos de las ventanas
        string titulo;
        string tituloWinRuidoSP = "Video con ruidos (Sal/Pimienta)";
        string tituloWinFiltroSP = "Video con filtros";
        string tituloWinBordeSP = "Video con bordes";

        // Creamos las ventanas
        namedWindow(tituloWinRuidoSP, WINDOW_AUTOSIZE);
        namedWindow(tituloWinFiltroSP, WINDOW_AUTOSIZE);
        namedWindow(tituloWinBordeSP, WINDOW_AUTOSIZE);

        int rangoSP = 30, tipo = 0;
        int kSP[2] = { 0, 0 };

        // Agregamos los trackbar para los ruidos sal y pimienta
        createTrackbar("Tipo", tituloWinRuidoSP, &tipo, 1);
        createTrackbar("Rango", tituloWinRuidoSP, &rangoSP, 80);
        
        // Agregamos los trackbar para los filtros median y gaussian
        createTrackbar("K Median", tituloWinFiltroSP, &kSP[0], 40);
        createTrackbar("K Gaussian", tituloWinFiltroSP, &kSP[1], 40);

        while (true) {
            video >> imagenframe;
            
            if (imagenframe.empty()) break;

            // Minimiza y cambia a escala de grises
            resize(imagenframe, imagenframe, Size(), 0.3, 0.3);
            cvtColor(imagenframe, imagenframe, COLOR_BGR2GRAY);

            // Agrega el ruido de sal y pimienta
            if (tipo == 0) {
                imagenRuidoSP = generarRuido(imagenframe, rangoSP);
                titulo = "IMAGEN CON RUIDO DE SAL";
            }
            else {
                imagenRuidoSP = generarRuido(imagenframe, rangoSP, false);
                titulo = "IMAGEN CON RUIDO DE PIMIENTA";
            }
            // Agrega el titulo
            Mat imagenLabel(30, imagenRuidoSP.cols, CV_8U, Scalar(128, 128, 128));
            putText(imagenLabel, titulo, Point(10, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            vconcat(imagenLabel, imagenRuidoSP, imagenRuido);
            imshow(tituloWinRuidoSP, imagenRuido);

            // Aplica los filtros median y gaussian a ruido con Sal y Pimienta
            aplicarFiltros(imagenRuidoSP, imagenFiltroSP, kSP);
            imshow(tituloWinFiltroSP, imagenFiltroSP);

            // Aplicamos los deectores de bordes Laplace y Canny a filtro con Sal y Pimienta
            aplicarDetectorBordes(imagenFiltroSP, imagenBordesSP);
            imshow(tituloWinBordeSP, imagenBordesSP);

            if (waitKey(23) == 27) break;
        }
        waitKey(0);
        destroyAllWindows();
    }

}
