// opencv_test.cpp : Defines the entry point for the console application.


#include "stdafx.h" //  자주 쓰이는 헤더들 여러 개를 한꺼번에 미리 컴파일
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray, output;
	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
		
	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	Canny(input_gray, output, 50, 150);

	namedWindow("Canny", WINDOW_AUTOSIZE);
	imshow("Canny", output);

	waitKey(0);

	return 0;
}
