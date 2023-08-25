#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat sobelfilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = sobelfilter(input_gray); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)

	Mat output = Mat::zeros(row, col, input.type());

	float Sx[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};
	float Sy[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum_x = 0; float sum_y = 0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
				
					int idx_x = i + a;
					int idx_y = j + b;

					// Use mirroring boundary process
					if (idx_x < 0) idx_x = 0;
					else if (idx_x >= row) idx_x = row -1 ;
					if (idx_y < 0) idx_y = 0;
					else if (idx_y >= col) idx_y = col -1 ;
			
					sum_x += input.at<G>(idx_x, idx_y) * Sx[a + 1][b + 1];					
					sum_y += input.at<G>(idx_x, idx_y) * Sy[a + 1][b + 1];
				}
			  
			}// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
		output.at<G>(i, j) = (G)sqrt(sum_x*sum_x + sum_y*sum_y);

		}
	}
	return output;
}