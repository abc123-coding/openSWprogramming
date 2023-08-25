#include <iostream>
#include <opencv2/opencv.hpp>

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

Mat adaptive_thres(const Mat input, int n, float b);

int main() {

	Mat input = imread("writing.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;

	cvtColor(input, input_gray, CV_RGB2GRAY); // Converting image to gray

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	output = adaptive_thres(input_gray, 2, 0.9); //Fix with uniform mean filtering with zero paddle

	namedWindow("Adaptive_threshold", WINDOW_AUTOSIZE);
	imshow("Adaptive_threshold", output);


	waitKey(0);

	return 0;
}


Mat adaptive_thres(const Mat input, int n, float bnumber) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);

	// Initializing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	float kernelvalue = kernel.at<float>(0, 0);  // To simplify, as the filter is uniform. All elements of the kernel value are same.

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) { //for each pixel in the output
		for (int j = 0; j < col; j++) {

			float sum1 = 0.0;

			for (int s = -n; s <= n; s++) {
				for (int t = -n; t <= n; t++) {
					int row_index = i + s;
					int col_index = j + t;
					if (row_index >= 0 && col_index >= 0 && row_index < row && col_index < col) { // zero-padding
						sum1 += input.at<G>(row_index, col_index) * kernelvalue;
					}
				}
			}

			float th = bnumber * sum1; // value of threshold : th 
			
			if (input.at<G>(i, j) > th) output.at<G>(i, j) = 255;
			else output.at<G>(i, j) = 0;
			
		}
	}
	return output;
}