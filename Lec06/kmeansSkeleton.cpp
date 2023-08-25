#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

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

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);


	// RGB
	Mat samples(input.rows * input.cols, 3, CV_32F);
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * input.rows, z) = input.at<Vec3b>(y, x)[z];


	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			// Find for each pixel of each channel of the output image the intensity of the cluster center.
			new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}

	imshow("clustered image", new_image);


	// grayscale
	Mat samples_gray(input_gray.rows * input_gray.cols, 1, CV_32F);
	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++)
				samples_gray.at<float>(y + x * input_gray.rows) = input_gray.at<uchar>(y, x);


	Mat labels_gray;
	Mat centers_gray;
	kmeans(samples_gray, clusterCount, labels_gray, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers_gray);

	Mat new_image_gray(input_gray.size(), input_gray.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels_gray.at<int>(y + x * input_gray.rows);
			new_image_gray.at<uchar>(y, x) = centers_gray.at<float>(cluster_idx);
		}

	imshow("clustered image_gray", new_image_gray);

	waitKey(0);

	return 0;
}