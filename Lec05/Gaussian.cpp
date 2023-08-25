#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

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

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	
	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Gaussianfilter_Gray(noise_Gray, 3, 30, 30, "zero-padding");
	Mat Denoised_Gray2 = Bilateralfilter_Gray(noise_Gray, 3, 30, 30, 0.3, "zero-padding");
	
	Mat Denoised_RGB = Gaussianfilter_RGB(noise_RGB, 3, 10, 10, "zero-padding");
	Mat Denoised_RGB2 = Bilateralfilter_RGB(noise_RGB, 3, 10, 10, 0.3, "zero-padding");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised2 (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised2 (Grayscale)", Denoised_Gray2);
	
	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	namedWindow("Denoised2 (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised2 (RGB)", Denoised_RGB2);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	float denom = 0.0;


	Mat output = Mat::zeros(row, col, input.type());
	Mat	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for (int x = -n; x <= n; x++) { // for each kernel window
		for (int y = -n; y <= n; y++) {
			float value = exp(-(pow(x, 2) / (2 * pow(sigma_t, 2))) - (pow(y, 2) / (2 * pow(sigma_s, 2))));
			kernel.at<float>(x + n, y + n) = value;
			denom += value; 
		}
	}

	for (int x = -n; x<= n; x++) {
		for (int y = -n; y <= n; y++) {
			kernel.at<float>(x + n, y + n) /= denom;
		}
	}

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				float sum = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0))
							sum += kernel.at<float>(x + n, y + n) * input.at<G>(i+x, j+y);
					}
				}
				output.at<G>(i, j) = (G)sum;
			}

			else if (!strcmp(opt, "mirroring")) {
				int tempa, tempb; float sum = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						sum += kernel.at<float>(x + n, y + n) * input.at<G>(tempa, tempb);
					}
				}
				output.at<G>(i, j) = (G)sum;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0, sum2 = 0.0;
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sum1 += kernel.at<float>(x + n, y + n) * input.at<G>(i + x, j + y);
							sum2 += kernel.at<float>(x + n, y + n);
						}
					}
				}
				output.at<G>(i, j) = (G)sum1 / sum2;
			}
		}
	}
	return output;
}

Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);

	Mat output = Mat::zeros(row, col, input.type());
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for(int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float denom = 0.0;

			for (int x = -n; x <= n; x++) {
				for (int y = -n; y <= n; y++) {
					if ((i + x >= 0) && (i + x < row) && (j + y >= 0) && (j + y < col)) {
						float value = exp(-(pow(x, 2) / (2 * pow(sigma_t, 2))) - (pow(y, 2) / (2 * pow(sigma_s, 2)))) * exp(-pow((input.at<G>(i, j) - input.at<G>(i + x, j + y)), 2) / (2 * pow(sigma_r, 2)));
						kernel.at<float>(x + n, y + n) = value;
						denom += value;
					}
				}
			}
			for (int x = -n; x <= n; x++) {
				for (int y = -n; y <= n; y++)
					kernel.at<float>(x + n, y + n) /= denom;
			}

			float sum = 0.0;

			if (!strcmp(opt, "zero-padding")) {
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x >= 0) && (i + x < row) && (j + y >= 0) && (j + y < col)) {
							sum += kernel.at<float>(x + n, y + n) * input.at<G>(i + x, j + y) ;
						}
					}
				}
				output.at<G>(i, j) = (G)sum ;
			}

			else if (!strcmp(opt, "mirroring")) {
				int tempa, tempb; 
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						sum += kernel.at<float>(x + n, y + n) * input.at<G>(tempa, tempb);
					}
				}
				output.at<G>(i, j) = (G)sum;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x >= 0) && (i + x < row) && (j + y >= 0) && (j + y < col)) {
							sum += kernel.at<float>(x + n, y + n) * input.at<G>(i + x, j + y);
							sum1 += kernel.at<float>(x + n, y + n);
						}
					}
					output.at<G>(i, j) = (G)sum;
				}
	
			}
		}
	}
	return output;
}
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char* opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	float denom = 0.0;
	int channel = input.channels();

	Mat output = Mat::zeros(row, col, input.type());
	Mat	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for (int x = -n; x <= n; x++) { // for each kernel window
		for (int y = -n; y <= n; y++) {
			float value = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(x + n, y + n) = value;
			denom += value;
		}
	}

	for (int x = -n; x <= n; x++) {
		for (int y = -n; y <= n; y++) {
			kernel.at<float>(x + n, y + n) /= denom;
		}
	}

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			double sumR = 0, sumG = 0, sumB = 0;
			if (!strcmp(opt, "zero-padding")) {
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sumR += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[0];
							sumG += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[1];
							sumB += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[2];
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sumR;
				output.at<Vec3d>(i, j)[1] = sumG;
				output.at<Vec3d>(i, j)[2] = sumB;
			}

			else if (!strcmp(opt, "mirroring")) {
				int tempa, tempb;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						sumR += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(tempa, tempb)[0];
						sumG += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(tempa, tempb)[1];
						sumB += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(tempa, tempb)[2];

					}
				}
				output.at<Vec3d>(i, j)[0] = sumR;
				output.at<Vec3d>(i, j)[1] = sumG;
				output.at<Vec3d>(i, j)[2] = sumB;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				double sum = 0;
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sumR += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[0];
							sumG += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[1];
							sumB += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[2];
							sum += kernel.at<float>(x + n, y + n);
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sumR / sum;
				output.at<Vec3d>(i, j)[1] = sumG / sum;
				output.at<Vec3d>(i, j)[2] = sumB / sum;
			} 
		}
	}
	return output;
}
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	float denom = 0.0;
	int channel = input.channels();

	Mat output = Mat::zeros(row, col, input.type());
	Mat	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float denom = 0.0;

		for (int x = -n; x <= n; x++) { // for each kernel window
			for (int y = -n; y <= n; y++) {
				if ((i + x >= 0) && (i + x < row) && (j + y >= 0) && (j + y < col)) {

					float diff_r = input.at<Vec3d>(i, j)[0] - input.at<Vec3d>(i + x, j + y)[0];
					float diff_g = input.at<Vec3d>(i, j)[1] - input.at<Vec3d>(i + x, j + y)[1];
					float diff_b = input.at<Vec3d>(i, j)[2] - input.at<Vec3d>(i + x, j + y)[2];
					float distance = sqrt(diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);
					
					float value = exp(-(pow(x, 2) / (2 * pow(sigma_t, 2))) - (pow(y, 2) / (2 * pow(sigma_s, 2)))) * exp(-distance / (2 * pow(sigma_r, 2)));
					kernel.at<float>(x + n, y + n) = value;
					denom += value;
				}
			}
		}

			for (int x = -n; x <= n; x++) {
				for (int y = -n; y <= n; y++) {
					kernel.at<float>(x + n, y + n) /= denom;
				}
			}

			double sumR = 0, sumG = 0, sumB = 0;

			if (!strcmp(opt, "zero-padding")) {
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sumR += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[0];
							sumG += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[1];
							sumB += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[2];
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sumR;
				output.at<Vec3d>(i, j)[1] = sumG;
				output.at<Vec3d>(i, j)[2] = sumB;
			}

			else if (!strcmp(opt, "mirroring")) {
				int tempa, tempb;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						sumR += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(tempa, tempb)[0];
						sumG += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(tempa, tempb)[1];
						sumB += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(tempa, tempb)[2];

					}
				}
				output.at<Vec3d>(i, j)[0] = sumR;
				output.at<Vec3d>(i, j)[1] = sumG;
				output.at<Vec3d>(i, j)[2] = sumB;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				double sum = 0;
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sumR += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[0];
							sumG += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[1];
							sumB += kernel.at<float>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[2];
							sum += kernel.at<float>(x + n, y + n);
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sumR / sum;
				output.at<Vec3d>(i, j)[1] = sumG / sum;
				output.at<Vec3d>(i, j)[2] = sumB / sum;
			}
		}
	}
	return output;
}