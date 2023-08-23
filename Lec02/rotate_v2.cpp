#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T> // template : 함수나 클래스를 개별적으로 다시 작성하지 않아도, 여러 자료 형으로 사용할 수 있도록 하게 만들어 놓은 틀.
Mat myrotate(const Mat input, float angle, const char* opt); // myrotate 함수 정의, input : 입력 사진, angle : 회전 각(반시계), opt : nearest or bilinear

int main()
{
	Mat input, rotated; // input : 입력 사진, rotated : 회전된 결과 사진
	
	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "nearest"); 

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated); 

	waitKey(0);

	return 0;
	
}

template <typename T>
/*
"output.at<T>(i, j)"를 "output.at<Vec3b>(i,j)" 대신 사용하는 이유는 "myrotate" 함수가 템플릿화되어 있기 때문입니다. 
템플릿 매개변수 T는 입력 및 출력 이미지의 데이터 형식을 지정하는 데 사용됩니다.

main 함수에서 입력 이미지는 "Mat input"으로 읽어들여지며, 이는 데이터 형식이 "Vec3b"(3바이트 벡터, 즉 24비트)입니다. 
그러나 "myrotate" 함수를 호출할 때 템플릿 인수를 사용하여 데이터 형식을 명시적으로 "Vec3b"로 지정했습니다. 따라서 "myrotate"에서 반환된 출력 이미지도 "Vec3b" 데이터 형식을 가지게 됩니다.

만약 "myrotate" 함수에서 "output.at<T>(i,j)" 대신 "output.at<Vec3b>(i,j)"를 사용하면, 이 코드는 "Vec3b" 데이터 형식을 가진 이미지에 대해서는 작동하지만,
다른 데이터 형식을 가진 이미지에 대해서는 실패하게 됩니다. "output.at<T>(i,j)"를 사용하면 함수에서는 호출할 때 데이터 형식이 올바르게 지정되는 한 모든 데이터 형식의 이미지를 처리할 수 있습니다.
*/
Mat myrotate(const Mat input, float angle, const char* opt) { // 이미지 회전 함수 (input: 입력 이미지, angle: 회전각도, opt: nearest or bilinear)

	// input에 대한 입력 처리
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180; // 라디안 변환

	// 회전한 이미지를 담을 큰 행렬의 행(sq_row), 열(sq_col) 계산
	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type()); 

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) { // output 픽셀 위치 (i,j)

			// inverse warping을 위해 input픽셀 위치 (y,x) 계산 & rotated 이미지 잘리지 않기 위한 평행이동 
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) { // input 픽셀 위치가 유효한지 확인

				if (!strcmp(opt, "nearest")) {	// nearest
					int X = round(x); int Y = round(y); // round the coordinates
					output.at<T>(i, j) = input.at<T>(Y,X); // copy the source pixel value to the destination
				}
				else if (!strcmp(opt, "bilinear")) { // bilinear 
					int X = floor(x); int Y = floor(y); 
					float lambda = x - X; float mu = y - Y;
					output.at<T>(i, j) = lambda * ( mu * input.at<T>(Y+ 1, X+ 1) + (1 - mu) * input.at<T>(Y + 1, X))
							+ (1 - lambda) * (mu * input.at<T>(Y, X+1) + (1 - mu) * input.at<T>(Y,X));
	
				}
			}
		}
	}
	return output;
}