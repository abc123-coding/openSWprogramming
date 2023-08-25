#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T> // template : �Լ��� Ŭ������ ���������� �ٽ� �ۼ����� �ʾƵ�, ���� �ڷ� ������ ����� �� �ֵ��� �ϰ� ����� ���� Ʋ.
Mat myrotate(const Mat input, float angle, const char* opt); // myrotate �Լ� ����, input : �Է� ����, angle : ȸ�� ��(�ݽð�), opt : nearest or bilinear

int main()
{
	Mat input, rotated; // input : �Է� ����, rotated : ȸ���� ��� ����
	
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
"output.at<T>(i, j)"�� "output.at<Vec3b>(i,j)" ��� ����ϴ� ������ "myrotate" �Լ��� ���ø�ȭ�Ǿ� �ֱ� �����Դϴ�. 
���ø� �Ű����� T�� �Է� �� ��� �̹����� ������ ������ �����ϴ� �� ���˴ϴ�.

main �Լ����� �Է� �̹����� "Mat input"���� �о�鿩����, �̴� ������ ������ "Vec3b"(3����Ʈ ����, �� 24��Ʈ)�Դϴ�. 
�׷��� "myrotate" �Լ��� ȣ���� �� ���ø� �μ��� ����Ͽ� ������ ������ ��������� "Vec3b"�� �����߽��ϴ�. ���� "myrotate"���� ��ȯ�� ��� �̹����� "Vec3b" ������ ������ ������ �˴ϴ�.

���� "myrotate" �Լ����� "output.at<T>(i,j)" ��� "output.at<Vec3b>(i,j)"�� ����ϸ�, �� �ڵ�� "Vec3b" ������ ������ ���� �̹����� ���ؼ��� �۵�������,
�ٸ� ������ ������ ���� �̹����� ���ؼ��� �����ϰ� �˴ϴ�. "output.at<T>(i,j)"�� ����ϸ� �Լ������� ȣ���� �� ������ ������ �ùٸ��� �����Ǵ� �� ��� ������ ������ �̹����� ó���� �� �ֽ��ϴ�.
*/
Mat myrotate(const Mat input, float angle, const char* opt) { // �̹��� ȸ�� �Լ� (input: �Է� �̹���, angle: ȸ������, opt: nearest or bilinear)

	// input�� ���� �Է� ó��
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180; // ���� ��ȯ

	// ȸ���� �̹����� ���� ū ����� ��(sq_row), ��(sq_col) ���
	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type()); 

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) { // output �ȼ� ��ġ (i,j)

			// inverse warping�� ���� input�ȼ� ��ġ (y,x) ��� & rotated �̹��� �߸��� �ʱ� ���� �����̵� 
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) { // input �ȼ� ��ġ�� ��ȿ���� Ȯ��

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