#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}

	return sqrt(sum);
}
pair<int, double> nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int firstNeighbor = -1;
	int secondNeighbor = -1;
	double firstMinDist = 1e6;
	double secondMinDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);  // each row of descriptor
		double dist = euclidDistance(vec, v);

		if (dist < firstMinDist) {
			secondMinDist = firstMinDist;
			secondNeighbor = firstNeighbor;

			firstMinDist = dist;
			firstNeighbor = i;
		}
		else if (dist < secondMinDist) {
			secondMinDist = dist;
			secondNeighbor = i;
		}
	}

	double temp = firstMinDist / secondMinDist;
	return make_pair(firstNeighbor, temp);
}
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1, vector<KeyPoint>& keypoints2, Mat& descriptors2, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold)
{
	for (int i = 0; i < descriptors1.rows; ++i) {

		Mat descriptor = descriptors1.row(i);
		int nn = nearestNeighbor(descriptor, keypoints2, descriptors2).first;

		if (crossCheck) {
			Mat descriptor2_nn = descriptors2.row(nn);
			int nearest = nearestNeighbor(descriptor2_nn, keypoints1, descriptors1).first;
			if (i != nearest) continue;
		}

		if (ratio_threshold) {
			double temp = nearestNeighbor(descriptor, keypoints2, descriptors2).second;
			if (temp > RATIO_THR) continue;
		}

		srcPoints.push_back(keypoints1[i].pt);
		dstPoints.push_back(keypoints2[nn].pt);
	}
}
Mat cal_affine(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points,	bool ransac){
	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);

	Mat M_trans, temp, affineM;
	if (ransac) {
		vector<float> inliers(srcPoints.size(), 0);
		affineM = findHomography(srcPoints, dstPoints, CV_RANSAC, 3.0, inliers);
	}
	else {
	// initialize matrix
		for (int i = 0; i < number_of_points; i++) {
			M.at<float>(2 * i, 0) = srcPoints[i].x;		M.at<float>(2 * i, 1) = srcPoints[i].y;		M.at<float>(2 * i, 2) = 1;
			M.at<float>(2 * i + 1, 3) = srcPoints[i].x;			M.at<float>(2 * i + 1, 4) = srcPoints[i].y;		M.at<float>(2 * i + 1, 5) = 1;
			b.at<float>(2 * i) = dstPoints[i].x;		b.at<float>(2 * i + 1) = dstPoints[i].y;
		}
		transpose(M, M_trans);		
		invert(M_trans * M, temp);	
		affineM = temp * M_trans * b;		
	}

	return affineM;
}
void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha) {

	int col = I_f.cols;
	int row = I_f.rows;

	// I2 is already in I_f by inverse warping
	for (int i = 0; i < I1.rows; i++) {
		for (int j = 0; j < I1.cols; j++) {
			bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0,0,0) ? true : false; // cond_12 = I1과 I2의 겹치는 내부이면 true, 아니면 false  

			if (cond_I2) // 겹치는 영역 픽셀값 I_f = alpha*I1 + (1-alpha)*I2'
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);
			else		 // 안 겹치는 영역 픽셀값은 원래 값 대입
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j);
		}
	}
}

int main() {

	Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);

	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	detector->detect(input1_gray, keypoints1);
	extractor->compute(input1_gray, keypoints1, descriptors1);
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.\n", srcPoints.size());

	bool ransac = false;

	Mat A12 = cal_affine(srcPoints, dstPoints, srcPoints.size(), ransac);
	Mat A21 = cal_affine(dstPoints, srcPoints, dstPoints.size(), ransac);
	
	const float I1_row = input1.rows;
	const float I1_col = input1.cols;
	const float I2_row = input2.rows;
	const float I2_col = input2.cols;

	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));
	
	int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));
	int bound_b = (int)round(max(I1_row - 1, max(p2.y, p3.y)));
	int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_r = (int)round(max(I1_col - 1, max(p3.x, p4.x)));

	input1.convertTo(input1, CV_32FC3, 1.0 / 255);  // covertTo 함수 : 행렬 타입 변환 
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);	// CV_32FC3 : 32비트 float 3채널 

	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));

	for (int i = bound_u; i <= bound_b; i++) {
		for (int j = bound_l; j <= bound_r; j++) {
			float x = A12.at<float>(0) * j + A12.at<float>(1) * i + A12.at<float>(2) - bound_l;
			float y = A12.at<float>(3) * j + A12.at<float>(4) * i + A12.at<float>(5) - bound_u;

			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * input2.at<Vec3f>(y2, x2) + (1 - mu) * input2.at<Vec3f>(y1, x2)) +
				(1 - lambda) * (mu * input2.at<Vec3f>(y2, x1) + (1 - mu) * input2.at<Vec3f>(y1, x1));
		}
	}

	// image stitching with blend
	blend_stitching(input1, input2, I_f, bound_l, bound_u, 0.5);

	namedWindow("result");
	imshow("result", I_f);


	waitKey(0);

	return 0;
}