#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
//#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/*
 * Goal: develop a system that is capable of automatically detecting trees in an image by creating a bounding box around each one.
 * In order to be recognized as a tree, it should be clearly visible and evident – small trees in the background, grass etc. do not need to be detected
 */

void equalize(Mat inputImg, Mat &inputEqualized);
void watershedSegmentation(Mat img, Mat &segmentedImg);
void Erosion(int, void*);
void templateSegmentation(Mat img, string pathTemplate);

void searchTemplateWithfeatures(Mat inputImg, string pathTemplate);
void extractFeatures(Mat img, vector<KeyPoint> &features, Mat &desciptors, int numFeatures);
void computeMatches(Mat templateDescriptors, Mat imageDescriptors, vector<DMatch> &matches, float ratio);
int printRectangle(Mat &frame, vector<Point2f> corners, Scalar color);

Mat src, erosion_dst, dilation_dst;
int erosion_elem = 0;
int erosion_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;



int main(int argc, char* argv[]) {
	//take input image from command line
	if (argc != 2) {
		cout << "Provide [image_path] as an argument in command line" << endl;
		cout << "The program stop!" << endl;
		return -1;
	}
	Mat inputImg, inputEqualized, segmentedImg;
	
	inputImg = imread(argv[1]);
	namedWindow("Input image", WINDOW_NORMAL);
	imshow("Input image", inputImg);
	waitKey();

	//equalization of the input image in HSV color code
	/*equalize(inputImg, inputEqualized);
	namedWindow("Equalized image", WINDOW_NORMAL);
	imshow("Equalized image", inputEqualized);
	waitKey(0);*/

	///watershedSegmentation
	//watershedSegmentation(inputImg, segmentedImg);

	///template matching with generalized hough trasform
	//templateSegmentation(inputImg, "../data/template/");

	///Matching features by template
	searchTemplateWithfeatures(inputImg, "../data/template/");
	//extractFeatures();
	
	return 0;
}

void searchTemplateWithfeatures(Mat inputImg, string pathTemplate) {
	vector<String> files;
	//Mat grayImg;
	//cvtColor(img, grayImg, COLOR_BGR2GRAY);
	utils::fs::glob(pathTemplate, "*.jpg", files);

	//extract from input image
	vector<KeyPoint> imgFeatures;
	Mat imgDescr;
	extractFeatures(inputImg, imgFeatures, imgDescr, 2000);

	for (int j = 0; j < files.size(); j++) {
		Mat temp;
		inputImg.copyTo(temp);
		Mat templImg = imread(files[j]);
		vector<KeyPoint> templFeatures;
		Mat templDescr;
		extractFeatures(templImg, templFeatures, templDescr, 1000);
		vector<DMatch> matches;
		computeMatches(templDescr, imgDescr, matches, 3);

		//refine with RANSAC algorithm
		vector<Point2f> templateImagePoints, frameImagePoints, objPointsRefined;
		Mat mask;
		for (int i = 0; i < matches.size(); i++) {
			templateImagePoints.push_back(templFeatures.at(matches[i].queryIdx).pt);
			frameImagePoints.push_back(imgFeatures.at(matches[i].trainIdx).pt);
		}
		//cout << "Find homography" << endl;
		Mat homography = findHomography(templateImagePoints, frameImagePoints, mask, RANSAC);
		for (int i = 0; i < mask.rows; i++) {
			if ((unsigned int)mask.at<uchar>(i)) {
				circle(temp, frameImagePoints[i], 5, Scalar((i * 10) % 255, (i * 20) % 255, (i * 50) % 255), 2);
				circle(templImg, templateImagePoints[i], 5, Scalar((i * 10) % 255, (i * 20) % 255, (i * 50) % 255), 2);
			}
		}
		Rect r = boundingRect(frameImagePoints);
		rectangle(temp, r, Scalar(0, 255, 0), 2);

		Mat outImgMatches = Mat(max(temp.rows,templImg.rows),temp.cols+templImg.cols,temp.type());
		temp.copyTo(outImgMatches(Range(0,temp.rows), Range(0, temp.cols)));
		templImg.copyTo(outImgMatches(Range(0, templImg.rows), Range(temp.cols, temp.cols + templImg.cols)));
		//drawMatches(templImg, templFeatures, inputImg, imgFeatures, matches, outImgMatches);
		namedWindow("Matches " + to_string(j), WINDOW_NORMAL);
		imshow("Matches " + to_string(j), outImgMatches);
		waitKey(0);
		
		/*vector<Point2f> rectangleCorners, rectTranlated;
		//matchObject(img, firstFrame, firstObjPoints, hom);
		rectangleCorners.push_back(Point2f(0, 0));
		rectangleCorners.push_back(Point2f(0, templImg.rows - 1));
		rectangleCorners.push_back(Point2f(templImg.cols - 1, 0));
		rectangleCorners.push_back(Point2f(templImg.cols - 1, templImg.rows - 1));
		perspectiveTransform(rectangleCorners, rectTranlated, homography);
		printRectangle(temp, rectTranlated, Scalar(255, 0, 0));

		namedWindow("Match " + to_string(j), WINDOW_NORMAL);
		imshow("Match " + to_string(j), temp);
		waitKey(0);*/

	}
}

void extractFeatures(Mat img, vector<KeyPoint> &features, Mat &desciptors, int numFeatures) {
	Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(numFeatures);
	detector->detectAndCompute(img, Mat(), features, desciptors);
}

void computeMatches(Mat templateDescriptors, Mat imageDescriptors, vector<DMatch> &matches, float ratio) {
	vector<DMatch> m;
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
	matcher->match(templateDescriptors, imageDescriptors, m);
	//refine with min distance
	float minDist = FLT_MAX;
	for (DMatch d : m) {
		if (d.distance < minDist) minDist = d.distance;
	}
	//cout << "Min dist = " << minDist << "ratio = " << ratio << "; over all = " << (ratio*minDist) << endl;
	for (DMatch d : m) {
		if (d.distance < ratio*minDist) matches.push_back(d);
	}

}


int printRectangle(Mat &frame, vector<Point2f> corners, Scalar color) {
	if (corners.size() != 4) return -1;
	circle(frame, corners[0], 3, Scalar(255, 255, 255), 1);
	circle(frame, corners[1], 3, Scalar(255, 255, 255), 1);
	circle(frame, corners[2], 3, Scalar(255, 255, 255), 1);
	circle(frame, corners[3], 3, Scalar(255, 255, 255), 1);
	line(frame, corners[0], corners[1], color, 2);
	line(frame, corners[1], corners[3], color, 2);
	line(frame, corners[0], corners[2], color, 2);
	line(frame, corners[3], corners[2], color, 2);
	return 0;
}


void equalize(Mat img, Mat &imgEq) {
	Mat imgHsv, channels[3], eqV;
	vector<Mat> hEqImage;
	//convert from BGR to HSV color space
	cvtColor(img, imgHsv, COLOR_BGR2HSV);
	//split each channel in a separate matrix
	split(imgHsv, channels);
	///equalization of the V channel
	//equalize the V histogram
	equalizeHist(channels[2], eqV);
	//stitch together all the three channels
	hEqImage = { channels[0], channels[1], eqV };
	merge(hEqImage, imgEq);
	//convert from HSV to BGR color space
	cvtColor(imgEq, imgEq, COLOR_HSV2BGR);
}

void templateSegmentation(Mat img, string pathTemplate) {
	vector<String> files;
	Mat grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	utils::fs::glob(pathTemplate, "*.jpg", files);
	//vector<Mat> templ;
	for (int j = 0; j < files.size(); j++) {
		cout << "Template " << files[j] << " - ";
		Mat templ = imread(files[j], IMREAD_GRAYSCALE);
		Ptr<GeneralizedHoughGuil> ghB = createGeneralizedHoughGuil();
		ghB->setMinDist(100);
		ghB->setLevels(360);
		ghB->setDp(2);
		ghB->setMaxBufferSize(1000);

		ghB->setMinAngle(0);
		ghB->setMaxAngle(10);
		ghB->setAngleStep(5);
		ghB->setAngleThresh(10000);

		ghB->setMinScale(0.5);
		ghB->setMaxScale(2);
		ghB->setScaleStep(0.1);
		ghB->setScaleThresh(1000);

		ghB->setPosThresh(100);

		ghB->setTemplate(templ);


		vector<Vec4f> position;
		ghB->detect(grayImg, position);
		cout << "Found : " << position.size() << " objects" << endl;
		if (position.size() == 0) continue;
		Mat out;
		img.copyTo(out);
		for (size_t i = 0; i < position.size(); ++i)
		{
			Point2f pos(position[i][0], position[i][1]);
			float scale = position[i][2];
			float angle = position[i][3];

			RotatedRect rect;
			rect.center = pos;
			rect.size = Size2f(img.cols * scale, img.rows * scale);
			rect.angle = angle;

			Point2f pts[4];
			rect.points(pts);

			line(out, pts[0], pts[1], Scalar(0, 0, 255), 3);
			line(out, pts[1], pts[2], Scalar(0, 0, 255), 3);
			line(out, pts[2], pts[3], Scalar(0, 0, 255), 3);
			line(out, pts[3], pts[0], Scalar(0, 0, 255), 3);
		}

		imshow("out"+to_string(j), out);
		waitKey(1);
	}


}


void watershedSegmentation(Mat img, Mat &segmentedImg) {
	Mat bw, bw1, bw2, dist, imgBlur;
	GaussianBlur(img, imgBlur, Size(7, 7), 500);
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); 
	Mat imgLaplacian;
	filter2D(imgBlur, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	imgBlur.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	// imshow( "Laplace Filtered Image", imgLaplacian );
	imshow("New Sharped Image", imgResult);

	cvtColor(imgBlur, bw, COLOR_BGR2GRAY);
	//imshow("Gray image", bw);
	//threshold(bw, bw, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	adaptiveThreshold(bw, bw1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
	imshow("Bin image - gaussian adaptive", bw1);
	//adaptiveThreshold(bw, bw2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);
	//imshow("Bin image - mean c adaptive", bw2);
	waitKey(0);
	src = bw1;
	namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
	createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
		&erosion_elem, max_elem,
		Erosion);
	createTrackbar("Kernel size:\n 2n +1", "Erosion Demo",
		&erosion_size, max_kernel_size,
		Erosion);
	Erosion(0, 0);
	waitKey();

	/*imshow("TEMP ", src);
	waitKey();*/

	distanceTransform(erosion_dst, dist, DIST_L2, 3);
	normalize(dist, dist, 0, 1., NORM_MINMAX);
	imshow("Distance trasform image", dist);
	waitKey(0);

	threshold(dist, dist, .6, 1., THRESH_BINARY);
	Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
	dilate(dist, dist, kernel1);
	imshow("Peaks", dist);
	waitKey(0);

	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
	vector<vector<Point> > contours;
	findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(dist.size(), CV_32SC1);
	// Draw the foreground markers
	
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, i, Scalar(i + 1), -1);
		//drawContours(markers, contours, i, Scalar(i+10,i+50, i+25), -1);
		//Rect r = boundingRect(contours[i]);
		//rectangle(markers, r, Scalar(i + 100 % 255), 2);
		cout << i << endl;
	}
	// Draw the background marker
	//circle(markers, Point(5, 5), 3, Scalar(255), -1);
	//imshow("Markers", markers * 10000);
	//waitKey();
	//markers.convertTo(markers, CV_32SC1);
	//cout << markers.type() << endl;

	// Perform the watershed algorithm
	watershed(img, markers);
	
	Mat mark;
	markers.convertTo(mark, CV_8U);
	bitwise_not(mark, mark);
	namedWindow("Markers_v2", WINDOW_NORMAL);
	imshow("Markers_v2", mark);
	waitKey();
	int compCount = markers.rows;
	vector<Vec3b> colorTab;
	for ( int i = 0; i < compCount; i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);
		colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	Mat wshed(markers.size(), CV_8UC3);
	// paint the watershed image
	for (int i = 0; i < markers.rows; i++)
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index == -1)
				wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			else if (index <= 0 || index > compCount)
				wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			else
				wshed.at<Vec3b>(i, j) = colorTab[index - 1];
		}
	wshed = wshed * 0.5 + img * 0.5;
	imshow("watershed transform", wshed);
	waitKey();
}

void Erosion(int, void*)
{
	int erosion_type = 0;
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(src, erosion_dst, element);
	imshow("Erosion Demo", erosion_dst);
}