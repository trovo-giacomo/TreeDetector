
#include "tree_detection_util.h"

/*
 * Author: Giacomo Trovo'
 * Computer Vision course - 9CFU - UNIPD
 * 1st session - 2nd appeal - 9th July 2020
 * tree_detection_util.cpp - class that contains alla the usefull methods to carry on the goal of detecting trees in an image
 */

 //template matching - windows size
#define WIN_ROWS 60
#define WIN_COLS 60

// Definition of status code for the finite state machine for the post-processing template matching results 
#define S_UNKNOWN 0 
#define S_FST_MTD 1
#define S_SND_MTD 2

//Threshold on MSSIM score
#define THRESHOLD_MSSIM 0.195

using namespace std;
using namespace cv;
using namespace cv;

//status varaible for the finite state machine set up for the evaluation of the best candidate among the selected ones - more detail afterwards
int TreeUtil::status = S_UNKNOWN;


/******* FUNCTIONS FOR INITIALIZATION ***********/
struct DataCanny {
	Mat src, dstBlur, dstCanny;
	int sigma, th, tl;
	string fileName;
};

void TreeUtil::findCanny(Mat inputImg, string pathTemplate) {
	vector<String> files;
	utils::fs::glob(pathTemplate, "*.*", files);
	for (int i = 5; i < files.size(); i++) {
		DataCanny *d = new DataCanny();
		namedWindow(files[i], WINDOW_NORMAL);
		d->src = imread(files[i]);
		resize(d->src, d->src, Size(WIN_COLS, WIN_ROWS));
		d->fileName = files[i];
		imshow(files[i], d->src);
		createTrackbar("Blur[*10]: ", files[i], &(d->sigma), 1000, TreeUtil::computeCanny, d);
		createTrackbar("Th: ", files[i], &(d->th), 1000, TreeUtil::computeCanny, d);
		createTrackbar("Tl: ", files[i], &(d->tl), 1000, TreeUtil::computeCanny, d);
		waitKey();

		imwrite("../data/cannyTemplate/t" + to_string(i) + ".jpg", d->dstCanny);
	}
}

void TreeUtil::computeCanny(int, void* data) {
	DataCanny *d = (DataCanny *)data;
	double sig = d->sigma / 10;
	GaussianBlur(d->src, d->dstBlur, Size(5, 5), sig, sig);
	Canny(d->dstBlur, d->dstCanny, d->th, d->tl);
	namedWindow(d->fileName, WINDOW_NORMAL);
	imshow(d->fileName, d->dstCanny);
}
/******* END FUNCTIONS FOR INITIALIZATION ***********/

/*
 * given an input image find a candidate rectangle for each temlate image stored in data/cannyTemplate folder
 * @param inputImg, image from which we want to detect a tree
 * @param pathTemplate, path to the folder in which are contained the template images (edges of trees)
 * return a list of rectangles, one for each template with reference to the input image that best match the each template
 */
vector<TreeUtil::treeData> TreeUtil::searchTemplateWithCanny(Mat inputImg, string pathTemplate) {

	// import all template images (already processed by the initialization step with canny edge detector)
	vector<String> files;
	utils::fs::glob(pathTemplate, "*.*", files);
	vector<double> scales = { 1.5, 2.0, 2.5, 3, 4, 5, 6, 10 }; //scales at which the template matching is performed
	int numMatches;

	vector<TreeUtil::treeData> treesDetected; //vector that will contain the result of template matching, all trees among all the template images

	for (int j = 0; j < files.size(); j++) {

		Mat tImg = imread(files[j], IMREAD_GRAYSCALE);
		if (tImg.size() != Size(WIN_COLS, WIN_ROWS))	resize(tImg, tImg, Size(WIN_COLS, WIN_ROWS));

		//for each template image -> perform template matching at several scales
		vector<TreeUtil::treeData> treeDataScales; //vector that will contain the best results for each scale
		double ratioSel = 0;

		for (int i = 0; i < scales.size(); i++) {
			//cout << "Sliding win file " << files[j] << " - scale: " << scales[i] << endl;

			TreeUtil::treeData data;
			data.fileName = files[j];
			slidingWindow(inputImg, tImg, scales[i], data); //perform template matching at the given scale - fill data structure with useful information and statistics

			double sc = data.scale;
			Rect treeWindow(data.rect.x*sc, data.rect.y*sc, data.rect.width*sc, data.rect.height*sc); // calculate the effective rectangle 
			data.rect = treeWindow;
			treeDataScales.push_back(data);
		}

		TreeUtil::treeData selectTree;
		//Select a rectangle (tree) according to the highest score (template matching one) within the best candidate trees at all scales
		selectTree.score = 0;
		for (TreeUtil::treeData td : treeDataScales) {
			if (td.score > selectTree.score) selectTree = td;
		}
		treesDetected.push_back(selectTree);

	}


	//count for each rectangle, the number of overlapping rectangle with reference to the central point
	vector<TreeUtil::treeData> final_detected_trees;
	for (int i = 0; i < treesDetected.size(); i++) {
		int cx = treesDetected[i].rect.x + (treesDetected[i].rect.width / 2.0);
		int cy = treesDetected[i].rect.y + (treesDetected[i].rect.height / 2.0);
		Point center = Point(cx, cy);
		treesDetected[i].numOverlRect = 0;
		for (int j = 0; j < treesDetected.size(); j++) {
			//compare if the center of the current rectangle i is contained in the rectangle 
			if (treesDetected[j].rect.contains(center) && i != j) {
				treesDetected[i].numOverlRect++;
			}
		}

		//one obtained the number of overrlapping rectangles for each rectangle - detect which method is suitted to post-elaborate each series of rectangles - FINITE STATE MACHINE
		double ratio = treesDetected[i].score / treesDetected[i].zncc; //metric used to select teh best tree in the fist method
		if (treesDetected[i].numOverlRect == 15) { //second method
			final_detected_trees.push_back(treesDetected[i]);
			if (status != S_SND_MTD) status = S_SND_MTD;
		}
		else if (treesDetected[i].numOverlRect >= 8 && ratio < 35000 && status != S_SND_MTD) { //fisrt method
			final_detected_trees.push_back(treesDetected[i]);
			if (status != S_FST_MTD) status = S_FST_MTD;
		}

		/*DEBUG CODE - visulize for each candidate image their relative statistics and assign them a quality number for evaluation purposes */
		/*Mat t;
		inputImg.copyTo(t);
		circle(t, center, 1, Scalar(50), 2);
		rectangle(t, treesDetected[i].rect, Scalar(125), 2);
		namedWindow("Final detection", WINDOW_NORMAL);
		imshow("Final detection", t);
		int k = waitKey(0);
		//treesDetected[i].qlt = (k - 48); // '0' = 48 number associated with 0 character
		printTreeData(treesDetected[i]);*/


	}
	//waitKey(0);
	//cout << endl;

	return final_detected_trees; //empty if status == S_UNKNOWN

}//END - searchTemplateWithCanny

/*
 * function that perform template matching at a given scale with a given template image and extract some statistics from the best result
 * @param img, input image from which we want to detect a tree
 * @param tImg, template image used in a sliding windows approach to find a tree in the orevious image
 * @param scale, scaling factor at which the template matching approach need to be done
 * @param data, data structure filled by the funciton with several information and statistics about the best match
 */
void TreeUtil::slidingWindow(Mat img, Mat tImg, double scale, TreeUtil::treeData &data) {
	//cout << "Original size " << img.size() << " - scaling factor " << scale << endl;
	vector<int> methods = { TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR,	TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED }; //list of different method suitted for template matching
	vector<string> methodsNames = { "TM_CCOEFF","TM_CCOEFF_NORMED", "TM_CCORR",	"TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED" };
	double maxVal, minVal;
	Point topLeft, bottomRight, minLoc, maxLoc;
	Mat imgInput, cImg;
	resize(img, img, Size(img.cols / scale, img.rows / scale)); //resize according to the scale factor
	Canny(img, cImg, 500, 600); //compute canny image of the input one
	///visulalize canny image of the target image
	//namedWindow("canny img", WINDOW_NORMAL);
	//imshow("canny img", cImg);
	//waitKey(1);

	int win_rows = WIN_ROWS, win_cols = WIN_COLS, stepSize = 30, match;
	int i = 2; //method -> TM_CCORR

	//template matching
	Mat res;
	matchTemplate(cImg, tImg, res, methods[i]);
	minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

	data.scale = scale;
	//analyze results
	if (methods[i] == TM_SQDIFF || methods[i] == TM_SQDIFF_NORMED) {
		topLeft = minLoc;
		//cout << "Method " << methodsNames[i] << " -> score: " << minVal ;
		data.score = minVal * (scale / 2); //the score of the template mathing is multiplied by the scale in order to be rubust against bigger trees
		//data.score = minVal;
	}
	else {
		topLeft = maxLoc;
		//cout << "Method " << methodsNames[i] << " -> score: " << maxVal ;
		//data.score = maxVal * scale;
		data.score = maxVal * (scale / 2); //the score of the template mathing is multiplied by the scale in order to be rubust against bigger trees
	}
	//compute the buonding rectangle at the given scale w.r.t. the selected tree
	bottomRight = Point2f(topLeft.x + tImg.cols, topLeft.y + tImg.rows);
	data.rect = Rect(topLeft, bottomRight);
	//compute zncc metric - comparing the just detected tree with the template image
	data.zncc = zncc(cImg(data.rect), tImg);


	/* DEBUG CODE - draw rectangle in the output image + quality evaluation */
	//cout << " ZNCC: " << data.zncc << endl;
	//rectangle(cImg, topLeft, bottomRight, Scalar(255));

	//namedWindow("Best match at scale " + to_string(scale), WINDOW_NORMAL);
	//imshow("Best match at scale " + to_string(scale), cImg);
	//waitKey(0);
	/*int k = waitKey();
	data.qlt = (k - 48); // '0' = 48 number associated with 0 character
	cout << "Key pressed: " << data.qlt << endl;*/

}

/*
 * function that computea the Zero Mean Cross Correlation metric given two images of the same size
 * @param inputRect, binary image from which we want to calculate the ZNCC metric
 * @param templ, binary image of the template from which we want to calculate the ZNCC metric
 */
double TreeUtil::zncc(Mat inputRect, Mat templ) {
	double zncc;
	Scalar meanImg, stdDevImg, meanT, stdDevT;
	meanStdDev(inputRect, meanImg, stdDevImg);
	meanStdDev(templ, meanT, stdDevT);
	for (int i = 0; i < inputRect.rows; i++) {
		for (int j = 0; j < inputRect.cols; j++) {
			zncc += ((inputRect.at<uchar>(i, j) - meanImg[0]) * (templ.at<uchar>(i, j) - meanT[0]));
		}
	}
	zncc = zncc / (stdDevImg[0] * stdDevT[0]);
	return abs(zncc);

}

/*
 * function that takes a vector of preselected trees and find the best matching one according to two algorithm
 * @param trees, vector of trees from which we need to extract the best performing one
 * @param inputImg, original image from which we want to detect a tree
 * @param selectedTree, dataStructure filled by the function with relevant infomation about the best matching tree
 * @return treu if a tree is selected as the candidate one, false otherwise. In this last case the selectedTree variable doesn't contains a valid result
 */
bool TreeUtil::refineResults(vector<TreeUtil::treeData> trees, Mat inputImg, TreeUtil::treeData &selectedTree) {
	if (status == S_SND_MTD) { //second algorithm evaluation
		//select the tree with the lower ratio and a zncc < 500
		double lowestRatio = DBL_MAX;
		for (TreeUtil::treeData tree : trees) {
			double ratio = tree.score / tree.zncc;
			if (ratio < lowestRatio && tree.zncc < 500) {
				lowestRatio = ratio;
				selectedTree = tree;
			}
		}
	}// if - second method
	else if (status == S_FST_MTD) { //first algorithm evaluation
		//select the tree with higest score
		double higestScore = 0;
		for (TreeUtil::treeData tree : trees) {
			if (tree.score > higestScore) {
				higestScore = tree.score;
				selectedTree = tree;
			}
		}
	}// - frist method

	 ///visualize results
	if (status == S_UNKNOWN) return false;

	//check with SSIM similarity metric whether the selected tree is a tree or not
	return isATree(inputImg(selectedTree.rect), "../data/template");

}
/*
 * function that perfrom a further refinement by computing the average MSSIM metric in order to say wheter the selected rectangle is actually a tree or not
 * @param rectangle, selected rectangle from which we want to determine if it is a tree or not
 * @param templateImages, path to the template image used to calculate the MSSIM metric
 * return treu if the rectangle passed as argument is actully a tree, false otherwise.
 */
bool TreeUtil::isATree(Mat rectangle, string templateImages) {
	vector<String> files;
	utils::fs::glob(templateImages, "*.*", files);
	double avgMSSIM = 0.0;
	cvtColor(rectangle, rectangle, COLOR_BGR2GRAY);
	for (int i = 0; i < files.size(); i++) {
		Mat templImg = imread(files[i]);
		resize(templImg, templImg, rectangle.size()); //resize template image to the size of the detected rectangle
		cvtColor(templImg, templImg, COLOR_BGR2GRAY);
		Scalar res = getMSSIM(templImg, rectangle);
		//avgSSIM += mean(res)[0]; //bgr image
		avgMSSIM += res[0]; //for gray scale image
		//cout << "template file: " << files[i] << " score: " << res[0] << endl;
	}
	avgMSSIM /= files.size();
	//cout << "score: " << avgMSSIM << endl;
	if (avgMSSIM > THRESHOLD_MSSIM) return false;
	return true;
}

/*
 * function that print statistics about a tree stored in an appropriate data sturcture
 * @param data, instance of the data structure that need to be printed
 */
void TreeUtil::printTreeData(TreeUtil::treeData data) {
	cout << "Tree - " << data.fileName << " scale: " << data.scale << " score: " << data.score << " ZNCC: " << data.zncc << " ratio: " << (data.score / data.zncc) << /*" quality: " << data.qlt <<*/ endl;
	cout << "Rect: " << data.rect << " Overlapping rectangle: " << data.numOverlRect << endl;
}
/*
 * function that compute the MSSIM given two images of the same size - source: https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
 * @param i1, first image to be compared
 * @param i2, second image to becompared
 * return a scalar containing the MSSIM metric in the first posizion
 */
Scalar TreeUtil::getMSSIM(const Mat& i1, const Mat& i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;
	//int d = COLOR_BGR2GRAY;

	Mat I1, I2;
	i1.convertTo(I1, d);           // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	Mat I2_2 = I2.mul(I2);        // I2^2
	Mat I1_2 = I1.mul(I1);        // I1^2
	Mat I1_I2 = I1.mul(I2);        // I1 * I2

	/*************************** END INITS **********************************/

	Mat mu1, mu2;   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	Scalar mssim = mean(ssim_map); // mssim = average of ssim map
	return mssim;
}