#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
/*
 * Author: Giacomo Trovò
 * Computer Vision course - 9CFU - UNIPD
 * 1st session - 2nd appeal - 9th July 2020
 */

//template matching - windows size
#define WIN_ROWS 60
#define WIN_COLS 60

// Definition of status code for the finite state machine for the post-processing template matching results 
#define S_UNKNOWN 0 
#define S_FST_MTD 1
#define S_SND_MTD 2

//Threshold on MSSIM score
#define THRESHOLD_MSSIM 0.194

using namespace std;
using namespace cv;

/*
 * Goal: develop a system that is capable of automatically detecting trees in an image by creating a bounding box around each one.
 * In order to be recognized as a tree, it should be clearly visible and evident � small trees in the background, grass etc. do not need to be detected
 */


//data structure to mantain information about the candidates rectangles that can be trees
struct treeData {
	string fileName = "";
	double scale, score, zncc;
	// int qlt; //quality of the matching [0,4] -> used during threshold evaluation for quality measurement
	Rect rect;
	int numOverlRect;
	double scoreOverlRect; // !! MAY BE USELESS !!
};

//status varaible for the finite state machine set up for the evaluation of the best candidate among the selected ones - more detail afterwards
int status = S_UNKNOWN;
//temp varaible
string inputImageName;
Mat tImg;

vector<treeData> searchTemplateWithCanny(Mat inputImg, string pathTemplate);

void slidingWindow(Mat img, Mat tImg, double scale, struct treeData &data );
void printTreeData(struct treeData data);
void findCanny(Mat inputImg, string templImg);
void computeCanny(int, void* data);
double zncc(Mat inputRect, Mat templ);
void visualizeResults(vector<treeData> selectTrees, Mat inputImg);
bool isATree(Mat rectangle, string templateImages);
Scalar getMSSIM(const Mat& i1, const Mat& i2);



int main(int argc, char* argv[]) {
	//take input image from command line
	if (argc != 2) {
		cout << "Provide [image_path] as an argument in command line" << endl;
		cout << "The program stop!" << endl;
		return -1;
	}
	Mat inputImg;
	//read input image and display it
	inputImageName = argv[1];
	inputImg = imread(argv[1]);
	namedWindow("Input image", WINDOW_NORMAL);
	imshow("Input image", inputImg);
	waitKey();
	//resizing input image to make it suitable to template matching and uniform all images to the same size
	resize(inputImg, inputImg, Size(1200, 600));


	///Template matching with Canny edge detector
	/* INITALIAZTION PHASE */
	//find canny parameter for the template images
	
	//findCanny(inputImg, "../data/template3/");

	/* TEMPLATE MATCHING WITH CANNY EDGE DETECTOR */
	vector<treeData> selectTrees = searchTemplateWithCanny(inputImg, "../data/cannyTemplate/");

	/* SELECTION OF THE FINAL CANDIDATE */
	visualizeResults(selectTrees, inputImg);

	/* VISUALIZATION OF RESULTS */


	destroyAllWindows();
	
	return 0;
}
/******* FUNCTIONS FOR INITIALIZATION ***********/

struct DataCanny {
	Mat src, dstBlur, dstCanny;
	int sigma, th, tl;
	string fileName;
};

void findCanny(Mat inputImg, string pathTemplate) {
	vector<String> files;
	utils::fs::glob(pathTemplate, "*.*", files);
	for (int i = 5; i < files.size(); i++) {
		DataCanny *d = new DataCanny();
		namedWindow(files[i], WINDOW_NORMAL);
		d->src = imread(files[i]);
		resize(d->src, d->src, Size(WIN_COLS, WIN_ROWS));
		d->fileName = files[i];
		imshow(files[i], d->src);
		createTrackbar("Blur[*10]: ", files[i], &(d->sigma), 1000, computeCanny, d);
		createTrackbar("Th: ", files[i], &(d->th), 1000, computeCanny, d);
		createTrackbar("Tl: ", files[i], &(d->tl), 1000, computeCanny, d);
		waitKey();

		imwrite("../data/cannyTemplate/t"+to_string(i)+".jpg", d->dstCanny);
	}
}

void computeCanny(int , void* data) {
	DataCanny *d = (DataCanny *)data;
	double sig = d->sigma / 10;
	GaussianBlur(d->src, d->dstBlur, Size(5, 5), sig, sig);
	Canny(d->dstBlur, d->dstCanny, d->th, d->tl);
	namedWindow(d->fileName, WINDOW_NORMAL);
	imshow(d->fileName, d->dstCanny);
}

/******* END FUNCTIONS FOR INITIALIZATION ***********/

vector<treeData> searchTemplateWithCanny(Mat inputImg, string pathTemplate){

	// import all template images (already processed by the initialization step with canny edge detector)
	vector<String> files;
	utils::fs::glob(pathTemplate, "*.*", files);
	vector<double> scales = { 1.5, 2.0, 2.5, 3, 4, 5, 6, 10};
	int numMatches;
	
	vector<struct treeData> treesDetected; //vector that will contain the result of template matching, all trees among all the template images

	for (int j = 0; j < files.size(); j++) {

		Mat tImg = imread(files[j], IMREAD_GRAYSCALE);
		if(tImg.size() != Size(WIN_COLS, WIN_ROWS))	resize(tImg, tImg, Size(WIN_COLS, WIN_ROWS));
		
		//for each template image -> perform template matching at several scales
		vector<struct treeData> treeDataScales; //vector that will contain the best results for each scale
		double ratioSel = 0;

		for (int i = 0; i < scales.size(); i++) {
			//cout << "Sliding win file " << files[j] << " - scale: " << scales[i] << endl;

			struct treeData data;	
			data.fileName = files[j];
			slidingWindow(inputImg, tImg, scales[i], data); //perform template matching at the given scale - fill data structure with useful information and statistics
			
			double sc = data.scale;
			Rect treeWindow(data.rect.x*sc, data.rect.y*sc, data.rect.width*sc, data.rect.height*sc); // calculate the effective rectangle 
			data.rect = treeWindow;
			treeDataScales.push_back(data);
		}

		treeData selectTree;
		//Select a rectangle (tree) according to the highest score (template matching one) within the best candidate trees at all scales
		selectTree.score = 0;
		for (treeData td : treeDataScales) {
			if (td.score > selectTree.score) selectTree = td;
		}
		treesDetected.push_back(selectTree);

	
		//cout << "ended" << endl << endl;
		//waitKey(1);
	}
	/* DEBUG MESSAGES */
	//cout << "--- Over all trees selected --- " <<  inputImageName << endl;
	//cout << "Number of tree detected: " << treesDetected.size() << endl;


	//count for each rectangle, the number of overlapping rectangle with reference to the central point
	vector<treeData> fst_mtd_trees, snd_mtd_trees;
	for (int i = 0; i < treesDetected.size(); i++) {
		int cx = treesDetected[i].rect.x + (treesDetected[i].rect.width / 2.0);
		int cy = treesDetected[i].rect.y + (treesDetected[i].rect.height / 2.0);
		Point center = Point(cx, cy);
		treesDetected[i].numOverlRect = 0;
		treesDetected[i].scoreOverlRect = 0.0;
		for (int j = 0; j < treesDetected.size(); j++) {
			//compare if the center of the current rectangle i is contained in the rectangle 
			if (treesDetected[j].rect.contains(center) && i != j) {
				treesDetected[i].numOverlRect++;
				treesDetected[i].scoreOverlRect += (1 / (2 * treesDetected[j].scale));
			}
		}
		
		//one obtained the number of overrlapping rectangles for each rectangle - detect which method is suitted to post-elaborate each series of rectangles - FINITE STATE MACHINE
		double ratio = treesDetected[i].score / treesDetected[i].zncc; //metric used to select teh best tree in the fist method
		if (treesDetected[i].numOverlRect == 15) { //second method
			snd_mtd_trees.push_back(treesDetected[i]);
			if(status != S_SND_MTD)status = S_SND_MTD;
		}
		else if (treesDetected[i].numOverlRect >= 8 && ratio < 35000 && status != S_SND_MTD) { //fisrt method
			fst_mtd_trees.push_back(treesDetected[i]);
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
		treesDetected[i].qlt = (k - 48); // '0' = 48 number associated with 0 character
		printTreeData(treesDetected[i]);*/
		
		
	}
	//waitKey(0);
	//cout << endl;

	vector<treeData> emptyList;
	if (status == S_SND_MTD) return snd_mtd_trees;
	else if (status == S_FST_MTD) return fst_mtd_trees;
	return emptyList; //status S_UNKWON -> no trees detected
}//END - searchTemplateWithCanny

void slidingWindow(Mat img, Mat tImg, double scale, struct treeData &data ) {
	//cout << "Original size " << img.size() << " - scaling factor " << scale << endl;
	vector<int> methods = { TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR,	TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED }; //list of different method suitted for template matching
	vector<string> methodsNames = { "TM_CCOEFF","TM_CCOEFF_NORMED", "TM_CCORR",	"TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED" };
	double maxVal, minVal;
	Point topLeft, bottomRight, minLoc, maxLoc;
	Mat imgInput, cImg;
	resize(img, img, Size(img.cols / scale, img.rows / scale));
	Canny(img, cImg, 500, 600);
	///visulalize canny image of the target image
	//namedWindow("canny img", WINDOW_NORMAL);
	//imshow("canny img", cImg);
	
	//cout << "Scaled size " << img.size() << endl;
	int win_rows = WIN_ROWS, win_cols = WIN_COLS, stepSize = 30, match;
	int i = 2; //method -> TM_CCOEFF
	
	//template matching
	Mat res;
	matchTemplate(cImg, tImg, res, methods[i]);
	minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

	data.scale = scale;
	//analyze results
	if(methods[i]== TM_SQDIFF || methods[i] == TM_SQDIFF_NORMED){
		topLeft = minLoc;
		//cout << "Method " << methodsNames[i] << " -> score: " << minVal ;
		data.score = minVal*(scale/2); //the score of the template mathing is multiplied by the scale in order to be rubust against bigger trees
		//data.score = minVal;
	}
	else {
		topLeft = maxLoc;
		//cout << "Method " << methodsNames[i] << " -> score: " << maxVal ;
		//data.score = maxVal * scale;
		data.score = maxVal * (scale / 2); //the score of the template mathing is multiplied by the scale in order to be rubust against bigger trees
	}
	//compute the buonding rectangle at the given scale w.r.t. the selected tree
	bottomRight = Point2f(topLeft.x+tImg.cols,topLeft.y+tImg.rows);
	data.rect = Rect(topLeft, bottomRight);
	//compute zncc metric - comparing the just detected tree with the template image
	data.zncc = zncc(cImg(data.rect), tImg);
	
	
	/* DEBUG CODE - draw rectangle in the output image + quality evaluation */
	//cout << " ZNCC: " << data.zncc << endl;
	//rectangle(imgInput, topLeft, bottomRight, Scalar(255));
	
	//namedWindow("Img at method " + methodsNames[i], WINDOW_NORMAL);
	//imshow("Img at method " + methodsNames[i], cImg);
	//waitKey(1);
	/*int k = waitKey();
	data.qlt = (k - 48); // '0' = 48 number associated with 0 character
	cout << "Key pressed: " << data.qlt << endl;*/

}

double zncc(Mat inputRect, Mat templ) {
	double zncc;
	Scalar meanImg, stdDevImg, meanT, stdDevT;
	meanStdDev(inputRect, meanImg, stdDevImg);
	meanStdDev(templ, meanT, stdDevT);
	for (int i = 0; i < inputRect.rows; i++) {
		for (int j = 0; j < inputRect.cols; j++) {
			zncc += ( (inputRect.at<uchar>(i, j) - meanImg[0]) * (templ.at<uchar>(i, j) - meanT[0]) );
		}
	}
	zncc = zncc / (stdDevImg[0] * stdDevT[0]);
	return abs(zncc);

}

//takes a vector preselected trees and find the best matching one according to two algorithm
void visualizeResults(vector<treeData> trees, Mat inputImg) {
	treeData selectedTree;
	if (status == S_SND_MTD) { //second algorithm evaluation
		//select the tree with the lower ratio and a zncc < 500
		double lowestRatio = DBL_MAX;
		for (treeData tree : trees) {
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
		for (treeData tree : trees) {
			if (tree.score > higestScore) {
				higestScore = tree.score;
				selectedTree = tree;
			}
		}
	}// - frist method
	
	 ///visualize results
	if (status == S_UNKNOWN) {
		cout << "NO TREES DETECTED!" << endl;
		return;
	}
	//check with SSIM similarity metric whether the selected tree is a tree or not
	if (isATree(inputImg(selectedTree.rect), "../data/template")) {
		//visualze the final detected tree
		cout << "Final detected tree - method: " << status << endl;
		printTreeData(selectedTree);

		rectangle(inputImg, selectedTree.rect, Scalar(125), 2);
		namedWindow("Final detection", WINDOW_NORMAL);
		imshow("Final detection", inputImg);
		waitKey(0);
	}
	else {
		cout << "NO TREES DETECTED!" << endl;
	}

}

bool isATree(Mat rectangle, string templateImages) {
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
		//cout << "template file: " << files[i] << " score: " << res << endl;
	}
	avgMSSIM /= files.size();
	//cout << " score: " << avgMSSIM << endl;
	if (avgMSSIM > THRESHOLD_MSSIM) return false;
	return true;
}

void printTreeData(treeData data){
	cout << "Tree - " << data.fileName << " scale: " << data.scale << " score: " << data.score << " ZNCC: " << data.zncc << " ratio: " << (data.score/data.zncc) << /*" quality: " << data.qlt <<*/ endl;
	cout << "Rect: " << data.rect << " Overlapping rectangle: " << data.numOverlRect << " Score overlapping rect: " << data.scoreOverlRect << " score2: " << (data.scoreOverlRect/ data.numOverlRect) <<  endl;
}
/*
https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
*/
Scalar getMSSIM(const Mat& i1, const Mat& i2)
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