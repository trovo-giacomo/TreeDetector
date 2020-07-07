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
 * tree_detection_util.h - header file of the homonimous .cpp file
 */




class TreeUtil {
public:
	static int status;

	//data structure to mantain information about the candidates rectangles that can be trees
	static struct treeData {
		std::string fileName = "";
		double scale, score, zncc;
		// int qlt; //quality of the matching [0,4] -> used during threshold evaluation for quality measurement
		cv::Rect rect;
		int numOverlRect;
	};
	static std::vector<treeData> searchTemplateWithCanny(cv::Mat inputImg, std::string pathTemplate);
	static void slidingWindow(cv::Mat img, cv::Mat tImg, double scale, treeData &data);
	static void printTreeData(treeData data);
	static void findCanny(cv::Mat inputImg, std::string templImg);
	static void computeCanny(int, void* data);
	static double zncc(cv::Mat inputRect, cv::Mat templ);
	static bool refineResults(std::vector< treeData> selectTrees, cv::Mat inputImg, treeData &selectedTree);
	static bool isATree(cv::Mat rectangle, std::string templateImages);
	static cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2);
};