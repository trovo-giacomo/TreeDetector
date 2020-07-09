#include "tree_detection_util.h"
#include <iostream>
/*
 * Author: Giacomo Trovo'
 * Computer Vision course - 9CFU - UNIPD
 * 1st session - 2nd appeal - 9th July 2020
 * main.cpp - executable program - its goal is to detect with a bounding box trees from the input image passed as an argument
 */



using namespace std;
using namespace cv;

/*
 * Goal: develop a system that is capable of automatically detecting trees in an image by creating a bounding box around each one.
 * In order to be recognized as a tree, it should be clearly visible and evident � small trees in the background, grass etc. do not need to be detected
 */


/******** Code to get time in milli seconds both from Linux system and from Windows one ********/
#if !defined(_WIN32) && !defined(_WIN64) // Linux - Unix
#  include <sys/time.h>
typedef timeval sys_time_t;
inline void system_time(sys_time_t* t) {
	gettimeofday(t, NULL);
}
inline long long time_to_msec(const sys_time_t& t) {
	return t.tv_sec * 1000LL + t.tv_usec / 1000;
}
#else // Windows and MinGW
#  include <sys/timeb.h>
typedef _timeb sys_time_t;
inline void system_time(sys_time_t* t) { _ftime(t); }
inline long long time_to_msec(const sys_time_t& t) {
	return t.time * 1000LL + t.millitm;
}
#endif


//Main function to detect trees in the input image passed as an argument
int main(int argc, char* argv[]) {
	//take input image from command line
	if (argc != 2) {
		cout << "Provide [image_path] as argument in command line" << endl;
		cout << "The program stops!" << endl;
		return -1;
	}
	Mat inputImg;
	//read input image and display it
	inputImg = imread(argv[1]);
	namedWindow("Input image", WINDOW_NORMAL);
	imshow("Input image", inputImg);
	cout << "Press a key to start processing the image..." << endl;
	waitKey();
	//time measurement - start
	sys_time_t t;
	system_time(&t);
	long startTime = time_to_msec(t);

	//resizing input image to make it suitable to template matching and uniform all images to the same size
	resize(inputImg, inputImg, Size(1200, 600));


	///Template matching with Canny edge detector
	/* INITALIAZTION PHASE */
	//find canny parameter for the template images
	
	//findCanny(inputImg, "../data/template3/");

	/* TEMPLATE MATCHING WITH CANNY EDGE DETECTOR */
	vector<TreeUtil::treeData> selectTrees = TreeUtil::searchTemplateWithCanny(inputImg, "../data/cannyTemplate/");

	/* SELECTION OF THE FINAL CANDIDATE */
	TreeUtil::treeData final_tree;
	bool hasATree = TreeUtil::refineResults(selectTrees, inputImg, final_tree);

	//time measurement - end
	system_time(&t);
	long endTime = time_to_msec(t);

	/* VISUALIZATION OF RESULTS */
	//visualze the final detected tree
	if (!hasATree) {
		cout << "NO TREES DETECTED!" << endl;

	}
	else {
		cout << "Final detected tree - method: " << TreeUtil::status << endl;
		cout << "Final tree rect: " <<  final_tree.rect << endl;
		cout << "Time expired: " << (endTime - startTime) << " ms" << endl;
		TreeUtil::printTreeData(final_tree);

		rectangle(inputImg, final_tree.rect, Scalar(125), 2);
		namedWindow("Final detection", WINDOW_NORMAL);
		imshow("Final detection", inputImg);
		cout << "Time expired: " << (endTime - startTime) << " ms" << endl;
		cout << "Press a key to terminate..." << endl;
		waitKey(0);
	}
	

	destroyAllWindows();
	
	return 0;
}

