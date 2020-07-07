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

#define WIN_ROWS 60
#define WIN_COLS 60
/* Definition of status code for the finite state machine for the post-processing template matching results */
#define S_UNKNOWN 0 
#define S_FST_MTD 1
#define S_SND_MTD 2

#define THRESHOLD_MSSIM 0.194

using namespace std;
using namespace cv;

/*
 * Goal: develop a system that is capable of automatically detecting trees in an image by creating a bounding box around each one.
 * In order to be recognized as a tree, it should be clearly visible and evident � small trees in the background, grass etc. do not need to be detected
 */
struct treeData {
  string fileName = "";
  double scale, score, zncc;
  int qlt; //quality of the matching [0,4]
  Rect rect;
  int numOverlRect;
  double scoreOverlRect;
};
int status = S_UNKNOWN;

void equalize(Mat inputImg, Mat &inputEqualized);
void watershedSegmentation(Mat img, Mat &segmentedImg);
void Erosion(int, void*);
void templateSegmentation(Mat img, string pathTemplate);

vector<treeData> searchTemplateCanny(Mat inputImg, string pathTemplate);
void extractFeatures(Mat img, vector<KeyPoint> &features, Mat &desciptors, int numFeatures);
void computeMatches(Mat templateDescriptors, Mat imageDescriptors, vector<DMatch> &matches, float ratio);
void computeMatchesFlann(Mat templateDescriptors, Mat imageDescriptors, vector<DMatch> &matches, float ratio);

int printRectangle(Mat &frame, vector<Point2f> corners, Scalar color);
int slidingWindow(Mat img, double scale, struct treeData &data);
void printTreeData(struct treeData data);
void findCanny(Mat inputImg, string templImg);
void computeCanny(int, void* data);
vector<treeData> refineWithFeatureMatching(Mat inputImage, string templateFeaturePath, vector<treeData> selecTrees);
double zncc(Mat inputRect, Mat templ);
double avgZncc(Mat inputRect, string templPath);
void visualizeResults(vector<treeData> selectTrees, Mat inputImg);
bool isATree(Mat rectangle, string templateImages);
Scalar getMSSIM(const Mat& i1, const Mat& i2);


//double computeZNCC(Mat img, Mat templImg);

Mat src, erosion_dst, dilation_dst;
int erosion_elem = 0;
int erosion_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

string inputImageName;
Mat tImg;



int main(int argc, char* argv[]) {
  //take input image from command line
  if (argc != 2) {
    cout << "Provide [image_path] as an argument in command line" << endl;
    cout << "The program stop!" << endl;
    return -1;
  }
  Mat inputImg, inputEqualized, segmentedImg;
  
  inputImageName = argv[1];
  inputImg = imread(argv[1]);
  namedWindow("Input image", WINDOW_NORMAL);
  imshow("Input image", inputImg);
  waitKey();

  resize(inputImg, inputImg, Size(1200, 600));


  //equalization of the input image in HSV color code
  /*equalize(inputImg, inputEqualized);
  namedWindow("Equalized image", WINDOW_NORMAL);
  imshow("Equalized image", inputEqualized);
  waitKey(0);*/

  ///watershedSegmentation
  //watershedSegmentation(inputImg, segmentedImg);

  ///template matching with generalized hough trasform
  //templateSegmentation(inputImg, "../data/template/");

  ///Matching features by template Canny
  //find canny parameter
  //findCanny(inputImg, "../data/template3/");
  vector<treeData> selectTrees = searchTemplateCanny(inputImg, "../data/cannyTemplate/");
  visualizeResults(selectTrees, inputImg);
  //selectTrees = refineWithFeatureMatching(inputImg, "../data/template3/", selectTrees);
  //extractFeatures();

  destroyAllWindows();
  
  return 0;
}

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

vector<treeData> searchTemplateCanny(Mat inputImg, string pathTemplate){
  vector<String> files;

  utils::fs::glob(pathTemplate, "*.*", files);
  vector<double> scales = { 1.5, 2.0, 2.5, 3, 4, 5, 6, 10};
  int numMatches;
  
  //extract from input image
    vector<struct treeData> treesDetected;
  for (int j = 0; j < files.size(); j++) {
    //for (int j = 0; j < 2; j++) {

    tImg = imread(files[j], IMREAD_GRAYSCALE);
    if(tImg.size() != Size(WIN_COLS, WIN_ROWS)) resize(tImg, tImg, Size(WIN_COLS, WIN_ROWS));
    //Canny(templImg, tImg, 500, 600);

    /*namedWindow("canny templ", WINDOW_NORMAL);
    imshow("canny templ", tImg);
    waitKey(1);*/
    //tImg = templImg;
    /*cv::Scalar meanT, stddevT;
    cv::meanStdDev(templImg, meanT, stddevT);*/
    vector<KeyPoint> templFeatures;
    Mat templDescr;
    //extractFeatures(templImg, templFeatures, templDescr, 2000);
    vector<struct treeData> treeDataScales;
    double ratioSel = 0;

    for (int i = 0; i < scales.size(); i++) {
      //cout << "Sliding win file " << files[j] << " - scale: " << scales[i] << endl;
      Mat temp;
      inputImg.copyTo(temp);
      struct treeData data; 
      data.fileName = files[j];
      numMatches = slidingWindow(temp, scales[i], data);
      
      double sc = data.scale;
      Rect treeWindow(data.rect.x*sc, data.rect.y*sc, data.rect.width*sc, data.rect.height*sc);
      data.rect = treeWindow;
      //add every rectangle to an vector in order to count how many overlapping rectangle there are for each one
      treeDataScales.push_back(data);
      //treesDetected.push_back(data);
    }
    //count how many rectangle are ovelapping for the the same template image in different scales
    /*Mat t;
    inputImg.copyTo(t);
    for (int i = 0; i < treeDataScales.size(); i++) {
      int cx = treeDataScales[i].rect.x + (treeDataScales[i].rect.width / 2.0);
      int cy = treeDataScales[i].rect.y + (treeDataScales[i].rect.height / 2.0);
      Point center = Point(cx, cy);
      cout << "x: " << treeDataScales[i].rect.x << " y: " << treeDataScales[i].rect.y << " w: " << treeDataScales[i].rect.width << " h: " << treeDataScales[i].rect.height << endl;
      treeDataScales[i].numOverlRect = 0;
      treeDataScales[i].scoreOverlRect = 0.0;
      for (int j = 0; j < treeDataScales.size(); j++) {
        //compare if the center of the current rectangle i is contained in the rectangle 
        if ((treeDataScales[j].rect.contains(center)) && (i != j)) {
          treeDataScales[i].numOverlRect++;
          treeDataScales[i].scoreOverlRect += (1 / treeDataScales[j].scale);
        }
      }
      //treesDetected[i].scoreOverlRect /= treesDetected[i].scale;
      //if(treesDetected[i].score[i] )
      
      circle(t, center, 3, Scalar(50),2);
      cout << "Center: " << center << endl;
      rectangle(t, treeDataScales[i].rect, Scalar(125), 2);
      namedWindow("Final detection", WINDOW_NORMAL);
      imshow("Final detection", t);
      int k = waitKey(0);
      treeDataScales[i].qlt = (k - 48); // '0' = 48 number associated with 0 character
      printTreeData(treeDataScales[i]);

    }*/

    treeData selectTree;
    ///Select trees according to the highest score
    selectTree.score = 0;
    for (treeData td : treeDataScales) {
      if (td.score > selectTree.score) selectTree = td;
    }
    //treesDetected.push_back(selectTree);
    ///Select trees according to the lower ZNCC
    /*selectTree.zncc = DBL_MAX;
    for (treeData td : treeDataScales) {
      if (td.zncc < selectTree.zncc) selectTree = td;
    }*/
    ///Select according to the highest ratio - not working at all! -> select only scale 10 rectangles
    /*double ratio = DBL_MAX;
    for (treeData td : treeDataScales) {
      double r = td.score / td.zncc;
      if (r < ratio) selectTree = td;
    }*/
    treesDetected.push_back(selectTree);

    //Select trees according to the ratio score/zncc
    /*for (treeData td : treeDataScales) {
      //if ( ((td.score > 7e+6) && (td.score / td.zncc) > 400000 ) && ((td.score / td.zncc) <  1.40e+6) ) selectTree = td;
      if (((td.score > 7e+6))) treesDetected.push_back(td);// selectTree = td;
    }*/
    //if(selectTree.fileName != "") treesDetected.push_back(selectTree);

    //if ((selectTree.score > 7e+6) && (selectTree.zncc < 500)) treesDetected.push_back(selectTree);
  
    //cout << "ended" << endl << endl;
    //waitKey(1);
  }
  //cout << "--- Over all trees selected --- " <<  inputImageName << endl;
  //cout << "Number of tree detected: " << treesDetected.size() << endl;
  Size originalSize = inputImg.size();
  /*for (int j = 0; j < 5; j++) {
    cout << "Quality: " << j << endl;
    for (int i = 0; i < treesDetected.size(); i++) {
      if (treesDetected[i].qlt != j) continue;
      Mat t;
      inputImg.copyTo(t);

      double sc = treesDetected[i].scale;
      Rect treeWindow(treesDetected[i].rect.x*sc, treesDetected[i].rect.y*sc, treesDetected[i].rect.width*sc, treesDetected[i].rect.height*sc);
      rectangle(t, treeWindow, Scalar(125), 2);

      printTreeData(treesDetected[i]);
      cout << "Rect: " << treeWindow << endl;
      namedWindow("Final detection", WINDOW_NORMAL);
      imshow("Final detection", t);
      waitKey(0);

    }
    cout << endl;
  }*/
  //visulalize better rectangle and assign each a score
  /*for (int i = 0; i < treesDetected.size(); i++) {
    Mat t;
    inputImg.copyTo(t);

    double sc = treesDetected[i].scale;
    Rect treeWindow(treesDetected[i].rect.x*sc, treesDetected[i].rect.y*sc, treesDetected[i].rect.width*sc, treesDetected[i].rect.height*sc);
    rectangle(t, treeWindow, Scalar(125), 2);
    namedWindow("Final detection", WINDOW_NORMAL);
    imshow("Final detection", t);
    int k = waitKey(0);
    treesDetected[i].qlt = (k - 48); // '0' = 48 number associated with 0 character
    //cout << "Key pressed: " << treesDetected[i].qlt << endl;
    treesDetected[i].rect = treeWindow;
    printTreeData(treesDetected[i]);
    cout << "Rect: " << treeWindow << endl;


  }*/

  //count for each rectangle number of overlapping rectangle comparing its central points
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
    //treesDetected[i].scoreOverlRect /= treesDetected[i].scale;
    //if (treesDetected[i].numOverlRect > 85 || treesDetected[i].scoreOverlRect>30 || (/*treesDetected[i].score > 9.9e+6 &&*/ treesDetected[i].zncc < 320)) {
    /*if (treesDetected[i].numOverlRect > 12) { //all rectangle are quite in the same position
      //search the 
    }*/
    //detect which method is suitted to post-elaborate each series of rectangles
    double ratio = treesDetected[i].score / treesDetected[i].zncc;
    if (treesDetected[i].numOverlRect == 15) {
      snd_mtd_trees.push_back(treesDetected[i]);
      if(status != S_SND_MTD)status = S_SND_MTD;
    }
    else if (treesDetected[i].numOverlRect >= 8 && ratio < 35000 && status != S_SND_MTD) {
      fst_mtd_trees.push_back(treesDetected[i]);
      if (status != S_FST_MTD) status = S_FST_MTD;
    }

      /*Mat t;
      inputImg.copyTo(t);
      circle(t, center, 1, Scalar(50), 2);
      rectangle(t, treesDetected[i].rect, Scalar(125), 2);
      namedWindow("Final detection", WINDOW_NORMAL);
      imshow("Final detection", t);
      int k = waitKey(0);
      treesDetected[i].qlt = (k - 48); // '0' = 48 number associated with 0 character
      printTreeData(treesDetected[i]);*/
    //}
    
    
  }
  //waitKey(0);
  cout << endl;
  vector<treeData> emptyList;
  if (status == S_SND_MTD) return snd_mtd_trees;
  else if (status == S_FST_MTD) return fst_mtd_trees;
  return emptyList; //status S_UNKWON -> no trees detected
  //return treesDetected;
}

int slidingWindow(Mat img, double scale, struct treeData &data ) {
  //cout << "Original size " << img.size() << " - scaling factor " << scale << endl;
  vector<int> methods = { TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR,  TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED };
  vector<string> methodsNames = { "TM_CCOEFF","TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED" };
  //vector<string> methodsNames = { "TM_CCOEFF", "TM_CCORR", "TM_SQDIFF" };
  //vector<int> methods = { TM_CCOEFF, TM_CCORR, TM_SQDIFF };
  int numCorrMatches = 0;
  double maxVal, minVal;
  Point topLeft, bottomRight, minLoc, maxLoc;
  Mat imgInput, originalImg, cImg;
  resize(img, img, Size(img.cols / scale, img.rows / scale));
  Canny(img, cImg, 500, 600);
  //namedWindow("canny img", WINDOW_NORMAL);
  //imshow("canny img", cImg);
  
  img.copyTo(originalImg);
  //cout << "Scaled size " << img.size() << endl;
  int win_rows = WIN_ROWS, win_cols = WIN_COLS, stepSize = 30, match;
  int i = 2; //method -> TM_CCOEFF
  //for (int i = 0; i < methods.size(); i++) {
    cImg.copyTo(imgInput);
    Mat res;
    matchTemplate(cImg, tImg, res, methods[i]);
    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    data.scale = scale;
    
    if(methods[i]== TM_SQDIFF || methods[i] == TM_SQDIFF_NORMED){
      topLeft = minLoc;
      //cout << "Method " << methodsNames[i] << " -> score: " << minVal ;
      data.score = minVal*(scale/2);
      //data.score = minVal;
    }
    else {
      topLeft = maxLoc;
      //cout << "Method " << methodsNames[i] << " -> score: " << maxVal ;
      //data.score = maxVal * scale;
      data.score = maxVal * (scale / 2);
    }
    bottomRight = Point2f(topLeft.x+tImg.cols,topLeft.y+tImg.rows);
    data.rect = Rect(topLeft, bottomRight);
    data.zncc = zncc(imgInput(data.rect), tImg);
    //cout << " ZNCC: " << data.zncc << endl;
    //draw rectangle in the output image
    //rectangle(imgInput, topLeft, bottomRight, Scalar(255));
    
    //namedWindow("Img at method " + methodsNames[i], WINDOW_NORMAL);
    //imshow("Img at method " + methodsNames[i], imgInput);
    //waitKey(1);
    /*int k = waitKey();
    data.qlt = (k - 48); // '0' = 48 number associated with 0 character
    cout << "Key pressed: " << data.qlt << endl;*/


  //}
  return numCorrMatches;
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

double avgZncc(Mat inputRect, string templPath) {
  vector<String> files;

  utils::fs::glob(templPath, "*.*", files);
  double tempZncc = 0.0;
  //read template images to compute avg zncc
  for (int j = 0; j < files.size(); j++) {
    Mat tImg = imread(files[j], IMREAD_GRAYSCALE);
    resize(tImg, tImg, inputRect.size());
    tempZncc += zncc(inputRect, tImg);
  }
  return (tempZncc / files.size());
}

void visualizeResults(vector<treeData> trees, Mat inputImg) {
  treeData selectedTree;
  if (status == S_SND_MTD) {
    //select tree with the lower ratio and a zncc < 500
    double lowestRatio = DBL_MAX;
    for (treeData tree : trees) {
      double ratio = tree.score / tree.zncc;
      if (ratio < lowestRatio && tree.zncc < 500) {
        lowestRatio = ratio;
        selectedTree = tree;
      }
    }
  }// if - second method
  else if (status == S_FST_MTD) {
    //select tree with higest score
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

vector<treeData> refineWithFeatureMatching(Mat inputImage, string templateFeaturePath, vector<treeData> selecTrees) {
  vector<treeData> refinedTrees;
  vector<String> files;
  vector <vector <KeyPoint>> templateKeyPoints;
  vector<Mat> templateDescrs;
  vector<Mat> templateImages;
  utils::fs::glob(templateFeaturePath, "*.*", files);
  //compute descriptors and key point for each template image
  for (int i = 0; i < files.size(); i++) {
    Mat templImg = imread(files[i]);
    //resize(templImg, templImg, Size(WIN_COLS, WIN_ROWS));
    templateImages.push_back(templImg);
    vector<KeyPoint> tFeatures;
    Mat tDescr;
    cout << "Extract features template " << files[i] << endl;
    extractFeatures(templImg, tFeatures, tDescr, 500);
    templateKeyPoints.push_back(tFeatures);
    templateDescrs.push_back(tDescr);
  }
  //for every selected tree refine the number of trees by comparing features with template images
  for (treeData tree : selecTrees) {
    int totMatches = 0;
    double totDist = 0;
    for (int i = 0; i < files.size(); i++) {
      vector<KeyPoint> rectFeatures;
      Mat rectDescr;
      vector<DMatch> matches;
      Mat outImg;
      //cout << "Extract features img - rect " << tree.rect << endl;
      //cout <<  " - img size " << inputImage.size() << " - img rect size " << inputImage(tree.rect).size() <<  endl;
      extractFeatures(inputImage(tree.rect), rectFeatures, rectDescr, 500);
      //cout << "Compute matches wit template " << files[i] << endl;
      computeMatches(templateDescrs[i], rectDescr, matches, 1.25);
      for (DMatch d : matches) totDist += d.distance;
      totMatches += matches.size();
      //cout << "Number of matches: " << matches.size() << endl;
      //cout << "Draw matches" << endl;
      //drawMatches(templateImages[i], templateKeyPoints[i], inputImage(tree.rect), rectFeatures, matches, outImg);
      //namedWindow("Matches", WINDOW_NORMAL);
      //imshow("Matches", outImg);
      //waitKey(1);
    }
    double avgDist = totDist / (files.size());
    cout << "Average distances: " << avgDist << endl;
    if(avgDist <= 4000) refinedTrees.push_back(tree);
  }
  for (treeData tree : refinedTrees) {
    Mat t;
    inputImage.copyTo(t);

    rectangle(t, tree.rect, Scalar(125), 2);
    namedWindow("Final detection", WINDOW_NORMAL);
    imshow("Final detection", t);
    waitKey(0);
  }
  return refinedTrees;

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

void computeMatchesFlann(Mat templateDescriptors, Mat imageDescriptors, vector<DMatch> &matches, float ratio) {
  vector<vector<DMatch>> knn_matches;
  cout << "Matcher flann" << endl;
  Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
  cout << "Compute matches with flann" << endl;
  matcher->knnMatch(templateDescriptors, imageDescriptors, knn_matches, 2);
  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  cout << "refine matches" << endl;
  for (size_t i = 0; i < knn_matches.size(); i++)
  {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
      matches.push_back(knn_matches[i][0]);
    }
  }
  cout << "matches size: " << matches.size() << endl;

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