#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "tripkg/vo.h"
#include "Thirdparty/DBoW2/include/DBoW2/DBoW2.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

//#include <tripkg/Frame.h>

#include <tripkg/SPextractor.h>
#include <tripkg/SuperPoint.h>




using namespace std;
using namespace cv;
using namespace ORB_SLAM2;
// using namespace DBoW2;
#define RATIO  0.4
const char* path_to_image = "/home/gleefe/Downloads/dataset/sequences/00/image_0/%06d.png";


// const int NIMAGES = 100;

int main(int argc, char** argv)
{
    

int nFeatures=2000;
float scaleFactor=1.2;
int nLevels=1;
float iniThFAST=0.015;
float minThFAST=0.007;
    
// if(torch::cuda::is_available()){
// cout<<"cuda is available"<<"\n";

// }



     
     char filename1[200];
//   char filename2[200];

  
 sprintf(filename1, path_to_image, 0);
//   sprintf(filename2, path_to_image, 1);
   Mat image1_c = imread(filename1);
//   Mat image2_c = imread(filename2);
//   if ( !image1_c.data || !image2_c.data ) { 
//     cout<< " --(!) Error reading images " << std::endl; return -1;
//   }
Mat image1,image2;
// Mat dst;

    cvtColor(image1_c, image1, COLOR_BGR2GRAY);
//     cvtColor(image2_c, image2, COLOR_BGR2GRAY);
  image2 = image1.clone();
// namedWindow("dst",WINDOW_AUTOSIZE);
  
   
vector<KeyPoint> keyPoints;

Mat mDescriptors;
Mat spimg_before;
Mat spimg;

ORBextractor* extractor;

clock_t topNStart = clock();
  
 
extractor = new ORBextractor(nFeatures,scaleFactor,nLevels,iniThFAST,minThFAST);

(*extractor)(image1,cv::Mat(),keyPoints,mDescriptors);



 /*
int numRetPoints = 1000; //choose exact number of return points
 float tolerance = 0.1; // tolerance of the number of return points
  //Sorting keypoints by deacreasing order of strength
  vector<float> responseVector;
  for (unsigned int i = 0; i < keyPoints.size(); i++)
    responseVector.push_back(keyPoints[i].response);

  // cout<<responseVector.size()<<"\n";

  vector<int> Indx(responseVector.size());
  std::iota(std::begin(Indx), std::end(Indx), 0);
  cv::sortIdx(responseVector, Indx, SORT_DESCENDING);
  vector<cv::KeyPoint> keyPointsSorted;
  for (unsigned int i = 0; i < keyPoints.size(); i++)
    keyPointsSorted.push_back(keyPoints[Indx[i]]);


  // cout<<keyPointsSorted.size()<<"\n";

  vector<cv::KeyPoint> sscKP = ssc(keyPointsSorted, numRetPoints, tolerance, image1.cols, image1.rows);
*/
 clock_t topNTotalTime =
      double(clock() - topNStart) * 1000 / (double)CLOCKS_PER_SEC;
  cout << "Finish SuperPoint in " << topNTotalTime << " miliseconds." << endl;

//  clock_t FASTTotalTime =
//       double(clock() - FASTStart) * 1000 / (double)CLOCKS_PER_SEC;
//   cout << "Finish FAST in " << FASTTotalTime << " miliseconds." << endl;

  // KeyPoint::convert(sscKP, points1, vector<int>());

// Mat Fastimg;
drawKeypoints(image1,keyPoints,spimg,Scalar::all(-1));
// drawKeypoints(image1,sscKP,Fastimg,Scalar::all(-1));
// cout<<mvKeys.size()<<"\n";
// cout<<sscKP.size()<<"\n";

imshow("superpoint",spimg);
// imshow("FastImage", Fastimg);
waitKey();


return 0;
}


// clock_t topNStart = clock();
//   vector<cv::KeyPoint> topnKP = topN(keyPointsSorted, numRetPoints);
//   clock_t topNTotalTime =
//       double(clock() - topNStart) * 1000 / (double)CLOCKS_PER_SEC;
//   cout << "Finish TopN in " << topNTotalTime << " miliseconds." << endl;
