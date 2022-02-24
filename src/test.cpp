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

// #include <tripkg/SPextractor.h>
// #include <tripkg/SuperPoint.h>
#include <tripkg/ORBextractor.h>



using namespace std;
using namespace cv;
using namespace ORB_SLAM2;
// using namespace DBoW2;

const char* path_to_image = "/home/gleefe/Downloads/dataset/sequences/00/image_0/%06d.png";


// const int NIMAGES = 100;

int main(int argc, char** argv)
{
    
// sp
// int nFeatures=500;
// float scaleFactor=1.2;
// int nLevels=1;
// float iniThFAST=0.015;
// float minThFAST=0.007;
    //orb
    int nFeatures = 2000;
    float scaleFactor = 1.2;
    int nLevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;

// if(torch::cuda::is_available()){
// cout<<"cuda is available"<<"\n";

// }



     
     char filename1[200];
  char filename2[200];

  
 sprintf(filename1, path_to_image, 0);
  sprintf(filename2, path_to_image, 1);
   Mat image1_c = imread(filename1);
  Mat image2_c = imread(filename2);
//   if ( !image1_c.data || !image2_c.data ) { 
//     cout<< " --(!) Error reading images " << std::endl; return -1;
//   }
Mat image1,image2;
// Mat dst;

    cvtColor(image1_c, image1, COLOR_BGR2GRAY);
    cvtColor(image2_c, image2, COLOR_BGR2GRAY);
  
// namedWindow("dst",WINDOW_AUTOSIZE);
  
   
vector<KeyPoint> keyPoints;
vector<KeyPoint> keyPoints2;
Mat mDescriptors;
Mat mDescriptors2;
Mat spimg_before;
Mat spimg,spimg2;

ORBextractor* extractor;


  
 clock_t topNStart = clock();
extractor = new ORBextractor(nFeatures,scaleFactor,nLevels,iniThFAST,minThFAST);

// clock_t topNStart = clock();

(*extractor)(image1,cv::Mat(),keyPoints,mDescriptors);

clock_t topNTotalTime1 =
      double(clock() - topNStart) * 1000 / (double)CLOCKS_PER_SEC;
  cout << "Finish SuperPoint first image in " << topNTotalTime1 << " miliseconds." << endl;
(*extractor)(image2,cv::Mat(),keyPoints2,mDescriptors2);


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
  cout << "Finish SuperPoint two image in " << topNTotalTime << " miliseconds." << endl;

//  clock_t FASTTotalTime =
//       double(clock() - FASTStart) * 1000 / (double)CLOCKS_PER_SEC;
//   cout << "Finish FAST in " << FASTTotalTime << " miliseconds." << endl;

  // KeyPoint::convert(sscKP, points1, vector<int>());

// Mat Fastimg;
drawKeypoints(image1,keyPoints,spimg,Scalar::all(-1));
// drawKeypoints(image2,keyPoints2,spimg2,Scalar::all(-1));
// drawKeypoints(image1,sscKP,Fastimg,Scalar::all(-1));
// cout<<mvKeys.size()<<"\n";
// cout<<sscKP.size()<<"\n";

imshow("superpoint",spimg);
// imshow("superpoint2",spimg2);
// imshow("FastImage", Fastimg);


/*
{
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( mDescriptors, mDescriptors2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches( image1, keyPoints, image2, keyPoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT );
    //-- Show detected matches
    imshow("Good Matches", img_matches );
}
*/




waitKey();


return 0;
}


