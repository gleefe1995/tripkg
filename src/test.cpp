#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "tripkg/vo.h"
#include "Thirdparty/DBoW2/include/DBoW2/DBoW2.h"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
// using namespace DBoW2;
#define RATIO  0.4
const char* path_to_image = "/home/gleefe/Downloads/dataset/sequences/00/image_0/%06d.png";


void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                       const int numToKeep );




// const int NIMAGES = 100;

int main(int argc, char** argv)
{
    vector<vector<cv::Mat > > features;
    
     
     char filename1[200];
  char filename2[200];



 sprintf(filename1, path_to_image, 0);
  sprintf(filename2, path_to_image, 1);
   Mat image1_c = imread(filename1);
  Mat image2_c = imread(filename2);
  if ( !image1_c.data || !image2_c.data ) { 
    cout<< " --(!) Error reading images " << std::endl; return -1;
  }
Mat image1,image2;

    cvtColor(image1_c, image1, COLOR_BGR2GRAY);
    cvtColor(image2_c, image2, COLOR_BGR2GRAY);
    Mat descriptor_1, descriptor_2;
vector<Point2f> points1;
vector<Point2f> points2;
vector<uchar> status;



vector<KeyPoint> keypoints_1, keypoints_2;

// image1 = imread("/home/gleefe1995/Downloads/dataset/sequences/00/image_0/000000.png", IMREAD_GRAYSCALE);
// image2 = imread("/home/gleefe1995/Downloads/dataset/sequences/00/image_0/000006.png", IMREAD_GRAYSCALE);






// goodFeaturesToTrack(image1, points1, 1500, 0.01, 10);
//  Size winSize1 = Size( 5, 5 );
//  Size zeroZone = Size( -1, -1 );
//  TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
//  cornerSubPix( image1, points1, winSize1, zeroZone, criteria );

//   vector<float> err;					
//   Size winSize=Size(21,21);																								
//   TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

//   calcOpticalFlowPyrLK(image1, image2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
  
 

//   //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
//   int indexCorrection = 0;
//   for( int i=0; i<status.size(); i++)
//      {  Point2f pt = points2.at(i- indexCorrection);
//      	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)||(pt.x>image1.cols)||(pt.y>image1.rows))	{
//      		  if((pt.x<0)||(pt.y<0)||(pt.x>image1.cols)||(pt.y>image1.rows))	{
//      		  	status.at(i) = 0;
//      		  }
//      		  points1.erase (points1.begin() + (i - indexCorrection));
//      		  points2.erase (points2.begin() + (i - indexCorrection));
          
          
//      		  indexCorrection++;
//      	}

//      }


// KeyPoint::convert( points1, keypoints_1);
// KeyPoint::convert( points2, keypoints_2);
// detector->compute(image1,Mat() ,keypoints_1, descriptor_1);
// detector->compute(image2, Mat(),keypoints_2, descriptor_2);
// detector->compute(image1,keypoints_1,descriptor_1);
// detector->compute(image2,keypoints_2,descriptor_2);
//detector->detectAndCompute(image2, mask, keypoints_2, descriptor_2);
 // type of mask is CV_8U
// roi is a sub-image of mask specified by cv::Rect object

// we set elements in roi region of the mask to 255 
vector<KeyPoint> keyPoints;
int hessian=10000;
Mat mask;
Ptr<ORB> detector = ORB::create(hessian);
detector->detect(image1,keyPoints,mask);
cout<<keyPoints.size()<<"\n";


 int numRetPoints = 1500; //choose exact number of return points
    //float percentage = 0.1; //or choose percentage of points to be return
    //int numRetPoints = (int)keyPoints.size()*percentage;

    float tolerance = 0.1; // tolerance of the number of return points

    //Sorting keypoints by deacreasing order of strength
    vector<float> responseVector;
    for (unsigned int i =0 ; i<keyPoints.size(); i++) responseVector.push_back(keyPoints[i].response);
    vector<int> Indx(responseVector.size()); std::iota (std::begin(Indx), std::end(Indx), 0);
    cv::sortIdx(responseVector, Indx, CV_SORT_DESCENDING);
    vector<cv::KeyPoint> keyPointsSorted;
    for (unsigned int i = 0; i < keyPoints.size(); i++) keyPointsSorted.push_back(keyPoints[Indx[i]]);

vector<cv::KeyPoint> sscKP = ssc(keyPointsSorted,numRetPoints,tolerance,image1.cols,image1.rows);
KeyPoint::convert( sscKP,points1);
KeyPoint::convert(points1,keypoints_1);
for (int i=0;i<keypoints_1.size();i++){
  cout<<keypoints_1[i].response<<"\n";
  cout<<sscKP[i].response<<"\n";
}
cout<<sscKP.size()<<"\n";
//adaptiveNonMaximalSuppresion(keypoints_1,1000);

// for (int i=0;i<image1.rows-30;i+=30){
//   for (int j=0;j<image1.cols-30;j+=30){
//     vector<KeyPoint> keypoints_1_tmp;
//     vector<Point2f> points1_tmp;
   
//     Mat mask = Mat::zeros(image1.size(), CV_8U); 
//     Rect bounds(0,0,image1.cols,image1.rows);
//   Mat roi(mask, cv::Rect(0+i,0+j,i+50,j+100) & bounds);
//   roi = Scalar(255);
// detector->detect(image1,keypoints_1_tmp,mask);
// KeyPoint::convert( keypoints_1_tmp,points1_tmp);
// for (int k=0;k<points1_tmp.size();k++){
// points1.push_back(points1_tmp[k]);
// }
//     // cout<<"j: "<<j<<"\n";
//   }
  
//   // cout<<i<<" "<<"\n";
// }
// cout<<points1.size()<<"\n";
// imshow("image1", image1_c);
// waitKey();
// for (int i=0;i<image1.rows;i++){
//   for (int j=0;j<image1.cols;j++){
    
//   }
// }

for (int i=0;i<sscKP.size();i++){
  
  int m=sscKP[i].pt.x;
  int n=sscKP[i].pt.y;
  
  //cout<<m<<" "<<n<<"\n";
  //cout<<keypoints_1[i].response<<"\n";
  circle(image1_c, Point(m,n),2,CV_RGB(0,0,255),2);
}
imshow("image1", image1_c);
waitKey();
// cout<<descriptor_1.rows<<"\n"; //739
// cout<<descriptor_1.cols<<"\n"; //32
//  features.push_back(vector<cv::Mat >());
//     changeStructure(descriptor_1, features.back());
//     features.push_back(vector<cv::Mat >());
//     changeStructure(descriptor_2, features.back());
//*****************************************test voc creation**********************************


//FlannBasedMatcher matcher;
//Ptr<DescriptorExtractor> extractor;
//Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12,20,2));
BFMatcher matcher(NORM_HAMMING);
vector<DMatch> matches;
matcher.match(descriptor_1,descriptor_2,matches);


double minDist,maxDist;
minDist=maxDist=matches[0].distance;
for (int i=1;i<matches.size();i++){
    double dist = matches[i].distance;
    if (dist<minDist) minDist=dist;
    if (dist>maxDist) maxDist=dist;
}


vector<DMatch> goodMatches;
double fTh= 8*minDist;
for (int i=0;i<matches.size();i++){
    if (matches[i].distance <=max(fTh,0.02))
    goodMatches.push_back(matches[i]);
}

Mat img_match;

drawMatches(image1, keypoints_1, image2, keypoints_2, goodMatches, img_match);
cout<<goodMatches.size()<<"\n";
cout<<matches.size()<<"\n";
//imshow("Matches", img_match);
// for (int i=0;i<matches.size();++i){
    
//     Point2f pt1 = keypoints_1[matches[i].queryIdx].pt;
//     Point2f pt2 = keypoints_2[matches[i].trainIdx].pt;
//     float a = pt1.x;
//     float b = pt1.y;
//     float c = pt2.x;
//     float d = pt2.y;
    
//     if ( (matches[i].distance<maxdist*RATIO))
//         goodMatches.push_back(matches[i]);
// }

//&& ((sqrt( (a-c)*(a-c)+(b-d)*(b-d)) )<100)
//(matches[i].distance<maxdist*RATIO) && 


return 0;
}


// void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
// {
//   out.resize(plain.rows);

//   for(int i = 0; i < plain.rows; ++i)
//   {
//     out[i] = plain.row(i);
//   }
// }


// void testDatabase(const vector<vector<cv::Mat > > &features)
// {
//   cout << "Creating a small database..." << endl;

//   // load the vocabulary from disk
//   OrbVocabulary voc("small_voc.yml.gz");
  
//   OrbDatabase db(voc, false, 0); // false = do not use direct index
//   // (so ignore the last param)
//   // The direct index is useful if we want to retrieve the features that 
//   // belong to some vocabulary node.
//   // db creates a copy of the vocabulary, we may get rid of "voc" now

//   // add images to the database
//   for(int i = 0; i < NIMAGES; i++)
//   {
//     db.add(features[i]);
//   }

//   cout << "... done!" << endl;

//   cout << "Database information: " << endl << db << endl;

//   // and query the database
//   cout << "Querying the database: " << endl;

//   QueryResults ret;
//   for(int i = 0; i < NIMAGES; i++)
//   {
//     db.query(features[i], ret, 4);

//     // ret[0] is always the same image in this case, because we added it to the 
//     // database. ret[1] is the second best match.

//     cout << "Searching for Image " << i << ". " << ret << endl;
//   }

//   cout << endl;

//   // we can save the database. The created file includes the vocabulary
//   // and the entries added
//   cout << "Saving database..." << endl;
//   db.save("small_db.yml.gz");
//   cout << "... done!" << endl;
  
//   // once saved, we can load it again  
//   cout << "Retrieving database once again..." << endl;
//   OrbDatabase db2("small_db.yml.gz");
//   cout << "... done! This is: " << endl << db2 << endl;
// }



 void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                       const int numToKeep )
    {
      if( keypoints.size() < numToKeep ) { return; }

      //
      // Sort by response
      //
      std::sort( keypoints.begin(), keypoints.end(),
                 [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
                 {
                   return lhs.response > rhs.response;
                 } );

      std::vector<cv::KeyPoint> anmsPts;

      std::vector<double> radii;
      radii.resize( keypoints.size() );
      std::vector<double> radiiSorted;
      radiiSorted.resize( keypoints.size() );

      const float robustCoeff = 1.11; // see paper

      for( int i = 0; i < keypoints.size(); ++i )
      {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for( int j = 0; j < i && keypoints[j].response > response; ++j )
        {
          radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        radii[i]       = radius;
        radiiSorted[i] = radius;
      }

      std::sort( radiiSorted.begin(), radiiSorted.end(),
                 [&]( const double& lhs, const double& rhs )
                 {
                   return lhs > rhs;
                 } );

      const double decisionRadius = radiiSorted[numToKeep];
      for( int i = 0; i < radii.size(); ++i )
      {
        if( radii[i] >= decisionRadius )
        {
          anmsPts.push_back( keypoints[i] );
        }
      }

      anmsPts.swap( keypoints );
    }