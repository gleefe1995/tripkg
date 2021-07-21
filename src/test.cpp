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
    
    
     
     char filename1[200];
  char filename2[200];

  Ptr<LineSegmentDetector> lsd=createLineSegmentDetector();
  


 sprintf(filename1, path_to_image, 0);
  sprintf(filename2, path_to_image, 1);
   Mat image1_c = imread(filename1);
  Mat image2_c = imread(filename2);
  if ( !image1_c.data || !image2_c.data ) { 
    cout<< " --(!) Error reading images " << std::endl; return -1;
  }
Mat image1,image2;
Mat dst;
    cvtColor(image1_c, image1, COLOR_BGR2GRAY);
    cvtColor(image2_c, image2, COLOR_BGR2GRAY);
  vector<Vec4f> lines_std;
namedWindow("dst",WINDOW_AUTOSIZE);
  
  lsd->detect(image1,lines_std);
  
  lsd->drawSegments(image1, lines_std);


  imshow("dst",image1);
  waitKey();


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