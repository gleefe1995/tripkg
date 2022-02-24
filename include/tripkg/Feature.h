
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

namespace Feature{

vector<cv::KeyPoint> ssc(vector<cv::KeyPoint> keyPoints, int numRetPoints, float tolerance, int cols, int rows)
{
  // several temp expression variables to simplify solution equation
  int exp1 = rows + cols + 2 * numRetPoints;
  long long exp2 = ((long long)4 * cols + (long long)4 * numRetPoints + (long long)4 * rows * numRetPoints + (long long)rows * rows + (long long)cols * cols - (long long)2 * rows * cols + (long long)4 * rows * cols * numRetPoints);
  double exp3 = sqrt(exp2);
  double exp4 = numRetPoints - 1;

  double sol1 = -round((exp1 + exp3) / exp4); // first solution
  double sol2 = -round((exp1 - exp3) / exp4); // second solution

  int high = (sol1 > sol2) ? sol1 : sol2; //binary search range initialization with positive solution
  int low = floor(sqrt((double)keyPoints.size() / numRetPoints));

  int width;
  int prevWidth = -1;

  vector<int> ResultVec;
  bool complete = false;
  unsigned int K = numRetPoints;
  unsigned int Kmin = round(K - (K * tolerance));
  unsigned int Kmax = round(K + (K * tolerance));

  vector<int> result;
  result.reserve(keyPoints.size());
  while (!complete)
  {
    width = low + (high - low) / 2;
    if (width == prevWidth || low > high)
    {                     //needed to reassure the same radius is not repeated again
      ResultVec = result; //return the keypoints from the previous iteration
      break;
    }
    result.clear();
    double c = width / 2; //initializing Grid
    int numCellCols = floor(cols / c);
    int numCellRows = floor(rows / c);
    vector<vector<bool>> coveredVec(numCellRows + 1, vector<bool>(numCellCols + 1, false));

    for (unsigned int i = 0; i < keyPoints.size(); ++i)
    {
      int row = floor(keyPoints[i].pt.y / c); //get position of the cell current point is located at
      int col = floor(keyPoints[i].pt.x / c);
      if (coveredVec[row][col] == false)
      { // if the cell is not covered
        result.push_back(i);
        int rowMin = ((row - floor(width / c)) >= 0) ? (row - floor(width / c)) : 0; //get range which current radius is covering
        int rowMax = ((row + floor(width / c)) <= numCellRows) ? (row + floor(width / c)) : numCellRows;
        int colMin = ((col - floor(width / c)) >= 0) ? (col - floor(width / c)) : 0;
        int colMax = ((col + floor(width / c)) <= numCellCols) ? (col + floor(width / c)) : numCellCols;
        for (int rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov)
        {
          for (int colToCov = colMin; colToCov <= colMax; ++colToCov)
          {
            if (!coveredVec[rowToCov][colToCov])
              coveredVec[rowToCov][colToCov] = true; //cover cells within the square bounding box with width w
          }
        }
      }
    }

    if (result.size() >= Kmin && result.size() <= Kmax)
    { //solution found
      ResultVec = result;
      complete = true;
    }
    else if (result.size() < Kmin)
      high = width - 1; //update binary search range
    else
      low = width + 1;
    prevWidth = width;
  }
  // retrieve final keypoints
  vector<cv::KeyPoint> kp;
  for (unsigned int i = 0; i < ResultVec.size(); i++)
    kp.push_back(keyPoints[ResultVec[i]]);

  return kp;
}


void featureDetection(Mat img_1, vector<Point2f> &points1, vector<pair<int, pair<int, Point2f>>> &points1_map, int &keyframe_number, int MAX_CORNERS
                          )
{

  // points1_map.clear();
  // vector <pair<int,Point2f>>().swap(points1_map);
  // goodFeaturesToTrack(img_1, points1, MAX_CORNERS, 0.01, 10);
  //  Size winSize = Size( 5, 5 );
  //  Size zeroZone = Size( -1, -1 );
  //  TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
  //  cornerSubPix( img_1, points1, winSize, zeroZone, criteria );
  //***********************************************************************************

  vector<KeyPoint> keyPoints;
  int fast_threshold = 1;
  bool nonmaxSuppression = true;
  FAST(img_1, keyPoints, fast_threshold, nonmaxSuppression);
  
  Mat mask;

  //detector->detect(img_1, keyPoints,mask);

  // KeyPoint::convert(keypoints_1, points1, vector<int>());
  int numRetPoints = MAX_CORNERS; //choose exact number of return points
  //float percentage = 0.1; //or choose percentage of points to be return
  //int numRetPoints = (int)keyPoints.size()*percentage;

  float tolerance = 0.1; // tolerance of the number of return points

  //Sorting keypoints by deacreasing order of strength
  vector<float> responseVector;
  for (unsigned int i = 0; i < keyPoints.size(); i++)
    responseVector.push_back(keyPoints[i].response);
  vector<int> Indx(responseVector.size());
  std::iota(std::begin(Indx), std::end(Indx), 0);
  cv::sortIdx(responseVector, Indx, SORT_DESCENDING);
  vector<cv::KeyPoint> keyPointsSorted;
  for (unsigned int i = 0; i < keyPoints.size(); i++)
    keyPointsSorted.push_back(keyPoints[Indx[i]]);

  vector<cv::KeyPoint> sscKP = ssc(keyPointsSorted, numRetPoints, tolerance, img_1.cols, img_1.rows);
  
  KeyPoint::convert(sscKP, points1, vector<int>());
  //cout << "The number of new detected points" << points1.size() << "\n";

  //***********************************************************************************
  vector<pair<int, pair<int, Point2f>>> points1_map_tmp;

  for (int i = 0; i < points1.size(); i++)
  {
    points1_map_tmp.push_back(make_pair(keyframe_number, make_pair(i, points1.at(i))));
  }
  points1_map = points1_map_tmp;

  keyframe_number++;
}




void featureTracking(Mat img_1, Mat img_2, vector<Point2f> &points1, vector<Point2f> &points2,
                     vector<pair<int, pair<int, Point2f>>> &points1_map, vector<pair<int, pair<int, Point2f>>> &points2_map,
                     vector<uchar> &status, vector<Point2f> &points2_tmp)
{

  //this function automatically gets rid of points for which tracking fails
  // points2_map.clear();
  // vector <pair<int,Point2f>>().swap(points2_map);
  vector<pair<int, pair<int, Point2f>>> points2_map_tmp;
  vector<float> err;
  Size winSize = Size(21, 21);
  TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
  //calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3);
  //calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
  // points2_tmp.clear();
  points2_tmp = points2;

  for (int i = 0; i < status.size(); i++)
  {
    points2_map_tmp.push_back(make_pair(points1_map[i].first, make_pair(points1_map[i].second.first, points2.at(i))));
  }
  //||(pt.x>img_1.cols)||(pt.y>img_1.rows)
  //||(pt.x>img_1.cols)||(pt.y>img_1.rows)
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for (int i = 0; i < status.size(); i++)
  {
    Point2f pt = points2.at(i - indexCorrection);
    if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
    {
      if ((pt.x < 0) || (pt.y < 0))
      {
        status.at(i) = 0;
      }
      points1.erase(points1.begin() + (i - indexCorrection));
      points2.erase(points2.begin() + (i - indexCorrection));

      // key_points1.erase(key_points1.begin() + (i - indexCorrection));
      // key_points2.erase(key_points2.begin() + (i - indexCorrection));

      points1_map.erase(points1_map.begin() + (i - indexCorrection));
      points2_map_tmp.erase(points2_map_tmp.begin() + (i - indexCorrection));

      indexCorrection++;
    }
  }

  points2_map = points2_map_tmp;
}

//||(pt.x>img_1.cols)||(pt.y>img_1.rows)
//||(pt.x>img_1.cols)||(pt.y>img_1.rows)
void erase_int_point2f(Mat img_1, vector<Point2f> &points2, vector<pair<int, pair<int, Point2f>>> &points1_map, vector<uchar> &status)
{
  int indexCorrection = 0;
  vector<Point2f> points2_tmp = points2;
  for (int i = 0; i < status.size(); i++)
  {
    Point2f pt = points2_tmp.at(i - indexCorrection);
    if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
    {
      if ((pt.x < 0) || (pt.y < 0))
      {
        status.at(i) = 0;
      }
      points2_tmp.erase(points2_tmp.begin() + (i - indexCorrection));
      points1_map.erase(points1_map.begin() + (i - indexCorrection));

      indexCorrection++;
    }
  }
}



}