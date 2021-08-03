#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <unordered_map>
#include <set>

#include <cmath>
#include <cstdio>

#include "ceres/ceres.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <std_msgs/UInt8MultiArray.h>

#include "Thirdparty/DBoW2/include/DBoW2/DBoW2.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/factory.h"
#include "g2o/core/robust_kernel_impl.h"

#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"

#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d/se3quat.h>

#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"

using namespace std;
using namespace cv;
using namespace DBoW2;
const int NIMAGES = 1101;
Ptr<ORB> detector = ORB::create(10000);
void loadFeatures(vector<vector<cv::Mat>> &features, const char *path_to_image);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat>> &features);
void testDatabase(const vector<vector<cv::Mat>> &features);
// using namespace cv::xfeatures2d;
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
// void featureDetection(Mat img_1, vector<Point2f>& points1, vector<pair<int,Point2f>>& points1_map)	{   //uses FAST as of now, modify parameters as necessary
//   vector<KeyPoint> keypoints_1;
//   int fast_threshold = 20;
//   bool nonmaxSuppression = true;
//   FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
//   KeyPoint::convert(keypoints_1, points1, vector<int>());
//   points1_map.clear();
//   for (int i=0;i<points1.size();i++){
//     points1_map.push_back(make_pair(i,points1.at(i)));
//   }
// }

void featureDetection_esential(Mat img_1, vector<Point2f> &points1, vector<pair<int, pair<int, Point2f>>> &points1_map, int &keyframe_number)
{

  // points1_map.clear();
  // vector <pair<int,Point2f>>().swap(points1_map);
  // goodFeaturesToTrack(img_1, points1, MAX_CORNERS, 0.01, 10);
  //  Size winSize = Size( 5, 5 );
  //  Size zeroZone = Size( -1, -1 );
  //  TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
  //  cornerSubPix( img_1, points1, winSize, zeroZone, criteria );
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());

  vector<pair<int, pair<int, Point2f>>> points1_map_tmp;

  for (int i = 0; i < points1.size(); i++)
  {
    points1_map_tmp.push_back(make_pair(keyframe_number, make_pair(i, points1.at(i))));
  }
  points1_map = points1_map_tmp;

  keyframe_number++;
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
  cv::sortIdx(responseVector, Indx, CV_SORT_DESCENDING);
  vector<cv::KeyPoint> keyPointsSorted;
  for (unsigned int i = 0; i < keyPoints.size(); i++)
    keyPointsSorted.push_back(keyPoints[Indx[i]]);

  vector<cv::KeyPoint> sscKP = ssc(keyPointsSorted, numRetPoints, tolerance, img_1.cols, img_1.rows);
  
  KeyPoint::convert(sscKP, points1, vector<int>());
  cout << "The number of new detected points" << points1.size() << "\n";

  //***********************************************************************************
  vector<pair<int, pair<int, Point2f>>> points1_map_tmp;

  for (int i = 0; i < points1.size(); i++)
  {
    points1_map_tmp.push_back(make_pair(keyframe_number, make_pair(i, points1.at(i))));
  }
  points1_map = points1_map_tmp;

  keyframe_number++;
}

// void featureDetection(Mat img_1, vector<Point2f>& points1, vector<pair<int,Point2f>>& points1_map)	{
//   int minHessian = 1500;
//   Ptr<ORB> detector = ORB::create(minHessian);
//   vector<KeyPoint> keypoints_1;
//   detector ->detect(img_1,keypoints_1);
//   KeyPoint::convert(keypoints_1,points1,vector<int>());
//   points1_map.clear();
//   for (int i=0;i<points1.size();i++){
//     points1_map.push_back(make_pair(i,points1.at(i)));
//   }
// }

// sort(BA_3d_points_map.begin(),BA_3d_points_map.end(),compare_point);
//       BA_3d_points_map.erase(unique(BA_3d_points_map.begin(),BA_3d_points_map.end()),BA_3d_points_map.end());

// bool compare_point (pair<int,pair<int,Point3d>> a,
//                     pair<int,pair<int,Point3d>> b){
//                       if(a.first==b.first){
//                         return a.second.first<b.second.first;
//                       }
//                       else{
//                         return a.first<b.first;
//                       }
//                     }

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

double gt_x = 0, gt_y = 0, gt_z = 0;
double getAbsoluteScale(int frame_id, int sequence_id, string path_to_pose)
{

  string line;
  int i = 0;
  ifstream myfile(path_to_pose);
  double x = 0, y = 0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while ((getline(myfile, line)) && (i <= frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;

      std::istringstream in(line);
      //cout << line << '\n';
      for (int j = 0; j < 12; j++)
      {
        in >> z;

        if (j == 7)
          y = z;
        if (j == 3)
          x = z;
      }

      i++;
    }
    myfile.close();
  }

  else
  {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));
}

void get_gt(int frame_id, int sequence_id, string path_to_pose)
{

  string line;
  int i = 0;
  ifstream myfile(path_to_pose);
  double x = 0, y = 0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while ((getline(myfile, line)) && (i <= frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;

      std::istringstream in(line);
      //cout << line << '\n';
      for (int j = 0; j < 12; j++)
      {
        in >> z;
        gt_z = z;
        if (j == 7)
          y = z, gt_y = y;
        if (j == 3)
          x = z, gt_x = x;
      }

      i++;
    }
    myfile.close();
  }

  else
  {
    cout << "Unable to open file";
  }
}

struct SnavelyReprojectionError
{
  SnavelyReprojectionError(double observed_x, double observed_y, Eigen::Vector4d point_3d_homo_eig, double focal, double ppx, double ppy)
      : observed_x(observed_x), observed_y(observed_y), point_3d_homo_eig(point_3d_homo_eig), focal(focal), ppx(ppx), ppy(ppy) {}

  template <typename T>
  bool operator()(const T *const rvec_eig,
                  const T *const tvec_eig,
                  T *residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation.

    const T theta = sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2]);

    const T tvec_eig_0 = tvec_eig[0];
    const T tvec_eig_1 = tvec_eig[1];
    const T tvec_eig_2 = tvec_eig[2];

    const T w1 = rvec_eig[0] / theta;
    const T w2 = rvec_eig[1] / theta;
    const T w3 = rvec_eig[2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    // Eigen::Matrix<T,3,3> R_solve_homo;
    // R_solve_homo << cos+w1*w1*(1-cos), w1*w2*(1-cos)-w3*sin, w1*w3*(1-cos)+w2*sin,
    //                 w1*w2*(1-cos)+w3*sin, cos+w2*w2*(1-cos), w2*w3*(1-cos)-w1*sin,
    //                 w1*w3*(1-cos)-w2*sin, w2*w3*(1-cos)+w1*sin, cos+w3*w3*(1-cos);

    Eigen::Matrix<T, 3, 4> Relative_homo_R;
    Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

    Eigen::Matrix<T, 3, 1> three_to_p_eig;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;
    // Kd = Kd.cast<T>();

    //Eigen::Matrix<T,4,1> point_3d_homo;
    //point_3d_homo_eig = point_3d_homo_eig.cast<T>();
    //point_3d_homo<<point_3d_homo_eig[0],point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3];

    three_to_p_eig = Kd.cast<T>() * Relative_homo_R * point_3d_homo_eig.cast<T>();
    // cv2eigen(three_to_p,three_to_p_eig);

    // three_to_p_eig[0]=three_to_p.at<double>(0);
    // three_to_p_eig[1]=three_to_p.at<double>(1);
    // three_to_p_eig[2]=three_to_p.at<double>(2);

    T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
    T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  // static ceres::CostFunction* Create(const double observed_x,
  //                                    const double observed_y,
  //                                    const Eigen::Vector4d point_3d_homo_eig,
  //                                    const double focal,
  //                                    const double ppx,
  //                                    const double ppy) {
  //   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3,3>(
  //       new SnavelyReprojectionError(observed_x, observed_y,point_3d_homo_eig,focal,ppx,ppy)));
  // }

  double observed_x;
  double observed_y;
  const Eigen::Vector4d point_3d_homo_eig;
  double focal;
  double ppx;
  double ppy;
};

struct SnavelyReprojectionError_Local
{
  SnavelyReprojectionError_Local(double observed_x, double observed_y, double focal, double ppx, double ppy, int j, Eigen::VectorXi number_of_3d_points_eig)
      : observed_x(observed_x), observed_y(observed_y), focal(focal), ppx(ppx), ppy(ppy), j(j), number_of_3d_points_eig(number_of_3d_points_eig) {}

  template <typename T>
  bool operator()(const T *const rvec_eig,
                  const T *const tvec_eig,
                  const T *const point_3d_homo_eig,
                  T *residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation.

    int number_of_3d_points = 0;

    int count = 0;
    //cout<<"index of BA_2d_points: "<<j<<"\n";
    for (int i = 0; i < number_of_3d_points_eig.size(); i++)
    {
      number_of_3d_points += number_of_3d_points_eig[i];
      if (j < number_of_3d_points)
      {
        count = i;
        break;
      }
    }

    const T theta = sqrt(rvec_eig[3 * count] * rvec_eig[3 * count] + rvec_eig[3 * count + 1] * rvec_eig[3 * count + 1] + rvec_eig[3 * count + 2] * rvec_eig[3 * count + 2]);
    const T tvec_eig_0 = tvec_eig[3 * count];
    const T tvec_eig_1 = tvec_eig[3 * count + 1];
    const T tvec_eig_2 = tvec_eig[3 * count + 2];

    const T w1 = rvec_eig[3 * count] / theta;
    const T w2 = rvec_eig[3 * count + 1] / theta;
    const T w3 = rvec_eig[3 * count + 2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    // Eigen::Matrix<T,3,3> R_solve_homo;
    // R_solve_homo << cos+w1*w1*(1-cos), w1*w2*(1-cos)-w3*sin, w1*w3*(1-cos)+w2*sin,
    //                 w1*w2*(1-cos)+w3*sin, cos+w2*w2*(1-cos), w2*w3*(1-cos)-w1*sin,
    //                 w1*w3*(1-cos)-w2*sin, w2*w3*(1-cos)+w1*sin, cos+w3*w3*(1-cos);

    Eigen::Matrix<T, 3, 4> Relative_homo_R;
    Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;
    
    Eigen::Matrix<T, 1, 3> three_to_p_eig;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;
    // Kd = Kd.cast<T>();

    //Eigen::Matrix<T,4,1> point_3d_homo;
    //point_3d_homo_eig = point_3d_homo_eig.cast<T>();
    //point_3d_homo<<point_3d_homo_eig[0],point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3];

    Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1], point_3d_homo_eig[2], static_cast<T>(1));
    // cout<<p3he(0,0)<<"\n";
    // cout<<p3he(1,0)<<"\n";
    // cout<<p3he(2,0)<<"\n";
    // cout<<p3he(3,0)<<"\n";

    three_to_p_eig = Kd.cast<T>() * Relative_homo_R * p3he;

    // cv2eigen(three_to_p,three_to_p_eig);
    //cout<<p3he<<"\n";
    // waitKey();
    //cout<<Kd.cast<T>()<<"\n";

    //   cout<<three_to_p_eig<<"\n";

    // cout<<Relative_homo_R(0,3)<<"\n";
    // cout<<Relative_homo_R(1,3)<<"\n";
    // cout<<Relative_homo_R(2,3)<<"\n";

    // three_to_p_eig[0]=three_to_p.at<double>(0);
    // three_to_p_eig[1]=three_to_p.at<double>(1);
    // three_to_p_eig[2]=three_to_p.at<double>(2);

    T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
    T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    // cout<<residuals[0]<<"\n";
    // cout<<residuals[1]<<"\n";
    // waitKey();
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  // static ceres::CostFunction* Create(const double observed_x,
  //                                    const double observed_y,
  //                                    const double focal,
  //                                    const double ppx,
  //                                    const double ppy) {
  //   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3,3,4>(
  //       new SnavelyReprojectionError_Local(observed_x, observed_y,focal,ppx,ppy)));
  // }

  double observed_x;
  double observed_y;
  int j;
  double focal;
  double ppx;
  double ppy;
  Eigen::VectorXi number_of_3d_points_eig;
};



// ----------------------------------------------------------------------------
struct SnavelyReprojectionError_full_ba
{
  SnavelyReprojectionError_full_ba(double observed_x, double observed_y, double focal, double ppx, double ppy, int j, Eigen::VectorXi number_of_3d_points_eig)
      : observed_x(observed_x), observed_y(observed_y), focal(focal), ppx(ppx), ppy(ppy), j(j), number_of_3d_points_eig(number_of_3d_points_eig) {}

  template <typename T>
  bool operator()(const T *const rvec_eig,
                  const T *const tvec_eig,
                  const T *const point_3d_homo_eig,
                  T *residuals) const
  {
    
    const T theta = sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2]);
    const T tvec_eig_0 = tvec_eig[0];
    const T tvec_eig_1 = tvec_eig[1];
    const T tvec_eig_2 = tvec_eig[2];

    
    const T w1 = rvec_eig[0] / theta;
    const T w2 = rvec_eig[1] / theta;
    const T w3 = rvec_eig[2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    // Eigen::Matrix<T,3,3> R_solve_homo;
    // R_solve_homo << cos+w1*w1*(1-cos), w1*w2*(1-cos)-w3*sin, w1*w3*(1-cos)+w2*sin,
    //                 w1*w2*(1-cos)+w3*sin, cos+w2*w2*(1-cos), w2*w3*(1-cos)-w1*sin,
    //                 w1*w3*(1-cos)-w2*sin, w2*w3*(1-cos)+w1*sin, cos+w3*w3*(1-cos);

    Eigen::Matrix<T, 3, 4> Relative_homo_R;
    Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;
    
    
    Eigen::Matrix<T, 1, 3> three_to_p_eig;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;
    // Kd = Kd.cast<T>();

    //Eigen::Matrix<T,4,1> point_3d_homo;
    //point_3d_homo_eig = point_3d_homo_eig.cast<T>();
    //point_3d_homo<<point_3d_homo_eig[0],point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3];

    Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1], point_3d_homo_eig[2], static_cast<T>(1));
    // cout<<p3he(0,0)<<"\n";
    // cout<<p3he(1,0)<<"\n";
    // cout<<p3he(2,0)<<"\n";
    // cout<<p3he(3,0)<<"\n";

    three_to_p_eig = Kd.cast<T>() * Relative_homo_R * p3he;

    // cv2eigen(three_to_p,three_to_p_eig);
    //cout<<"p3he: "<<p3he<<"\n";
    // waitKey();
    //cout<<Kd.cast<T>()<<"\n";

    //   cout<<three_to_p_eig<<"\n";

    // cout<<"RhR 03"<<Relative_homo_R(0,3)<<"\n";
    // cout<<"RhR 13"<<Relative_homo_R(1,3)<<"\n";
    // cout<<"RhR 23"<<Relative_homo_R(2,3)<<"\n";

    // three_to_p_eig[0]=three_to_p.at<double>(0);
    // three_to_p_eig[1]=three_to_p.at<double>(1);
    // three_to_p_eig[2]=three_to_p.at<double>(2);

    T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
    T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

    // cout<<"predicted_x: "<<predicted_x<<"\n";
    // cout<<"predicted_y: "<<predicted_y<<"\n";
    // cout<<"j: "<<j<<"\n";
    // cout<<"observed_x: "<<T(observed_x)<<"\n";
    // cout<<"observed_y: "<<T(observed_y)<<"\n";
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    // cout<<"residual 0"<<residuals[0]<<"\n";
    // cout<<"residual 1"<<residuals[1]<<"\n";
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  // static ceres::CostFunction* Create(const double observed_x,
  //                                    const double observed_y,
  //                                    const double focal,
  //                                    const double ppx,
  //                                    const double ppy) {
  //   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3,3,4>(
  //       new SnavelyReprojectionError_Local(observed_x, observed_y,focal,ppx,ppy)));
  // }

  double observed_x;
  double observed_y;
  int j;
  double focal;
  double ppx;
  double ppy;
  Eigen::VectorXi number_of_3d_points_eig;
};


struct SnavelyReprojectionError_Local_pose_fixed
{
  SnavelyReprojectionError_Local_pose_fixed(double observed_x, double observed_y, double focal, double ppx, double ppy, int j, Eigen::VectorXi number_of_3d_points_eig,
                                            Eigen::VectorXd rvec_eig, Eigen::VectorXd tvec_eig)
      : observed_x(observed_x), observed_y(observed_y), focal(focal), ppx(ppx), ppy(ppy), j(j), number_of_3d_points_eig(number_of_3d_points_eig),rvec_eig(rvec_eig), tvec_eig(tvec_eig) {}

  template <typename T>
  bool operator()(const T *const point_3d_homo_eig,
                  T *residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation.

    int number_of_3d_points = 0;

    int count = 0;
    //cout<<"index of BA_2d_points: "<<j<<"\n";
    // for (int i = 0; i < number_of_3d_points_eig.size(); i++)
    // {
    //   number_of_3d_points += number_of_3d_points_eig[i];
    //   if (j < number_of_3d_points)
    //   {
    //     count = i;
    //     break;
    //   }
    // }

    const T theta = static_cast<T>( sqrt(rvec_eig[3 * count] * rvec_eig[3 * count] + rvec_eig[3 * count + 1] * rvec_eig[3 * count + 1] + rvec_eig[3 * count + 2] * rvec_eig[3 * count + 2]));
    const T tvec_eig_0 = static_cast<T>(tvec_eig[3 * count]);
    const T tvec_eig_1 = static_cast<T>(tvec_eig[3 * count + 1]);
    const T tvec_eig_2 = static_cast<T>(tvec_eig[3 * count + 2]);

    const T w1 = static_cast<T>(rvec_eig[3 * count] / theta);
    const T w2 = static_cast<T>(rvec_eig[3 * count + 1] / theta);
    const T w3 = static_cast<T>(rvec_eig[3 * count + 2] / theta);

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    // Eigen::Matrix<T,3,3> R_solve_homo;
    // R_solve_homo << cos+w1*w1*(1-cos), w1*w2*(1-cos)-w3*sin, w1*w3*(1-cos)+w2*sin,
    //                 w1*w2*(1-cos)+w3*sin, cos+w2*w2*(1-cos), w2*w3*(1-cos)-w1*sin,
    //                 w1*w3*(1-cos)-w2*sin, w2*w3*(1-cos)+w1*sin, cos+w3*w3*(1-cos);

    Eigen::Matrix<T, 3, 4> Relative_homo_R;
    Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

    Eigen::Matrix<T, 1, 3> three_to_p_eig;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;
    // Kd = Kd.cast<T>();
    
    //Eigen::Matrix<T,4,1> point_3d_homo;
    //point_3d_homo_eig = point_3d_homo_eig.cast<T>();
    //point_3d_homo<<point_3d_homo_eig[0],point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3];

    Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1], point_3d_homo_eig[2], static_cast<T>(1));
    // cout<<p3he(0,0)<<"\n";
    // cout<<p3he(1,0)<<"\n";
    // cout<<p3he(2,0)<<"\n";
    // cout<<p3he(3,0)<<"\n";

    three_to_p_eig = Kd.cast<T>() * Relative_homo_R * p3he;

    // cv2eigen(three_to_p,three_to_p_eig);
    // cout<<p3he<<"\n";
    // waitKey();
    //cout<<Kd.cast<T>()<<"\n";

    //   cout<<three_to_p_eig<<"\n";

    // cout<<Relative_homo_R(0,3)<<"\n";
    // cout<<Relative_homo_R(1,3)<<"\n";
    // cout<<Relative_homo_R(2,3)<<"\n";

    // three_to_p_eig[0]=three_to_p.at<double>(0);
    // three_to_p_eig[1]=three_to_p.at<double>(1);
    // three_to_p_eig[2]=three_to_p.at<double>(2);

    T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
    T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    // cout<<residuals[0]<<"\n";
    // cout<<residuals[1]<<"\n";
    // waitKey();
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  // static ceres::CostFunction* Create(const double observed_x,
  //                                    const double observed_y,
  //                                    const double focal,
  //                                    const double ppx,
  //                                    const double ppy) {
  //   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3,3,4>(
  //       new SnavelyReprojectionError_Local(observed_x, observed_y,focal,ppx,ppy)));
  // }

  double observed_x;
  double observed_y;
  int j;
  double focal;
  double ppx;
  double ppy;
  Eigen::VectorXi number_of_3d_points_eig;
  Eigen::VectorXd rvec_eig;
  Eigen::VectorXd tvec_eig;
};















//--------------------------------------------------------------------------
void loadFeatures(vector<vector<cv::Mat>> &features, const char *path_to_image)
{
  features.clear();
  features.reserve(NIMAGES);
  char filename1[200];
  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for (int i = 0; i < NIMAGES; ++i)
  {
    sprintf(filename1, path_to_image, NIMAGES);
    // stringstream ss;
    // ss << "images/image" << i << ".png";

    cv::Mat image = cv::imread(filename1, 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);
    
    features.push_back(vector<cv::Mat>());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for (int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat>> &features)
{
  // branching factor and depth levels
  const int k = 10;
  const int L = 4;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
       << voc << endl
       << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for (int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for (int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl
       << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat>> &features, OrbDatabase &db, bool &Isloopdetected, int &keyframe_prev_id, int &keyframe_curr_id)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  // OrbVocabulary voc("/home/gleefe/catkin_ws/small_voc.yml.gz");

  // OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database

  // for(int i = 0; i < features.size(); i++)
  // {
  //   db.add(features[i]);
  // }
  db.add(features.back());
  cout << "... done!" << endl;

  cout << "Database information: " << endl
       << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret2;
  for (int i = 0; i < features.size(); i++)
  {
    db.query(features[i], ret2, 15);

    // ret[0] is always the same image in this case, because we added it to the
    
    // database. ret[1] is the second best match.
    if (i==features.size()-1){
    cout << "Searching for Image " << i<<" "<<ret2<<"\n";
    }

    if (features.size() > 15)
    {
      int entry_id = 0;
      double score = 0;
      for (int j = 0; j < 15; j++)
      {
        int idid = ret2[j].Id;

        if (abs(i - idid) > 15)
        {
          entry_id = ret2[j].Id;
          score = ret2[j].Score;
          break;
        }
      }
      if (score != 0)
      {
        //cout << "Searching for Image " << i<<" "<<"Best search Id and Score: "<<entry_id<<" "<<score<<"\n";

        if ( (score > 0.14)&&(i==features.size()-1))
        {
          cout << "loop detected!!"<< "\n";
          cout << "Score: "<<score << "\n";
          keyframe_prev_id = min(entry_id, i);
          keyframe_curr_id = max(entry_id, i);
          // ret=ret2;
          Isloopdetected = 1;
        }
      }
    }
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  // cout << "Saving database..." << endl;
  // db.save("small_db.yml.gz");
  // cout << "... done!" << endl;

  // once saved, we can load it again
  // cout << "Retrieving database once again..." << endl;
  // OrbDatabase db2("small_db.yml.gz");
  // cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------
//g2o

void addPoseVertex(g2o::SparseOptimizer *optimizer, g2o::SE3Quat &pose, bool set_fixed,int id)
{
  // std::cout << "add pose: t=" << pose.translation().transpose()
  //           << " r=" << pose.rotation().coeffs().transpose() << std::endl;
  g2o::VertexSE3 *v_se3 = new g2o::VertexSE3;
  v_se3->setId(id);
  if (set_fixed)
    v_se3->setEstimate(pose);
  v_se3->setFixed(set_fixed);
  optimizer->addVertex(v_se3);
}

void addEdgePosePose(g2o::SparseOptimizer *optimizer, int id0, int id1, g2o::SE3Quat &relpose)
{
  // std::cout << "add edge: id0=" << id0 << ", id1=" << id1
  //           << ", t=" << relpose.translation().transpose()
  //           << ", r=" << relpose.rotation().coeffs().transpose() << std::endl;

  g2o::EdgeSE3 *edge = new g2o::EdgeSE3;
  edge->setVertex(0, optimizer->vertices().find(id0)->second);
  edge->setVertex(1, optimizer->vertices().find(id1)->second);
  edge->setMeasurement(relpose);
  Eigen::MatrixXd info_matrix = Eigen::MatrixXd::Identity(6, 6) * 10.;
  edge->setInformation(info_matrix);
  optimizer->addEdge(edge);
}

void ToVertexSim3(const g2o::VertexSE3 &v_se3,
                  g2o::VertexSim3Expmap *const v_sim3, double scale)
{
  Eigen::Isometry3d se3 = v_se3.estimate().inverse();
  Eigen::Matrix3d r = se3.rotation();
  Eigen::Vector3d t = se3.translation();

  // cout<<"Convert vertices to Sim3: "<<"\n";
  // cout<<"r: "<<se3.rotation()<<"\n";
  // cout<<"t: "<<se3.translation()<<"\n";
  g2o::Sim3 sim3(r, t, scale);

  v_sim3->setEstimate(sim3);
}

// Converte EdgeSE3 to EdgeSim3
void ToEdgeSim3(const g2o::EdgeSE3 &e_se3, g2o::EdgeSim3 *const e_sim3,double scale)
{
  Eigen::Isometry3d se3 = e_se3.measurement().inverse();
  Eigen::Matrix3d r = se3.rotation();
  Eigen::Vector3d t = se3.translation();

  // cout<<"Convert edges to Sim3:"<<"\n";
  // cout<<"r: "<<se3.rotation()<<"\n";
  // cout<<"t: "<<se3.translation()<<"\n";
  g2o::Sim3 sim3(r, t, scale);

  e_sim3->setMeasurement(sim3);
}
//------------------------------------------------------------------------------------

/*
struct SnavelyReprojectionError_Local_pose_fix {
  SnavelyReprojectionError_Local_pose_fix(double observed_x, double observed_y,double focal,double ppx,double ppy,int j,Eigen::VectorXi number_of_3d_points_eig,Eigen::VectorXd rvec_)
      : observed_x(observed_x), observed_y(observed_y),focal(focal), ppx(ppx),ppy(ppy),j(j),number_of_3d_points_eig(number_of_3d_points_eig) {}

  template <typename T>
  bool operator()(
                  const T* const point_3d_homo_eig,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    
     
    int number_of_3d_points=0;
   
    int count=0;
    //cout<<"index of BA_2d_points: "<<j<<"\n";
    for (int i=0;i<number_of_3d_points_eig.size();i++){
      number_of_3d_points+=number_of_3d_points_eig[i];
    if (j<number_of_3d_points){
      count=i;
    
    break;
    }
    
    }
    // cout<<"index of keyframe: "<<count<<"\n";
    


    const T theta=sqrt (rvec_eig[3*count]*rvec_eig[3*count] + rvec_eig[3*count+1]*rvec_eig[3*count+1] + rvec_eig[3*count+2]*rvec_eig[3*count+2]);
     const T tvec_eig_0=tvec_eig[3*count];
     const T tvec_eig_1=tvec_eig[3*count+1];
     const T tvec_eig_2=tvec_eig[3*count+2];
    
     const T w1=rvec_eig[3*count]/theta;
     const T w2=rvec_eig[3*count+1]/theta;
     const T w3=rvec_eig[3*count+2]/theta;
    


    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    // Eigen::Matrix<T,3,3> R_solve_homo;
    // R_solve_homo << cos+w1*w1*(1-cos), w1*w2*(1-cos)-w3*sin, w1*w3*(1-cos)+w2*sin,
    //                 w1*w2*(1-cos)+w3*sin, cos+w2*w2*(1-cos), w2*w3*(1-cos)-w1*sin,
    //                 w1*w3*(1-cos)-w2*sin, w2*w3*(1-cos)+w1*sin, cos+w3*w3*(1-cos);

    
    Eigen::Matrix<T,3,4> Relative_homo_R;
    Relative_homo_R<<cos+w1*w1*( static_cast<T>(1)-cos),   w1*w2*(static_cast<T>(1)-cos)-w3*sin, w1*w3*(static_cast<T>(1)-cos)+w2*sin, tvec_eig_0,
                    w1*w2*(static_cast<T>(1)-cos)+w3*sin, cos+w2*w2*(static_cast<T>(1)-cos), w2*w3*(static_cast<T>(1)-cos)-w1*sin, tvec_eig_1,
                    w1*w3*(static_cast<T>(1)-cos)-w2*sin, w2*w3*(static_cast<T>(1)-cos)+w1*sin, cos+w3*w3*(static_cast<T>(1)-cos), tvec_eig_2;
                    

      
      Eigen::Matrix<T,1,3> three_to_p_eig;
      
      Eigen::Matrix<double,3,3> Kd;
      Kd << focal, 0, ppx,
              0  , focal, ppy,
              0  ,  0   ,  1;
      // Kd = Kd.cast<T>();
      
      //Eigen::Matrix<T,4,1> point_3d_homo;
      //point_3d_homo_eig = point_3d_homo_eig.cast<T>();
      //point_3d_homo<<point_3d_homo_eig[0],point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3];

      
      Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3]);
      // cout<<p3he(0,0)<<"\n";
      // cout<<p3he(1,0)<<"\n";
      // cout<<p3he(2,0)<<"\n";
      // cout<<p3he(3,0)<<"\n";
     
     
      three_to_p_eig=Kd.cast<T>()*Relative_homo_R*p3he;
      
      // cv2eigen(three_to_p,three_to_p_eig);
      // cout<<p3he<<"\n";
      // waitKey();
    //cout<<Kd.cast<T>()<<"\n";
     

    //   cout<<three_to_p_eig<<"\n";
     
     
      
        
      // cout<<Relative_homo_R(0,3)<<"\n";
      // cout<<Relative_homo_R(1,3)<<"\n";
      // cout<<Relative_homo_R(2,3)<<"\n";
      

      

      // three_to_p_eig[0]=three_to_p.at<double>(0);
      // three_to_p_eig[1]=three_to_p.at<double>(1);
      // three_to_p_eig[2]=three_to_p.at<double>(2);
      
      T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
      T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);
      
      

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    
    // cout<<residuals[0]<<"\n";
    // cout<<residuals[1]<<"\n";
    // waitKey();
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  // static ceres::CostFunction* Create(const double observed_x,
  //                                    const double observed_y,
  //                                    const double focal,
  //                                    const double ppx,
  //                                    const double ppy) {
  //   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3,3,4>(
  //       new SnavelyReprojectionError_Local(observed_x, observed_y,focal,ppx,ppy)));
  // }

  double observed_x;
  double observed_y;
  int j;
  double focal;
  double ppx;
  double ppy;
  Eigen::VectorXi number_of_3d_points_eig;
};


*/

//    static ceres::CostFunction* Create(const double observed_x,
//                                    const double observed_y,
//                                    const Eigen::Vector4d point_3d_homo_eig,
//                                    const double focal,
//                                    const double ppx,
//                                    const double ppy) {
//   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3,3>(
//       new SnavelyReprojectionError(observed_x, observed_y,point_3d_homo_eig,focal,ppx,ppy)));
// }
// ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
//       corr_2d_point_eig(0,i), corr_2d_point_eig(1,i),corr_3d_point_eig.col(i),focal,pp.x,pp.y);
//   problem.AddResidualBlock(cost_function,
//                            NULL ,
//                            rvec_eig.data(),
//                            tvec_eig.data());

// Make Ceres automatically detect the bundle structure. Note that the
// standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
// for standard bundle adjustment problems.

//*******************************************************************************
//sprintf(filename, path_to_image, numFrame);
//imread(filename);
//vector<pair<numFrame,<rvec,vector>,<tvec,vector>>

//rvec,tvec,,numFrame,

/*
 if (numFrame_vec.size()==local_ba_frame){
   cout<<"localBA"<<"\n";

      Size winSize1 = Size( 5, 5 );
      Size zeroZone = Size( -1, -1 );
      TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
      Eigen::MatrixXd rvec_vec_eig(3,local_ba_frame);
      Eigen::MatrixXd tvec_vec_eig(3,local_ba_frame);
      char local_frame_file_name[100];
      Mat local_image_c;
      
      for (int i=0;i<local_ba_frame;i++){
        rvec_vec_eig(0,i)=rvec_vec[i].x;
        rvec_vec_eig(1,i)=rvec_vec[i].y;
        rvec_vec_eig(2,i)=rvec_vec[i].z;
        tvec_vec_eig(0,i)=tvec_vec[i].x;
        tvec_vec_eig(1,i)=tvec_vec[i].y;
        tvec_vec_eig(2,i)=tvec_vec[i].z;
      }
      for (int i=0;i<local_ba_frame;i++){

       sprintf(local_frame_file_name, path_to_image, numFrame_vec[i]);
       local_image_c = imread(local_frame_file_name);
       cvtColor(local_image_c, local_image, COLOR_BGR2GRAY);

      vector<Point2f> local_Features;
      vector<Point2f> new_currFeatures_corrf(new_currFeatures_corr.begin(),new_currFeatures_corr.end());
      Mat local_desc, new_currdesc;
      vector<KeyPoint> local_keypoints, new_currkeypoints;
      //Ptr<ORB> detector_orb = ORB::create(2000);
      goodFeaturesToTrack(local_image, local_Features, 4000, 0.01, 1);
      //cornerSubPix( local_image, local_Features, winSize1, zeroZone, criteria );
      //detector->detect(local_image,local_keypoints,vector<int>());
      KeyPoint::convert( local_Features, local_keypoints);
      KeyPoint::convert( new_currFeatures_corrf, new_currkeypoints);

      detector->compute(local_image,local_keypoints,local_desc);
      detector->compute(currImage,new_currkeypoints,new_currdesc);
      
      BFMatcher matcher(NORM_HAMMING);
      vector<DMatch> matches;
      matcher.match(new_currdesc,local_desc,matches);


      double minDist,maxDist;
      minDist=maxDist=matches[0].distance;
      for (int i=1;i<matches.size();i++){
      double dist = matches[i].distance;
      if (dist<minDist) minDist=dist;
      if (dist>maxDist) maxDist=dist;
}


      vector<DMatch> goodMatches;
      double fTh= 4*minDist;
      for (int i=0;i<matches.size();i++){
        if (matches[i].distance <=max(fTh,0.02))
          goodMatches.push_back(matches[i]);
        }
      cout<<goodMatches.size()<<"\n";
      Mat img_match;

      drawMatches(currImage, new_currkeypoints, local_image, local_keypoints, goodMatches, img_match);
      cout<<goodMatches.size()<<"\n";
      cout<<matches.size()<<"\n";
      imshow("Matches", img_match);
      waitKey();
        }


}
*/
/*
for (int i=0;i<point_3d_map_erase.size();i++){
    point_3d_map.push_back(point_3d_map_erase[i]);
    point_3d_map_first.push_back(point_3d_map_first_erase[i]);
  }
  

  sort(point_3d_map.begin(),point_3d_map.end(),compare1);
  sort(point_3d_map_first.begin(),point_3d_map_first.end());

   
    // Mat local_R_solve;
    // Mat local_R_solve_inv;
    // Mat local_t_solve_f;
    Rodrigues(local_rvec,local_R_solve);
    
    
    local_R_solve_inv = local_R_solve.t();
    local_t_solve_f = -local_R_solve_inv*local_tvec;
    
    local_R_solve.copyTo(Relative_homo_R.rowRange(0,3).colRange(0,3));
    local_tvec.copyTo(Relative_homo_R.rowRange(0,3).col(3));

    
    
    cvtColor(local_image, local_image_c, COLOR_GRAY2BGR);
    for(int i = 0; i < point_3d_map.size(); i++) {
      
      int m = new_currFeatures_corr.at(i).x;
      int n = new_currFeatures_corr.at(i).y;
      circle(local_image_c, Point(m, n) ,2, CV_RGB(255,0,0), 2);
      
      int point_3d_map_number = point_3d_map.size();
      Mat prev_p3hh(4,point_3d_map_number,CV_64F);
      // Mat prev_p3d2;
      // Mat prev_p3d22;
      Mat prev_p3h;
      prev_p3h = prev_p3hh.col(i); 
      // prev_p3d2 = prev_p3h/prev_p3h.at<float>(3);
      // prev_p3d2.convertTo(prev_p3d22, CV_64F);
      prev_p3h.at<double>(0)=point_3d_map[i].second.x;
      prev_p3h.at<double>(1)=point_3d_map[i].second.y;
      prev_p3h.at<double>(2)=point_3d_map[i].second.z;
      prev_p3h.at<double>(3)=1;
      
        three_to_p=Kd*Relative_homo_R*prev_p3h;
        int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
        int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
        circle(local_image_c, Point(c, d) ,2, CV_RGB(0,255,0), 2);
      
      
      
  }

     imshow( "local_image", local_image_c );
     */

//****************************************************************************************BRIEF descriptor*************************************************************************************************
/*
   cout<<"BRIEF descriptor start"<<"\n";
vector<KeyPoint> keypoints_1, keypoints_2;
Mat descriptor_1, descriptor_2;
   KeyPoint::convert( currFeatures, keypoints_1);
KeyPoint::convert( new_currFeatures, keypoints_2);
// detector->compute(image1,Mat() ,keypoints_1, descriptor_1);
// detector->compute(image2, Mat(),keypoints_2, descriptor_2);
detector->compute(currImage,keypoints_1,descriptor_1);
detector->compute(currImage,keypoints_2,descriptor_2);


BFMatcher matcher(NORM_HAMMING);
vector<DMatch> matches;
// cout<<descriptor_1.cols<<"\n";
// cout<<descriptor_2.cols<<"\n";
//****************************************************************************************delete same point*************************************************************************************************


if((descriptor_1.rows>0)&&(descriptor_2.rows>0)){
matcher.match(descriptor_1,descriptor_2,matches);


double minDist,maxDist;
minDist=maxDist=matches[0].distance;
for (int i=1;i<matches.size();i++){
    double dist = matches[i].distance;
    if (dist<minDist) minDist=dist;
    if (dist>maxDist) maxDist=dist;
}


vector<DMatch> goodMatches;
//vector<DMatch> badMatches;
double fTh= 2*minDist;
for (int i=0;i<matches.size();i++){
    if (matches[i].distance <=max(fTh,0.02)){
    goodMatches.push_back(matches[i]);
    }
    
}
// draw same point matching
 
Mat img_match;
 drawMatches(currImage, keypoints_1, currImage, keypoints_2, goodMatches, img_match);
cout<<"same point size: "<<goodMatches.size()<<"\n";
// cout<<matches.size()<<"\n";
 imshow("Matches", img_match);
 waitKey();

cout<<"BRIEF descriptor end"<<"\n";


   
int index=0;
int new_curr_size=new_currFeatures_tmp_tmp.size();
cout<<"new_currFeatures_tmp_tmp size: "<<new_currFeatures_tmp_tmp.size()<<"\n";
    for (int i=0;i<new_curr_size;i++){
      int count=0;
      for (int j=0;j<goodMatches.size();j++){
        
      if (i==goodMatches[j].trainIdx){
          count++;
      }
      }
      if (count==0){
      new_currFeatures_tmp_tmp.push_back(new_currFeatures_tmp_tmp[i]);
      new_curr_points_map_tmp_tmp.push_back(new_curr_points_map_tmp_tmp[i]);
      point_3d_map_tmp_tmp.push_back(point_3d_map_tmp_tmp[i]);
      }
      }
      new_currFeatures_tmp_tmp.erase(new_currFeatures.begin(),new_currFeatures.begin()+new_curr_size);
      new_curr_points_map_tmp_tmp.erase(new_curr_points_map_tmp_tmp.begin(),new_curr_points_map_tmp_tmp.begin()+new_curr_size);
      point_3d_map_tmp_tmp.erase(point_3d_map_tmp_tmp.begin(),point_3d_map_tmp_tmp.begin()+new_curr_size);
    
    cout<<"delete same point: "<<goodMatches.size()<<"\n";
    
    cout<<"after delete same point curr Features size: "<<new_currFeatures_tmp_tmp.size()<<"\n";
}
*/
//****************************************************************************************delete same point end*************************************************************************************************
//****************************************************************************************BRIEF descriptor end*************************************************************************************************



//after loop closing bundle adjustment
/*
{

if (number_of_3d_points.size()==local_ba_frame){
  cout<<"local BA start"<<"\n";
  //cout<<"before local ba tvec: "<<tvec.at<double>(0)<<" "<<tvec.at<double>(1)<<" "<<tvec.at<double>(2)<<"\n";

  int rvec_eig_local_size=3*local_ba_frame;
    // Eigen::MatrixXd rvec_eig_local(1,3*local_ba_frame);
    // Eigen::MatrixXd tvec_eig_local(1,3*local_ba_frame);
    Eigen::VectorXd rvec_eig_local(rvec_eig_local_size);
    Eigen::VectorXd tvec_eig_local(rvec_eig_local_size);
    


    for (int i=0; i<local_ba_frame;i++){
      rvec_eig_local[3*i]=rvec_vec[i].x;
      rvec_eig_local[3*i+1]=rvec_vec[i].y;
      rvec_eig_local[3*i+2]=rvec_vec[i].z;
      tvec_eig_local[3*i]=tvec_vec[i].x;
      tvec_eig_local[3*i+1]=tvec_vec[i].y;
      tvec_eig_local[3*i+2]=tvec_vec[i].z;
    }    
    
   
    Eigen::VectorXd BA_2d_points_eig(2);
    Eigen::MatrixXd BA_3d_points_eig(3,BA_3d_points_map_tmp.size());
    Eigen::VectorXi number_of_3d_points_eig(number_of_3d_points.size());
    
    int half_3d_points=0;
    for (int i=0;i<local_ba_frame;i++){
      number_of_3d_points_eig[i]=number_of_3d_points[i];
      //cout<<number_of_3d_points_eig[i]<<"\n";
      if (i<local_ba_frame/2){
        half_3d_points+=number_of_3d_points_eig[i];
      }
    }
    

    for (int i=0;i<BA_3d_points_map_tmp.size();i++){
      BA_3d_points_eig(0,i)=BA_3d_points_map_tmp[i].second.second.x;
      BA_3d_points_eig(1,i)=BA_3d_points_map_tmp[i].second.second.y;
      BA_3d_points_eig(2,i)=BA_3d_points_map_tmp[i].second.second.z;
      // BA_3d_points_eig(3,i)=1;
      
    }
    
    
    
    ceres::Problem problem2;
    
    int index=0;
    
    cout<<half_3d_points<<"\n";
    //-number_of_3d_points[local_ba_frame-1]
    for (int i = 0; i < BA_2d_points_map.size(); i++) {
          index++;
          //cout<<BA_2d_points_map.at(i).first*10000+BA_2d_points_map.at(i).second.first<<"\n";
          auto it =find(BA_3d_map_points.begin(), BA_3d_map_points.end(), BA_2d_points_map.at(i).first*10000+BA_2d_points_map.at(i).second.first);
          
          //auto it = BA_3d_map_points.find(BA_2d_points_map[j].at(i).first*1000+BA_2d_points_map[j].at(i).second.first);
          BA_2d_points_eig[0]=(double)BA_2d_points_map.at(i).second.second.x;
          BA_2d_points_eig[1]=(double)BA_2d_points_map.at(i).second.second.y;
          
          
           ceres::CostFunction* cost_function2 = 
          new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local_pose_fixed, 2,3>(
            new SnavelyReprojectionError_Local_pose_fixed(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,i,number_of_3d_points_eig,rvec_eig_local,tvec_eig_local)
          );
       
    
    problem2.AddResidualBlock(cost_function2,
                             NULL ,
                             BA_3d_points_eig.col(it-BA_3d_map_points.begin()).data());
          }
      
      
  cout<<"local BA solver start"<<"\n";
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 12;
  options.max_num_iterations=1000;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem2, &summary);
  // int num_threads=summary.num_threads_used;
  // cout<<num_threads<<"\n";
  // waitKey();
  //std::cout << summary.FullReport() << "\n";
  //**********************8
  
  //*************************



  
  rvec_vec.clear();
  //vector <Point3d>().swap(rvec_vec);
  tvec_vec.clear();
  //vector <Point3d>().swap(tvec_vec);
   for (int i=0;i<local_ba_frame;i++){
     double rvec_eig_1=rvec_eig_local[3*i];
     double rvec_eig_2=rvec_eig_local[3*i+1];
     double rvec_eig_3=rvec_eig_local[3*i+2];
     double tvec_eig_1=tvec_eig_local[3*i+0];
     double tvec_eig_2=tvec_eig_local[3*i+1];
     double tvec_eig_3=tvec_eig_local[3*i+2];
     
     
      rvec_vec.push_back(Point3d(rvec_eig_1,rvec_eig_2,rvec_eig_3));
      tvec_vec.push_back(Point3d(tvec_eig_1,tvec_eig_2,tvec_eig_3));
      
    }
    
    
    int BA_3d_points_map_size=BA_3d_points_map.size();
    // vector <pair<int,pair<int,Point3d>>>().swap(BA_3d_points_map);
    int point_3d_map_size=point_3d_map.size();
  for (int i=0;i<BA_3d_points_map_size;i++){
      int map_first=BA_3d_points_map[i].first;
      int map_second_first=BA_3d_points_map[i].second.first;
      
      auto it =find(BA_3d_map_points.begin(), BA_3d_map_points.end(), map_first*10000+map_second_first);
      
      int eig_index=it-BA_3d_map_points.begin();
      BA_3d_points_map.push_back(make_pair(BA_3d_points_map[i].first,make_pair(BA_3d_points_map[i].second.first,Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index)))));
      
      if (i>=BA_3d_points_map_size-number_of_3d_points[number_of_3d_points.size()-1]){
        point_3d_map.push_back(make_pair(BA_3d_points_map[i].first,make_pair(BA_3d_points_map[i].second.first,Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index)))));
      }
    }
    

BA_3d_points_map.erase(BA_3d_points_map.begin(),BA_3d_points_map.begin()+BA_3d_points_map_size);
point_3d_map.erase(point_3d_map.begin(),point_3d_map.begin()+point_3d_map_size);


// for (int i=0;i<point_3d_map.size();i++){
//   cout<<point_3d_map[i].second.second<<"\n";
//   waitKey();
// }
  //cout<<"after local ba point_3d_map size: "<<point_3d_map.size()<<"\n";
  cout<<"BA_3d_points_map size: "<<BA_3d_points_map.size()<<"\n";
     
     

  rvec.at<double>(0)=rvec_vec[local_ba_frame-1].x;
  rvec.at<double>(1)=rvec_vec[local_ba_frame-1].y;
  rvec.at<double>(2)=rvec_vec[local_ba_frame-1].z;

  tvec.at<double>(0)=tvec_vec[local_ba_frame-1].x;
  tvec.at<double>(1)=tvec_vec[local_ba_frame-1].y;
  tvec.at<double>(2)=tvec_vec[local_ba_frame-1].z;

  

Rodrigues(rvec,R_solve);
   
    R_solve_inv = R_solve.t();
    // cout<<R_solve_inv<<"\n";
    // waitKey(1000);
    t_solve_f = -R_solve_inv*tvec;


  for(int i=0;i<local_ba_frame;i++){
    trajectory = cloud2->points[i];
        
        Mat rvec_tmp(3,1,CV_64F);
        Mat tvec_tmp(3,1,CV_64F);
        
        rvec_tmp.at<double>(0)=rvec_vec[i].x;
        rvec_tmp.at<double>(1)=rvec_vec[i].y;
        rvec_tmp.at<double>(2)=rvec_vec[i].z;

        tvec_tmp.at<double>(0)=tvec_vec[i].x;
        tvec_tmp.at<double>(1)=tvec_vec[i].y;
        tvec_tmp.at<double>(2)=tvec_vec[i].z;

        Mat R_solve_tmp(3,3,CV_64F);
        Rodrigues(rvec_tmp,R_solve_tmp);
        
        Mat R_solve_inv_tmp=R_solve_tmp.t();
        Mat t_solve_f_tmp=-R_solve_inv_tmp*tvec_tmp;
        

        trajectory.x=t_solve_f_tmp.at<double>(0);
        trajectory.y=t_solve_f_tmp.at<double>(1);
        trajectory.z=t_solve_f_tmp.at<double>(2);
        trajectory.r=30;
        trajectory.g=144;
        trajectory.b=255;
        msg4->points.push_back(trajectory);
        msg2->points.push_back(trajectory);

        
  }
    keyframe_pub.publish(msg4);
    

}

}
*/


// Add landmark edges
        /*
        { 
          cout<<"Add landmark edges"<<"\n";
          int number_of_points=0;
          int number_of_points_prev=0;
          for (int i=0;i<gt_poses.size();i++){
            //cout<<"i: "<<i<<"\n";
          
      //       edge->setVertex(0, optimizer->vertices().find(poseid)->second);
      // edge->setVertex(1, optimizer->vertices().find(ptid)->second);
          number_of_points+=number_of_3d_points_loop_tmp[i];
          
          g2o::VertexSim3Expmap* vtx =
          static_cast<g2o::VertexSim3Expmap*>(optimizer_sim3.vertex(i));
          g2o::Sim3 sim3 = vtx->estimate().inverse();
          
          //cout<<"from j: "<<number_of_points_prev<<", to: "<<number_of_points<<"\n";
          for (int j=number_of_points_prev;j<number_of_points;j++){
            g2o::EdgeSE3PointXYZ* landmarkObservation =  new EdgeSE3PointXYZ;
          //landmarkObservation->vertices()[0] = optimizer_sim3.vertex(i);
          landmarkObservation->setVertex(0,optimizer_sim3.vertices().find(i)->second);
            auto it =find(BA_3d_map_points_tmp.begin(), BA_3d_map_points_tmp.end(), BA_2d_points_tmp.at(j).first*10000+BA_2d_points_tmp.at(j).second.first);

          //cout<<"it-BA_3d_map_points_tmp: "<<it-BA_3d_map_points_tmp.begin()<<"\n";

          //landmarkObservation->vertices()[1] = optimizer_sim3.vertex(gt_poses.size()+(it-BA_3d_map_points_tmp.begin()));
          landmarkObservation->setVertex(1,optimizer_sim3.vertices().find(gt_poses.size()+(it-BA_3d_map_points_tmp.begin()))->second);
          
          Eigen::Vector3d trans_landmark;
          if (i!=gt_poses.size()-1){
          trans_landmark[0]=-(sim3.translation()[0]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.x);
          trans_landmark[1]=-(sim3.translation()[1]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.y);
          trans_landmark[2]=-(sim3.translation()[2]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.z);
          }
          else{
            g2o::VertexSim3Expmap* vtx2 =
          static_cast<g2o::VertexSim3Expmap*>(optimizer_sim3.vertex(0));
          g2o::Sim3 sim32 = vtx2->estimate().inverse();
            trans_landmark[0]=-(sim32.translation()[0]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.x);
          trans_landmark[1]=-(sim32.translation()[1]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.y);
          trans_landmark[2]=-(sim32.translation()[2]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.z);
          }
          //cout<<"trans_landmark: "<<trans_landmark<<"\n";
          landmarkObservation->setMeasurement(trans_landmark);
          // landmarkObservation->setInformation(simEdge.information);
          // landmarkObservation->information() = Eigen::Matrix<double, 3, 3>::Identity();
          Eigen::MatrixXd info_matrix = Eigen::MatrixXd::Identity(3,3);
          for(int i=0; i<3; i++){
              info_matrix(i, i) = 1;
          }
          landmarkObservation->setInformation(info_matrix);
          
          //cout<<"Add landmark edge"<<"\n";
          landmarkObservation->setParameterId(0, 0);
          //optimizer_sim3.addEdge(landmarkObservation);
          
          }
          number_of_points_prev=number_of_points;
          }
        
        }
        cout<<"edge size: "<<optimizer_sim3.edges().size()<<"\n";
        
        cout<<"vertex size: "<<optimizer_sim3.vertices().size()<<"\n";
        */











       //landmark optimization
        //Add landmark vertices
        /*
          cout<<"Add landmark vertices"<<"\n";
          {
            g2o::VertexPointXYZ* landmark = new VertexPointXYZ;
            
            for (int i=0;i<BA_3d_points_tmp_tmp.size();i++){
            landmark->setId(i+gt_poses.size());
            Eigen::Vector3d trans_vtx;
            // trans_vtx->x
            trans_vtx[0]=BA_3d_points_tmp_tmp[i].second.x;
            trans_vtx[1]=BA_3d_points_tmp_tmp[i].second.y;
            trans_vtx[2]=BA_3d_points_tmp_tmp[i].second.z;
            landmark->setEstimate(trans_vtx);
            optimizer_sim3.addVertex(landmark);          
            }
          }
          cout<<"vertex size: "<<optimizer_sim3.vertices().size()<<"\n";
          cout<<"BA_3d_map_points_tmp size: "<<BA_3d_map_points_tmp.size()<<"\n";
          cout<<"BA_3d_points_tmp size: "<<BA_3d_points_tmp_tmp.size()<<"\n";


        // Add landmark edges
        
        { 
          cout<<"Add landmark edges"<<"\n";
          int number_of_points=0;
          int number_of_points_prev=0;
          for (int i=0;i<gt_poses.size();i++){
            //cout<<"i: "<<i<<"\n";
          
      //       edge->setVertex(0, optimizer->vertices().find(poseid)->second);
      // edge->setVertex(1, optimizer->vertices().find(ptid)->second);
          number_of_points+=number_of_3d_points_loop_tmp[i];
          
          g2o::VertexSim3Expmap* vtx =
          static_cast<g2o::VertexSim3Expmap*>(optimizer_sim3.vertex(i));
          g2o::Sim3 sim3 = vtx->estimate().inverse();
          
          //cout<<"from j: "<<number_of_points_prev<<", to: "<<number_of_points<<"\n";
          for (int j=number_of_points_prev;j<number_of_points;j++){
            g2o::EdgeSE3PointXYZ* landmarkObservation =  new EdgeSE3PointXYZ;
          //landmarkObservation->vertices()[0] = optimizer_sim3.vertex(i);
          landmarkObservation->setVertex(0,optimizer_sim3.vertices().find(i)->second);
            auto it =find(BA_3d_map_points_tmp.begin(), BA_3d_map_points_tmp.end(), BA_2d_points_tmp.at(j).first*10000+BA_2d_points_tmp.at(j).second.first);

          //cout<<"it-BA_3d_map_points_tmp: "<<it-BA_3d_map_points_tmp.begin()<<"\n";

          //landmarkObservation->vertices()[1] = optimizer_sim3.vertex(gt_poses.size()+(it-BA_3d_map_points_tmp.begin()));
          landmarkObservation->setVertex(1,optimizer_sim3.vertices().find(gt_poses.size()+(it-BA_3d_map_points_tmp.begin()))->second);
          
          Eigen::Vector3d trans_landmark;
          
          trans_landmark[0]=-(sim3.translation()[0]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.x);
          trans_landmark[1]=-(sim3.translation()[1]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.y);
          trans_landmark[2]=-(sim3.translation()[2]-BA_3d_points_tmp_tmp[it-BA_3d_map_points_tmp.begin()].second.z);
          
          
          //cout<<"trans_landmark: "<<trans_landmark<<"\n";
          landmarkObservation->setMeasurement(trans_landmark);
          // landmarkObservation->setInformation(simEdge.information);
          // landmarkObservation->information() = Eigen::Matrix<double, 3, 3>::Identity();
          Eigen::MatrixXd info_matrix = Eigen::MatrixXd::Identity(3,3);
          for(int i=0; i<3; i++){
              info_matrix(i, i) = 1;
          }
          landmarkObservation->setInformation(info_matrix);
          
          //cout<<"Add landmark edge"<<"\n";
          landmarkObservation->setParameterId(0, 0);
          //optimizer_sim3.addEdge(landmarkObservation);
          
          }
          number_of_points_prev=number_of_points;
          }
        
        }
        cout<<"edge size: "<<optimizer_sim3.edges().size()<<"\n";
        
        cout<<"vertex size: "<<optimizer_sim3.vertices().size()<<"\n";





        cout<<"initializing ..."<<"\n";
        optimizer_sim3.initializeOptimization();
        cout << "optimizing ..." << endl;
        optimizer_sim3.optimize(100);
        */











        /*
        // vector<pair<int,pair<int,Point3d>>> BA_3d_points_tmp;
          vector<pair<int,Point3d>> BA_3d_points_tmp;
          
          vector<pair<int,pair<int,Point2f>>> BA_2d_points_tmp;
          vector<int> number_of_3d_points_loop_tmp;
          //int index_of_points=0;
          // for (int i=0;i<keyframe_prev;i++){
          //   index_of_points+=number_of_3d_points_loop[i];
          // }

          int whole_size=0;
            for (int i=0;i<=keyframe_curr;i++){
              number_of_3d_points_loop_tmp.push_back(number_of_3d_points_loop[i]);
              whole_size+=number_of_3d_points_loop[i];
            }
            
          BA_3d_points_tmp.reserve(BA_3d_points_map_loop.size());  
            vector<int> BA_3d_map_points_tmp;
            for (int i=0;i<BA_3d_points_map_loop.size();i++){
              
              BA_3d_points_tmp.push_back(make_pair(10000*BA_3d_points_map_loop[i].first+BA_3d_points_map_loop[i].second.first,BA_3d_points_map_loop[i].second.second));
              BA_2d_points_tmp.push_back(BA_2d_points_map_loop[i]);
              BA_3d_map_points_tmp.push_back(10000*BA_3d_points_map_loop[i].first+BA_3d_points_map_loop[i].second.first);
              
            }

          
          cout<<whole_size<<"\n";
          cout<<number_of_3d_points_loop_tmp.size()<<"\n";
          cout<<BA_2d_points_tmp.size()<<"\n";
          cout<<BA_3d_points_tmp.size()<<"\n";
          cout<<BA_2d_points_map_loop.size()<<"\n";
          cout<<BA_3d_points_map_loop.size()<<"\n";
          cout<<"sorting BA_3d_pionts"<<"\n";

          sort(BA_3d_points_tmp.begin(),BA_3d_points_tmp.end(),compare_point2);
          sort(BA_3d_map_points_tmp.begin(),BA_3d_map_points_tmp.end());

          
          int BA_3d_points_tmp_size=BA_3d_points_tmp.size();
          vector<pair<int,Point3d>> BA_3d_points_tmp_tmp;
          for (int i=0;i<BA_3d_points_tmp.size();i++){
            if ((i>0)&&(BA_3d_points_tmp[i-1].first!=BA_3d_points_tmp[i].first))
            {
              BA_3d_points_tmp_tmp.push_back(BA_3d_points_tmp[i]);
            }
            else if(i==0){
              BA_3d_points_tmp_tmp.push_back(BA_3d_points_tmp[i]);
            }
          }
          



          // auto it = unique(BA_3d_points_tmp.begin(),BA_3d_points_tmp.end());
          // BA_3d_points_tmp.erase(it,BA_3d_points_tmp.end());

          
          vector<int>::iterator it2=unique(BA_3d_map_points_tmp.begin(),BA_3d_map_points_tmp.end());
          BA_3d_map_points_tmp.erase(it2,BA_3d_map_points_tmp.end());
          
          // waitKey();
          
          
          // cout<<BA_3d_points_tmp_tmp.size()<<"\n";
          */