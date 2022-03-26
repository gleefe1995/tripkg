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

// #include <torch/torch.h>

// #include <tripkg/ORBextractor.h>

using namespace std;
using namespace cv;
using namespace DBoW2;
// using namespace ORB_SLAM2;
const int NIMAGES = 1101;
Ptr<ORB> detector = ORB::create();


void loadFeatures(vector<vector<cv::Mat>> &features, const char *path_to_image);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat>> &features);
void testDatabase(const vector<vector<cv::Mat>> &features);



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

void testDatabase(const vector<vector<cv::Mat>> &features, OrbDatabase &db, bool &Isloopdetected, int &keyframe_prev_id, int &keyframe_curr_id, double thre_score)
{
  //cout << "Creating a small database..." << endl;

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
  //cout << "... done!" << endl;

  //cout << "Database information: " << endl
       //<< db << endl;

  // and query the database
  //cout << "Querying the database: " << endl;

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

        if ( (score > thre_score)&&(i==features.size()-1))
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
