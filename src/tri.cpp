
//0514
#include "tripkg/vo.h"


#define MAX_FRAME 5000
#define MIN_NUM_FEAT 2000
#define MAX_CORNERS 1500
#define local_ba_frame 12
#define reprojectionError 3
#define max_feature_number 300
#define local_ba 1
#define loopclosing 1
#define inlier_ratio_def 0.7
#define min_keyframe 5
#define MAX_IMAGE_NUMBER 1101
#define parallax_def 10
#define max_distance 150
const double focal = 718.8560; //00-02
    const cv::Point2d pp(607.1928, 185.2157);
    // const double focal = 721.5377; //03
    // const cv::Point2d pp(609.5593, 172.854);
    // const double focal = 707.0912; //04-12
    // const cv::Point2d pp(601.8873, 183.1104);
const char* path_to_image = "/home/gleefe/Downloads/dataset/sequences/00/image_0/%06d.png";
string path_to_pose = "/home/gleefe/Downloads/dataset/poses/00.txt";


using namespace std;
using namespace cv;
using namespace DBoW2;
using namespace g2o;

extern "C" void G2O_FACTORY_EXPORT g2o_type_VertexSE3(void);

bool compare_point (pair<int,pair<int,Point3d>> a,
                    pair<int,pair<int,Point3d>> b){
                      if(a.first==b.first){
                        return a.second.first<b.second.first;
                      }
                      
                        return a.first<b.first;
                      
                    }

bool compare_point2 (pair<int,Point3d> a,
                    pair<int,Point3d> b){
                      
                        return a.first<b.first;
                      
                    }



int main(int argc, char **argv){
    g2o_type_VertexSE3();
    google::InitGoogleLogging(argv[0]);
    Ptr<Feature2D> detector = ORB::create();
    //ros
    ros::init(argc, argv, "tri");
    ros::NodeHandle n;
    ros::Publisher world_points_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("world_points", 1);
    ros::Publisher traj_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("traj", 1);
    ros::Publisher gt_traj_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("gt_traj", 1);
    ros::Publisher tracking_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("tracking", 1);
    ros::Publisher keyframe_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("keyframe", 1);
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    
    //image
    image_transport::ImageTransport it(n);
    image_transport::Publisher image_pub = it.advertise("camera/image", 1);
    
    ros::Rate loop_rate(5);
    
   //uint32_t shape = visualization_msgs::Marker::ARROW;
   
   

  Mat R_f, t_f; //the final rotation and tranlation vectors containing the 
  int error=0;

  ofstream myfile;
  myfile.open ("results1_1.txt");

  double scale = 1.00;
  char filename1[200];
  char filename2[200];
  
  sprintf(filename1, path_to_image, 0);
  sprintf(filename2, path_to_image, 1);

  char text[100];
  char error_text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;  
  Point textOrg(10,50);
  Point textOrg2(460, 490);

  //read the first two frames from the dataset
  Mat image1_c = imread(filename1);
  Mat image2_c = imread(filename2);

  if ( !image1_c.data || !image2_c.data ) { 
    cout<< " --(!) Error reading images " << std::endl; return -1;
  }
    
    int feature_number=0;
    
     
  
    Mat image1;
    Mat image2;

    cvtColor(image1_c, image1, COLOR_BGR2GRAY);
    cvtColor(image2_c, image2, COLOR_BGR2GRAY);

    vector<Point2f> points1;
    vector<Point2f> points2;
    vector<KeyPoint> key_points1;
    vector<KeyPoint> key_points2;
    vector<Point2f> points2_tmp;
    vector<pair<int,pair<int,Point2f>>> points1_map;
    vector<pair<int,pair<int,Point2f>>> points2_map;
    int keyframe_num=0;
    featureDetection(image1, points1, points1_map,keyframe_num,MAX_CORNERS);
    // featureDetection_esential(prevImage, prevFeatures,prev_points_map,keyframe_num);  
    
    vector<uchar> status;
    Mat tri_prevImage = image1.clone();
    vector<Point2f> tri_prevFeatures;
    vector<pair<int,pair<int,Point2f>>> tri_prev_points_map = points1_map;
    featureTracking(image1,image2,points1,points2,points1_map,points2_map, status,points2_tmp);
    
    erase_int_point2f(image1,points2_tmp,tri_prev_points_map,status);
    
    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    
    
    Mat K = (Mat_<float>(3,3)<< focal, 0, pp.x,
                                                     0, focal, pp.y,
                                                        0,  0,   1);
    
    
    

    Mat E, R, t, mask;
    int index;
    E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);


    scale = getAbsoluteScale(1, 0,path_to_pose);
    get_gt(1, 0,path_to_pose);

    //t=scale*t;

     Mat prevImage = image2;
     Mat currImage;
     vector<Point2f> prevFeatures = points2;
     vector<Point2f> currFeatures;
     vector<pair<int,pair<int,Point2f>>> prev_points_map = points2_map;
     vector<pair<int,pair<int,Point2f>>> curr_points_map;
     vector<KeyPoint> prev_keypoints = key_points2;
     vector<KeyPoint> curr_keypoints;
      char filename[100];

    R_f = R.clone();
    t_f = t.clone();
    
      clock_t begin = clock();

      //namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
     // namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(1000, 1200, CV_8UC3);
  
  
  Mat Rt0 = Mat::eye(3, 4, CV_64FC1);
  Mat Rt1 = Mat::eye(3, 4, CV_64FC1);
 
  Mat Kd;
  K.convertTo(Kd, CV_64F);




pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg(new pcl::PointCloud<pcl::PointXYZRGB>);
    msg->header.frame_id = "map";
    msg->height = cloud->height;
    msg->width = cloud->width;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg2(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointXYZRGB point;
      pcl::PointXYZRGB trajectory;
      
      msg2->header.frame_id = "map";
      msg2->height = cloud2->height;
      msg2->width = cloud2->width;

 pcl::PointCloud<pcl::PointXYZRGB>::Ptr gt_msg(new pcl::PointCloud<pcl::PointXYZRGB>);
gt_msg->header.frame_id = "map";
      gt_msg->height = cloud2->height;
      gt_msg->width = cloud2->width;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3(new pcl::PointCloud<pcl::PointXYZRGB>);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg4(new pcl::PointCloud<pcl::PointXYZRGB>);
msg4->header.frame_id = "map";
      msg4->height = cloud2->height;
      msg4->width = cloud2->width;

pcl::PointXYZRGB tracking_points;

Mat point3d_homo;
      Mat _p3h;
      Mat three_to_p;
      Mat p3d2;
      Mat p3d22;

Mat R_f2, t_f2;     
      Mat R_f1t;
      Mat t_f1;
      Mat R_f2t; 
vector<pair<int,pair<int,Point3d>>> point_3d_map;
      
sensor_msgs::ImagePtr image_msg;



    Mat rvec,tvec;
    // Eigen::Matrix<double,3,1> rvec;
    // Eigen::Matrix<double,3,1> tvec;

    Mat R_solve;
    Mat R_solve_inv;
    Mat t_solve_inv;
    Mat R_solve_prev;
    Mat t_solve_prev;
    Mat t_solve_f;
    R = R.t();
    


    
int init_check=0;
int numFrame=2;
int number_frame=0;

      Mat R_tri,t_tri;


      vector<Point2f> new_prevFeatures;
    vector<pair<int,pair<int,Point2f>>> new_prev_points_map;
    vector<Point2f> new_currFeatures;
    vector<pair<int,pair<int,Point2f>>> new_curr_points_map;


    Mat new_tri_prevImage;
    vector<Point2f> new_tri_prevFeatures;
    vector<pair<int,pair<int,Point2f>>> new_tri_prev_points_map;
    vector<Point2f> currFeatures2;
    double prev_rot_ang=0;
int tri_numFrame;
    Mat local_rvec;
    Mat local_tvec;
    Mat local_image;
    vector<int> keyframe_vec;
    vector<Point3d> rvec_vec;
    vector<Point3d> tvec_vec;

    vector<Point3d> rvec_vec_loop;
    vector<Point3d> tvec_vec_loop;
    
vector<pair<int,pair<int,Point3d>>> BA_3d_points_map;
vector<pair<int,pair<int,Point2f>>> BA_2d_points_map;

vector<pair<int,pair<int,Point3d>>> BA_3d_points_map_loop;
vector<pair<int,pair<int,Point2f>>> BA_2d_points_map_loop;

vector<int> number_of_3d_points;
vector<int> number_of_3d_points_loop;


vector<int> BA_3d_map_points;

double tracking_number_last=0;
double tracking_number_current=0;

  Mat t_solve_f_prev;
  int numFrame_prev=0;
  vector<pair<int,pair<int,Point3d>>> BA_3d_points_map_tmp;

  vector<vector<cv::Mat > > features;
  features.reserve(MAX_IMAGE_NUMBER);

  OrbVocabulary voc("0630_KITTI00-22_10_4_voc.yml.gz");
  //OrbVocabulary voc("kitti00_10^4_voc.yml.gz");
  OrbDatabase db(voc, false, 0); // false = do not use direct index

  bool Isloopdetected=0;

  int keyframe_prev;
  int keyframe_curr;
 
  vector<Point3d> t_solve_f_vec;
  vector<Mat> R_solve_inv_vec;

  

  vector<int> numFrame_vec;

  cloud2->points.resize(MAX_FRAME);
    trajectory = cloud2->points[0];
    trajectory.x=0;
    trajectory.y=0;
    trajectory.z=0;
    trajectory.r=0;
    trajectory.g=255;
    trajectory.b=0;
msg2->points.push_back(trajectory);
traj_pub.publish(msg2);
  vector<Eigen::Quaterniond> quat_vec;

  int keyframe_number=0;

  int once_loop_detected=0;

  vector<KeyPoint> keypoints_tmp;
      
      Mat descriptor_tmp;
      // KeyPoint::convert( prevFeatures, keypoints_1);
      //detector->compute(currImage,keypoints_1,descriptor_1);
      Mat mask_tmp;
      detector->detectAndCompute(prevImage, mask_tmp, keypoints_tmp, descriptor_tmp);
      
       features.push_back(vector<cv::Mat >());
      changeStructure(descriptor_tmp, features.back());
      
      QueryResults ret;
      testDatabase(features,db,Isloopdetected,keyframe_prev,keyframe_curr);

t_solve_f_vec.push_back(Point3d(0,0,0));
      
    
    
      numFrame_vec.push_back(0);
      Eigen::Matrix3d mat_eig;
      Mat mat_identity=Mat::eye(Size(3,3),CV_64F);
      for (int i=0;i<3;i++){
            for(int j=0;j<3;j++){
              mat_eig(i,j)=mat_identity.at<double>(i,j);
            }
          }
      Eigen::Quaterniond quat(mat_eig);
      quat_vec.push_back(quat);

      // for (int i=0;i<prev_points_map.size();i++){
      // // BA_3d_points_map_loop.push_back(point_3d_map[i]);
      // //BA_2d_points_map_loop.push_back(prev_points_map[i]);
      // // number_of_3d_points_loop.push_back(prev_points_map.size());
      // }
      Mat rvec_tmp1;
      Rodrigues(mat_identity,rvec_tmp1);
      rvec_vec_loop.push_back(Point3d(rvec_tmp1.at<double>(0),rvec_tmp1.at<double>(1),rvec_tmp1.at<double>(2)));
      tvec_vec_loop.push_back(Point3d(0,0,0));

      int prev_traj_num=0;

      vector<int> loop_keyframe_num;

      vector<Eigen::Vector3d> trans_vec_loop;
      vector<Eigen::Quaterniond> quat_vec_loop;
// -------------------------------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------------------------------


    
      while(ros::ok){
        while (init_check==0){
      sprintf(filename1, path_to_image, numFrame);

      Mat currImage_c = imread(filename1);
      cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	  vector<uchar> status;
       
      
      float prev=prevFeatures.size();
      featureTracking(prevImage, currImage, prevFeatures, currFeatures,prev_points_map,curr_points_map,status,points2_tmp);
      erase_int_point2f(currImage,points2_tmp,tri_prev_points_map,status);
      float curr=currFeatures.size();
      float tracking_ratio=curr/prev;
      //cout<<"tracking_ratio: "<<tracking_ratio<<"\n";

      for (int i=0;i<prevFeatures.size();i++){
        int m=prevFeatures[i].x;
        int n=prevFeatures[i].y;
        int c=currFeatures[i].x;
        int d=currFeatures[i].y;
        circle(currImage_c, Point(m,n),2,CV_RGB(0,255,0),2);
        circle(currImage_c, Point(c,d),2,CV_RGB(0,0,255),2);
      }
      //imshow( "Road facing camera", currImage_c );
      
      
      
      E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
      float inlier_number=0;
      for (int i=0;i<currFeatures.size();i++){
        if (mask.at<bool>(i)==1) inlier_number++;
      }
      float e_inlier_ratio=inlier_number/curr;
      //cout<<e_inlier_ratio<<"\n";
      //waitKey();
      recoverPose(E, currFeatures,prevFeatures , R, t, focal, pp, mask);

      scale = getAbsoluteScale(numFrame, 0,path_to_pose);
      get_gt(numFrame, 0,path_to_pose);


      if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
      
       R_f2 = R_f;
        R_f2t = R_f2.t();
        t_f2 = -R_f2t*t_f ;
        

        //t_f = t_f + scale*(R_f*t);
        t_f = t_f + (R_f*t);
        R_f = R*R_f;
        
        
        R_f1t = R_f.t();
        t_f1 = -R_f1t*t_f;
      
      }
      
      else {
      //cout << "scale below 0.1, or incorrect translation" << endl;
      }

      numFrame++;

    

    int x = int(t_f.at<double>(0)) + 600;
    int y = -int(t_f.at<double>(2)) + 800;

    int c = int(gt_x) + 600;
    int d = -int(gt_z) +800;
      

    circle(traj, Point(c,d),1,CV_RGB(0,0,255),2);

    circle(traj,Point(x,y),1,CV_RGB(0,255,0),2);

    rectangle( traj, Point(10, 30), Point(570, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    

    //imshow( "Road facing camera", currImage_c );
    //imshow( "Trajectory", traj );
  
         
    

      
    //msg2->points.push_back(trajectory);
    

    
    trajectory.x=gt_x;
    trajectory.y=gt_y;
    trajectory.z=gt_z;
    trajectory.r=0;
    trajectory.g=0;
    trajectory.b=255.0f;
    gt_msg->points.push_back(trajectory);

    gt_traj_pub.publish(gt_msg);
    
    image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", currImage_c).toImageMsg();
    image_pub.publish(image_msg);

//        marker.header.frame_id = "map";
// marker.header.stamp = ros::Time();
// marker.ns = "my_namespace";
// marker.id = 0;
// marker.type = visualization_msgs::Marker::SPHERE;
// marker.action = visualization_msgs::Marker::ADD;
// marker.pose.position.x = gt_x;
// marker.pose.position.y = gt_y;
// marker.pose.position.z = gt_z;
// marker.pose.orientation.x = 0.0;
// marker.pose.orientation.y = 0.0;
// marker.pose.orientation.z = 0.0;
// marker.pose.orientation.w = 1.0;
// marker.scale.x = 2;
// marker.scale.y = 2;
// marker.scale.z = 2;
// marker.color.a = 1.0; // Don't forget to set the alpha!
// marker.color.r = 0.0;
// marker.color.g = 255.0;
// marker.color.b = 0.0;
// //only if using a MESH_RESOURCE marker type:

// marker_pub.publish( marker );
// waitKey();

    waitKey(1);
    




if (e_inlier_ratio<0.5){
  featureDetection(prevImage, prevFeatures, prev_points_map,keyframe_num,MAX_CORNERS);
     
    tri_prev_points_map = prev_points_map;
    featureTracking(prevImage, currImage, prevFeatures, currFeatures,prev_points_map,curr_points_map, status,points2_tmp);
    erase_int_point2f(prevImage,points2_tmp,tri_prev_points_map,status);
    
    R_f2t.copyTo(Rt0.rowRange(0,3).colRange(0,3));
    t_f2.copyTo(Rt0.rowRange(0,3).col(3));
      
}



if (prevFeatures.size() < 200)	{
      //cout<<"make 3d point!!"<<"\n";


      
      
      R_f1t.copyTo(Rt1.rowRange(0,3).colRange(0,3));
      t_f1.copyTo(Rt1.rowRange(0,3).col(3));
      
      //tri_prevFeatures.clear();
      // currFeatures2.clear();
      
      //for (int i=0;i<tri_prev_points_map.size();i++){
        
        tri_prevFeatures.clear();
        for (int i=0;i<tri_prev_points_map.size();i++){
            // int j=curr_points_map[i].second.first;
            
            tri_prevFeatures.push_back(tri_prev_points_map[i].second.second);
            
            //currFeatures2.push_back(curr_points_map[j].second);
          }
      
      
      
      triangulatePoints(Kd*Rt0,Kd*Rt1,tri_prevFeatures,currFeatures,point3d_homo);
      
      int currFeatures_size=currFeatures.size();

      cloud->points.resize (point3d_homo.cols);
      
      point_3d_map.clear();
      vector <pair<int,pair<int,Point3d>>>().swap(point_3d_map);

    for(int i = 0; i < point3d_homo.cols; i++) {
      
      int m = currFeatures.at(i).x;
      int n = currFeatures.at(i).y;
      circle(currImage_c, Point(m, n) ,1, CV_RGB(255,0,0), 2);

      point = cloud->points[i];
      _p3h = point3d_homo.col(i); 
      p3d2 = _p3h/_p3h.at<float>(3);
      p3d2.convertTo(p3d22, CV_64F);
      
      point.x = p3d22.at<double>(0);
      point.y = p3d22.at<double>(1);
      point.z = p3d22.at<double>(2);
    
      point.r=100;
      point.g=100;
      point.b=100;

      three_to_p=Kd*Rt1*p3d22;
      int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
      int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
      circle(currImage_c, Point(c,d),1,CV_RGB(0,255,0),1);
      
      int point_diff_x = (m-c)*(m-c);
      int point_diff_y = (n-d)*(n-d);

      float parallax2=0;
      float x_para2=currFeatures[i].x-tri_prevFeatures[i].x;
      float y_para2=currFeatures[i].y-tri_prevFeatures[i].y;

      parallax2=sqrt(x_para2*x_para2+y_para2*y_para2);
//(sqrt( pow(point.x-t_f.at<double>(0),2)+pow(point.y-t_f.at<double>(1),2)+pow(point.z-t_f.at<double>(2),2) ) < 30)
//&&(sqrt(point_diff_x+point_diff_y)<10)

      if (  //(sqrt( pow(point.x-t_f.at<double>(0),2)+pow(point.y-t_f.at<double>(1),2)+pow(point.z-t_f.at<double>(2),2) ) < 150)
          //&& (sqrt( pow(point.x-t_f.at<double>(0),2)+pow(point.y-t_f.at<double>(1),2)+pow(point.z-t_f.at<double>(2),2) ) > 1)
          (c>0)&&(d>0)&&(c<currImage.cols)&&(d<currImage.rows)
          &&(sqrt(point_diff_x+point_diff_y)<reprojectionError)
          &&(parallax2>parallax_def)
          )
        {

        msg->points.push_back(point);
        point_3d_map.push_back(make_pair(curr_points_map[i].first,make_pair(curr_points_map[i].second.first,Point3d(point.x,point.y,point.z))));
        currFeatures.push_back(currFeatures[i]);
        curr_points_map.push_back(curr_points_map[i]);
        tri_prev_points_map.push_back(tri_prev_points_map[i]);
        //curr_keypoints.push_back(curr_keypoints[i]);
      }
      // msg->points.push_back(point);
      // point_3d_map.push_back(make_pair(curr_points_map[i].first,Point3d(point.x,point.y,point.z)));
       
  }
  currFeatures.erase(currFeatures.begin(),currFeatures.begin()+currFeatures_size);
  curr_points_map.erase(curr_points_map.begin(),curr_points_map.begin()+currFeatures_size);
  tri_prev_points_map.erase(tri_prev_points_map.begin(),tri_prev_points_map.begin()+currFeatures_size);
  //curr_keypoints.erase(curr_keypoints.begin(),curr_keypoints.begin()+currFeatures_size);
    // imshow( "Road facing camera", currImage_c );
    // waitKey();
  
    // imshow( "Road facing camera", currImage_c );
    //   waitKey();
    // cout<<currFeatures.size()<<"\n";
    // waitKey();
    if (currFeatures.size()>0.1*200){
      keyframe_number++;

      for (int j=0;j<2;j++){
      for (int i=0;i<point_3d_map.size();i++){
      BA_3d_points_map_loop.push_back(point_3d_map[i]);
      if (j==1){
        BA_3d_points_map.push_back(point_3d_map[i]);
      }
      }
      number_of_3d_points_loop.push_back(point_3d_map.size());
      }
      number_of_3d_points.push_back(point_3d_map.size());

      for (int i=0;i<point_3d_map.size();i++){
        BA_2d_points_map_loop.push_back(tri_prev_points_map[i]);
      }
      for (int i=0;i<point_3d_map.size();i++){
        BA_2d_points_map_loop.push_back(curr_points_map[i]);
        BA_2d_points_map.push_back(curr_points_map[i]);
      }
      // cout<<BA_3d_points_map_loop.size()<<"\n";
      // cout<<BA_2d_points_map_loop.size()<<"\n";
      // for (int i=0;i<BA_3d_points_map_loop.size();i++){
      //   cout<<BA_3d_points_map_loop[i].first<<" "<<BA_3d_points_map_loop[i].second.first<<"\n";
      //   cout<<BA_2d_points_map_loop[i].first<<" "<<BA_2d_points_map_loop[i].second.first<<"\n";
      // }
      // waitKey();

      Rodrigues(R_f1t,rvec);
      rvec_vec_loop.push_back(Point3d(rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2)));
      tvec_vec_loop.push_back(Point3d(t_f1.at<double>(0),t_f1.at<double>(1),t_f1.at<double>(2)));
      rvec_vec.push_back(Point3d(rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2)));
      tvec_vec.push_back(Point3d(t_f1.at<double>(0),t_f1.at<double>(1),t_f1.at<double>(2)));

      trajectory.x=t_f.at<double>(0);
      trajectory.y=t_f.at<double>(1);
      trajectory.z=t_f.at<double>(2);
      trajectory.r=0;
      trajectory.g=255;
      trajectory.b=0;
      msg2->points.push_back(trajectory);

      vector<KeyPoint> keypoints_tmp;
      
      Mat descriptor_tmp;
      // KeyPoint::convert( prevFeatures, keypoints_1);
      //detector->compute(currImage,keypoints_1,descriptor_1);
      Mat mask_tmp;
      detector->detectAndCompute(currImage, mask_tmp, keypoints_tmp, descriptor_tmp);
      
       features.push_back(vector<cv::Mat >());
      changeStructure(descriptor_tmp, features.back());
      
      QueryResults ret;
      testDatabase(features,db,Isloopdetected,keyframe_prev,keyframe_curr);

t_solve_f_vec.push_back(Point3d(t_f.at<double>(0),t_f.at<double>(1),t_f.at<double>(2)));
      
      numFrame_vec.push_back(numFrame);
      Eigen::Matrix3d mat_eig;
      // Mat mat_identity=Mat::eye(Size(3,3),CV_64F);
      for (int i=0;i<3;i++){
            for(int j=0;j<3;j++){
              mat_eig(i,j)=R_f.at<double>(i,j);
            }
          }
      Eigen::Quaterniond quat(mat_eig);
      quat_vec.push_back(quat);




    featureDetection(prevImage, new_prevFeatures,new_prev_points_map,keyframe_num,MAX_CORNERS);
    
    
    new_tri_prev_points_map=new_prev_points_map;
    featureTracking(prevImage, currImage, new_prevFeatures, new_currFeatures,new_prev_points_map,new_curr_points_map, status,points2_tmp);
    erase_int_point2f(prevImage,points2_tmp,new_tri_prev_points_map,status);
    tracking_number_last=new_tri_prev_points_map.size();
      R_tri = R_f2t.clone();
      t_tri = t_f2.clone();
      
      Rodrigues(R_f1t,local_rvec);

      local_tvec=t_f1.clone();
      // numFrame_vec.push_back(numFrame);
      // rvec_vec.push_back(Point3d(local_rvec.at<double>(0),local_rvec.at<double>(1),local_rvec.at<double>(2)));
      // tvec_vec.push_back(Point3d(local_tvec.at<double>(0),local_tvec.at<double>(1),local_tvec.at<double>(2)));
      

      
      R_solve_prev=R_f1t.clone();
      t_solve_prev=t_f1.clone();
       
      prevImage = currImage.clone();
      prevFeatures = currFeatures;
      prev_points_map = curr_points_map;

      Mat R_f_inv=R_f.t();
      R_solve_inv = R_f;
      t_solve_f = t_f.clone();
      t_solve_f_prev=t_f.clone();
      

      new_prevFeatures = new_currFeatures;
      new_prev_points_map = new_curr_points_map;

      number_frame=numFrame;
      numFrame_prev=numFrame;
      tri_numFrame=numFrame;
      







      world_points_pub.publish(msg);
      traj_pub.publish(msg2);
      
      
      
      
      double diagonal = R_f_inv.at<double>(0,0)+R_f_inv.at<double>(1,1)+R_f_inv.at<double>(2,2);

    double rot_ang = acos( (diagonal-1.0)/2);
    prev_rot_ang = rot_ang*(180/CV_PI);
      init_check=1;

    }
    else{
      // vector<Point2f> prevFeatures_tmp=prevFeatures;
      // vector<pair<int,pair<int,Point2f>>> prev_points_map_tmp=prev_points_map;
      // featureDetection_esential(prevImage, prevFeatures,prev_points_map,keyframe_num);
      featureDetection(prevImage, prevFeatures, prev_points_map,keyframe_num,MAX_CORNERS);
      // for (int i=0;i<prevFeatures.size();i++){
      //   prevFeatures_tmp.push_back(prevFeatures[i]);
      //   prev_points_map_tmp.push_back(prev_points_map[i]);
      // }

    tri_prev_points_map = prev_points_map;
    featureTracking(prevImage, currImage, prevFeatures, currFeatures,prev_points_map,curr_points_map, status,points2_tmp);
    erase_int_point2f(prevImage,points2_tmp,tri_prev_points_map,status);
    
    
    R_f2t.copyTo(Rt0.rowRange(0,3).colRange(0,3));
    t_f2.copyTo(Rt0.rowRange(0,3).col(3));
      prevImage = currImage.clone();
    prevFeatures = currFeatures;
    prev_points_map = curr_points_map;
    }
    
    }
    else{
        prevImage = currImage.clone();
    prevFeatures = currFeatures;
    prev_points_map = curr_points_map;
    }

    
    

      }
    



//******************************************************************************************pnp start**********************************************************************************************************************************//

 for(numFrame=number_frame+1; numFrame < MAX_FRAME; numFrame++)	{
    
    //cout<<"frame start"<<"\n";
    
   // Mat test = Mat::zeros(currImage.rows,currImage.cols, CV_8UC3);

  	sprintf(filename, path_to_image, numFrame);
    
  	Mat currImage_c = imread(filename);
  	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	vector<uchar> status;
    
  
    featureTracking(prevImage, currImage, prevFeatures, currFeatures,prev_points_map,curr_points_map, status,points2_tmp);
    
    int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points2_tmp.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  
     		  points2_tmp.erase (points2_tmp.begin() + (i - indexCorrection));
          
          point_3d_map.erase(point_3d_map.begin() + (i - indexCorrection));
          
          
     		  indexCorrection++;
     	}

     }
     
    //cout<<"point_3d_map size: "<<point_3d_map.size()<<"\n";

    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg3(new pcl::PointCloud<pcl::PointXYZRGB>);
      msg3->header.frame_id = "map";
      msg3->height = cloud3->height;
      msg3->width = cloud3->width;
      
    cloud3->points.resize(BA_3d_points_map_tmp.size()+point_3d_map.size());
      for (int i=0;i<BA_3d_points_map_tmp.size();i++){
      tracking_points = cloud3->points[i];
      tracking_points.x=BA_3d_points_map_tmp[i].second.second.x;
    tracking_points.y=BA_3d_points_map_tmp[i].second.second.y;
    tracking_points.z=BA_3d_points_map_tmp[i].second.second.z;
    tracking_points.r=250;
    tracking_points.g=250;
    tracking_points.b=210;
    msg3->points.push_back(tracking_points);
      }
    for (int i=0;i<point_3d_map.size();i++){
      tracking_points = cloud3->points[i];
      tracking_points.x=point_3d_map[i].second.second.x;
    tracking_points.y=point_3d_map[i].second.second.y;
    tracking_points.z=point_3d_map[i].second.second.z;
    tracking_points.r=255;
    tracking_points.g=0;
    tracking_points.b=0;
    msg3->points.push_back(tracking_points);
      }
      


    
    // cout<<points2_tmp.size()<<"\n";
    // waitKey();

    vector<uchar> status1;
    featureTracking(prevImage, currImage, new_prevFeatures, new_currFeatures,new_prev_points_map,new_curr_points_map, status1,points2_tmp);
    erase_int_point2f(prevImage,points2_tmp,new_tri_prev_points_map,status1);
    
    float parallax=0;

    for (int i=0;i<new_tri_prev_points_map.size();i++){
      float x_para=new_currFeatures[i].x-new_tri_prev_points_map[i].second.second.x;
      float y_para=new_currFeatures[i].y-new_tri_prev_points_map[i].second.second.y;
      //cout<<sqrt(x_para*x_para+y_para*y_para)<<" ";
      parallax+=sqrt(x_para*x_para+y_para*y_para);
    }
    
    parallax/=new_currFeatures.size();
    // cout<<parallax<<"\n";
    // waitKey();
    tracking_number_current=new_currFeatures.size();
    
    
    
    
    vector<Point3d> corr_3d_point;
    vector<Point2f> corr_2d_point;
    Mat Relative_homo_R = Mat::eye(3,4,CV_64FC1);
    
    for (int i=0;i<point_3d_map.size();i++){
      
        corr_3d_point.push_back(point_3d_map[i].second.second);
        corr_2d_point.push_back(curr_points_map[i].second.second);
        
       }
      
    
    

    int corr_3d_point_number=corr_3d_point.size();

    double point_3d_map_number=point_3d_map.size();

    
    //double corr_point_ratio = corr_3d_point_number/point_3d_map_number;
    
    
    vector<Point2d> corr_2d_pointd(corr_2d_point.begin(),corr_2d_point.end());
    
    
    Mat array;
    
    
    
    // Eigen::Matrix<double,1,3> rvec_eig;
    // Eigen::Matrix<double,1,3> tvec_eig;
    // Eigen::Matrix<double,1,4> corr_3d_point_eig;
    //cout<<"corr_3d_point: "<<corr_3d_point.size()<<"\n";
    

    solvePnPRansac(corr_3d_point,corr_2d_pointd,Kd,noArray(),rvec,tvec,false,100,3.0F,0.99,array,SOLVEPNP_P3P);
    // cout<<rvec<<"\n";
    // cout<<tvec<<"\n";
    float inlier_ratio=float(array.rows)/float(corr_2d_pointd.size());
    int inlier_number = array.rows;
    //cout<<"inlier_ratio: "<<inlier_ratio<<"\n"; 
    //cout<<"inlier number: "<<array.rows<<"\n";
    //cout<<"inlier_ratio: "<<inlier_ratio<<"\n";
    //cout<<corr_2d_pointd.size()<<"\n";
    
    int erase_number=0;
    
    
    for (int i=0;i<array.rows;i++){
      corr_3d_point.push_back(corr_3d_point[array.at<int>(i)]);
      corr_2d_pointd.push_back(corr_2d_pointd[array.at<int>(i)]);
      
    }
    

    corr_3d_point.erase(corr_3d_point.begin(),corr_3d_point.begin()+corr_3d_point_number);
    corr_2d_pointd.erase(corr_2d_pointd.begin(),corr_2d_pointd.begin()+corr_3d_point_number);
    
    
    //cout<<array.rows<<"\n";
    //cout<<corr_3d_point.size()<<"\n";

    Rodrigues(rvec,R_solve);
    R_solve.copyTo(Relative_homo_R.rowRange(0,3).colRange(0,3));
    tvec.copyTo(Relative_homo_R.rowRange(0,3).col(3));
    

  for(int i = 0; i < corr_3d_point.size(); i++) {
      
      int m = corr_2d_pointd.at(i).x;
      int n = corr_2d_pointd.at(i).y;
      circle(currImage_c, Point(m, n) ,2, CV_RGB(0,0,255), 2);
      
      int corr_3d_point_int = corr_3d_point.size();
      Mat prev_p3hh(4,corr_3d_point_int,CV_64F);
      // Mat prev_p3d2;
      // Mat prev_p3d22;
      Mat prev_p3h;
      prev_p3h = prev_p3hh.col(i); 
      // prev_p3d2 = prev_p3h/prev_p3h.at<float>(3);
      // prev_p3d2.convertTo(prev_p3d22, CV_64F);
      prev_p3h.at<double>(0)=corr_3d_point.at(i).x;
      prev_p3h.at<double>(1)=corr_3d_point.at(i).y;
      prev_p3h.at<double>(2)=corr_3d_point.at(i).z;
      prev_p3h.at<double>(3)=1;
      
        three_to_p=Kd*Relative_homo_R*prev_p3h;
        int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
        int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
        circle(currImage_c, Point(c, d) ,2, CV_RGB(0,255,0), 2);

        //cout<<"x,y diff :"<<m-c<<","<<n-d<<"\n";
  }
    //cout<<point_3d_map.size()<<"\n";
//imshow( "Road facing camera", currImage_c );
   
//****************************************************************************************************motion only BA************************************************************************************************//
    //cout<<"motion only BA start!!"<<"\n";
    Eigen::Vector3d rvec_eig;
    Eigen::Vector3d tvec_eig;
    rvec_eig[0]=rvec.at<double>(0);
    rvec_eig[1]=rvec.at<double>(1);
    rvec_eig[2]=rvec.at<double>(2);
    
    tvec_eig[0]=tvec.at<double>(0);
    tvec_eig[1]=tvec.at<double>(1);
    tvec_eig[2]=tvec.at<double>(2);
    
    Eigen::MatrixXd corr_2d_point_eig(2,corr_2d_pointd.size());

    for (int i=0;i<corr_2d_pointd.size();i++){
      corr_2d_point_eig(0,i)=corr_2d_pointd[i].x;
      corr_2d_point_eig(1,i)=corr_2d_pointd[i].y;
    }
    

    Eigen::MatrixXd corr_3d_point_eig(4,corr_3d_point.size());

    for (int i=0;i<corr_3d_point.size();i++){
      corr_3d_point_eig(0,i)=corr_3d_point[i].x;
      corr_3d_point_eig(1,i)=corr_3d_point[i].y;
      corr_3d_point_eig(2,i)=corr_3d_point[i].z;
      corr_3d_point_eig(3,i)=1;
    }
    
    ceres::Problem problem;

    for (int i = 0; i < corr_3d_point.size(); i++) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    
    ceres::CostFunction* cost_function = 
          new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3,3>(
            new SnavelyReprojectionError(corr_2d_point_eig(0,i), corr_2d_point_eig(1,i),corr_3d_point_eig.col(i),focal,pp.x,pp.y)
            );

    problem.AddResidualBlock(cost_function,
                             NULL ,
                             rvec_eig.data(),
                             tvec_eig.data());

  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 12;
  options.max_num_iterations=10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.FullReport() << "\n";

 

   for (int i=0;i<3;i++){
      rvec.at<double>(i)=double(rvec_eig[i]);
      tvec.at<double>(i)=double(tvec_eig[i]);
    }

   //cout<<"motion only BA end!!"<<"\n";
    //***********************************************************************************motion only BA end********************************************************************************//
   

    Rodrigues(rvec,R_solve);
    
    double diagonal = R_solve.at<double>(0,0)+R_solve.at<double>(1,1)+R_solve.at<double>(2,2);

    double rot_ang = acos( (diagonal-1.0)/2);
    rot_ang = rot_ang*(180/CV_PI);
    //cout<<rot_ang<<"\n";
    double rot_ang_diff = abs(rot_ang-prev_rot_ang);
    //cout<<rot_ang_diff<<"\n";
    //waitKey();

    R_solve_inv = R_solve.t();
    t_solve_f = -R_solve_inv*tvec;
    
    R_solve.copyTo(Relative_homo_R.rowRange(0,3).colRange(0,3));
    tvec.copyTo(Relative_homo_R.rowRange(0,3).col(3));

    //diff=0;

    currImage_c = imread(filename);
    //cout<<corr_3d_point.size()<<"\n";
    for(int i = 0; i < corr_3d_point.size(); i++) {
      
      int m = corr_2d_pointd.at(i).x;
      int n = corr_2d_pointd.at(i).y;
      circle(currImage_c, Point(m, n) ,2, CV_RGB(0,0,255), 2);
      
      int corr_3d_point_int = corr_3d_point.size();
      Mat prev_p3hh(4,corr_3d_point_int,CV_64F);
      // Mat prev_p3d2;
      // Mat prev_p3d22;
      Mat prev_p3h;
      prev_p3h = prev_p3hh.col(i); 
      // prev_p3d2 = prev_p3h/prev_p3h.at<float>(3);
      // prev_p3d2.convertTo(prev_p3d22, CV_64F);
      prev_p3h.at<double>(0)=corr_3d_point.at(i).x;
      prev_p3h.at<double>(1)=corr_3d_point.at(i).y;
      prev_p3h.at<double>(2)=corr_3d_point.at(i).z;
      prev_p3h.at<double>(3)=1;
      
        three_to_p=Kd*Relative_homo_R*prev_p3h;
        int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
        int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
        circle(currImage_c, Point(c, d) ,2, CV_RGB(0,255,0), 2);
      
      int point_diff_x = (m-c)*(m-c);
      int point_diff_y = (n-d)*(n-d);
      
      
      //diff += (sqrt(point_diff_x+point_diff_y));
      
  }
  
  image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", currImage_c).toImageMsg();
    image_pub.publish(image_msg);
    
 
  
    
    get_gt(numFrame, 0,path_to_pose);
      
     double corr_2d_pointd_number=corr_2d_pointd.size();
     double tracking_ratio=tracking_number_current/tracking_number_last;
        
    double t_x=t_solve_f.at<double>(0)-t_solve_f_prev.at<double>(0);
    double t_y=t_solve_f.at<double>(1)-t_solve_f_prev.at<double>(1);
    double t_z=t_solve_f.at<double>(2)-t_solve_f_prev.at<double>(2);

    double distance_prev_keyframe=sqrt(t_x*t_x+t_y*t_y+t_z*t_z);
    
    //cout<<"traveled distance prev keyframe: "<<distance_prev_keyframe<<" m"<<"\n";
    
//(inlier_ratio<0.6)||(tracking_ratio<0.3)
    
//(corr_2d_pointd_number<50)||(new_prevFeatures.size()<200)
//(tracking_ratio<0.85)&&(parallax>=15)
//&&(distance_prev_keyframe>=1)
//&&(parallax>=15)

//&&(parallax>=15)&&(distance_prev_keyframe>=1)
/**************************************************************3d point Insert******************************************************************************/

     if (((inlier_ratio<inlier_ratio_def))||(new_currFeatures.size()<200))	{////////////////////////////////////////////////////////////////////////////////////////////////////
      //  cout<<"keyframe insert!!"<<"\n";
      //  cout<<"inlier_ratio: "<<inlier_ratio<<"\n";
      //  cout<<"corr_2d_pointd_number: "<<corr_2d_pointd_number<<"\n";
      //  cout<<"average parallax: "<<parallax<<"\n";
      //  cout<<"tracking_ratio: "<<tracking_ratio<<"\n";
      //  cout<<"rotation angle differnce: "<<rot_ang_diff<<"\n";
       
      vector<pair<int,pair<int,Point3d>>> point_3d_map_tmp=point_3d_map;
      point_3d_map.clear();
      vector <pair<int,pair<int,Point3d>>>().swap(point_3d_map);

      
      R_tri.copyTo(Rt0.rowRange(0,3).colRange(0,3));
      t_tri.copyTo(Rt0.rowRange(0,3).col(3));
      
      R_solve.copyTo(Rt1.rowRange(0,3).colRange(0,3));
      tvec.copyTo(Rt1.rowRange(0,3).col(3));
      
      new_tri_prevFeatures.clear();
      vector <Point2f>().swap(new_tri_prevFeatures);
      
      //for (int i=0;i<new_tri_prev_points_map.size();i++){
        for (int i=0;i<new_tri_prev_points_map.size();i++){
          //if (new_tri_prev_points_map[i].first==new_curr_points_map[j].first){
            new_tri_prevFeatures.push_back(new_tri_prev_points_map[i].second.second);
            //currFeatures2.push_back(new_curr_points_map[j].second);
            //break;
          }
        
      
      
      

      //cout<<new_tri_prevFeatures<<"\n";
      
      
      //cout<<"triangulate!"<<"\n";
      triangulatePoints(Kd*Rt0,Kd*Rt1,new_tri_prevFeatures,new_currFeatures,point3d_homo);
      
      //cout<<point3d_homo.cols<<"\n";
      //int new_currFeatures_number=new_currFeatures.size();
      // vector<Point2f> new_currFeatures_corr;
      vector<int> point_3d_map_first;

     
      int new_currFeatures_size=new_currFeatures.size();

      // cloud->points.resize (point3d_homo.cols);
    
    double diff_sum=0;
    int average=0;
    for(int i = 0; i < point3d_homo.cols; i++) {
      
      int m = new_currFeatures.at(i).x;
      int n = new_currFeatures.at(i).y;
      //circle(currImage_c, Point((int)m, (int)n) ,2, CV_RGB(255,0,0), 2);

      // point = cloud->points[i];
      _p3h = point3d_homo.col(i); 
      p3d2 = _p3h/_p3h.at<float>(3);
      p3d2.convertTo(p3d22, CV_64F);
      


      point.x = p3d22.at<double>(0);
      point.y = p3d22.at<double>(1);
      point.z = p3d22.at<double>(2);
      point.r=100;
      point.g=100;
      point.b=100;

      three_to_p=Kd*Rt1*p3d22;
      int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
      int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
      //circle(currImage_c, Point((int)c,(int)d),2,CV_RGB(0,255,0),2);

      double dist_x = (point.x-t_solve_f.at<double>(0))*(point.x-t_solve_f.at<double>(0));
      double dist_y = (point.y-t_solve_f.at<double>(1))*(point.y-t_solve_f.at<double>(1));
      double dist_z = (point.z-t_solve_f.at<double>(2))*(point.z-t_solve_f.at<double>(2));


      int point_diff_x = (m-c)*(m-c);
      int point_diff_y = (n-d)*(n-d);
      diff_sum+=sqrt(point_diff_x+point_diff_y);
      average=diff_sum/i;

      float parallax2=0;
      float x_para2=new_currFeatures[i].x-new_tri_prevFeatures[i].x;
      float y_para2=new_currFeatures[i].y-new_tri_prevFeatures[i].y;

      parallax2=sqrt(x_para2*x_para2+y_para2*y_para2);

      //(sqrt( dist_x+dist_y+dist_z )<150)
            //&&(sqrt( dist_x+dist_y+dist_z )>distance_prev_keyframe)

      
      if ( 
             //&&(sqrt( dist_x+dist_y+dist_z )>5)
            (c>0)&&(d>0)&&(c<currImage.cols)&&(d<currImage.rows)
             &&(sqrt(point_diff_x+point_diff_y)<reprojectionError)
            &&(parallax2>=parallax_def)
            //&&(sqrt( dist_x+dist_y+dist_z )<distance_prev_keyframe*40)
            &&(sqrt( dist_x+dist_y+dist_z )<max_distance)
            ){ /////////////here
            //&&(sqrt(point_diff_x+point_diff_y)<5)
        // cout<<sqrt(point_diff_x+point_diff_y)<<"\n";
        // waitKey();
       
        //circle(currImage_c, Point(m, n) ,2, CV_RGB(255,0,0), 2);
       
        //circle(currImage_c, Point(c,d),2,CV_RGB(0,255,0),2);

        //msg->points.push_back(point);
        point_3d_map.push_back(make_pair(new_curr_points_map[i].first,make_pair(new_curr_points_map[i].second.first,Point3d(point.x,point.y,point.z))));
        //point_3d_map_first.push_back(new_curr_points_map[i].first);
        new_currFeatures.push_back(new_currFeatures[i]);
        new_curr_points_map.push_back(new_curr_points_map[i]);
      }
      
      // msg->points.push_back(point);
      // point_3d_map.push_back(make_pair(new_curr_points_map[i].first,Point3d(point.x,point.y,point.z)));
  }
     
      new_currFeatures.erase(new_currFeatures.begin(),new_currFeatures.begin()+new_currFeatures_size);
      new_curr_points_map.erase(new_curr_points_map.begin(),new_curr_points_map.begin()+new_currFeatures_size);
      
      // cout<<"keyframe_number: "<<keyframe_num-1<<"\n";
      // cout<<"entire frame number: "<<numFrame<<"\n";

       vector<Point2f> new_currFeatures_tmp_tmp2=new_currFeatures;
vector<pair<int,pair<int,Point2f>>> new_curr_points_map_tmp_tmp2=new_curr_points_map;
vector<pair<int,pair<int,Point3d>>> point_3d_map_tmp_tmp2=point_3d_map;

   
    vector<Point2f> new_currFeatures_tmp=currFeatures;
    vector<pair<int,pair<int,Point2f>>> new_curr_points_map_tmp=curr_points_map;

    int indexCorr=0;
    
    int new_curr_tmp_tmp_size=new_currFeatures_tmp_tmp2.size();
    
    for (int j=0;j<new_curr_tmp_tmp_size;j++){
      int count=0;
    for (int i=0;i<new_currFeatures_tmp.size();i++){
      
        float x_dis = new_currFeatures_tmp[i].x-new_currFeatures_tmp_tmp2[j].x;
        float y_dis = new_currFeatures_tmp[i].y-new_currFeatures_tmp_tmp2[j].y;
        if (sqrt(x_dis*x_dis+y_dis*y_dis )<10){
          count++;
        }
      }
        if (count==0){
          new_currFeatures_tmp_tmp2.push_back(new_currFeatures_tmp_tmp2[j]);
          point_3d_map_tmp_tmp2.push_back(point_3d_map_tmp_tmp2[j]);
          new_curr_points_map_tmp_tmp2.push_back(new_curr_points_map_tmp_tmp2[j]);
          
        }
    }
    new_currFeatures_tmp_tmp2.erase(new_currFeatures_tmp_tmp2.begin(),new_currFeatures_tmp_tmp2.begin()+new_curr_tmp_tmp_size);
    point_3d_map_tmp_tmp2.erase(point_3d_map_tmp_tmp2.begin(),point_3d_map_tmp_tmp2.begin()+new_curr_tmp_tmp_size);
    new_curr_points_map_tmp_tmp2.erase(new_curr_points_map_tmp_tmp2.begin(),new_curr_points_map_tmp_tmp2.begin()+new_curr_tmp_tmp_size);
      
    

          vector<Point2f> new_currFeatures_tmp_tmp=new_currFeatures_tmp_tmp2;
vector<pair<int,pair<int,Point2f>>> new_curr_points_map_tmp_tmp=new_curr_points_map_tmp_tmp2;
vector<pair<int,pair<int,Point3d>>> point_3d_map_tmp_tmp=point_3d_map_tmp_tmp2;

// cout<<" after new_currFeatures_tmp_tmp2 size: " << new_currFeatures_tmp_tmp2.size()<<"\n";


    for (int i=0;i<point_3d_map_tmp_tmp.size();i++){
      point_3d_map_tmp.push_back(point_3d_map_tmp_tmp[i]);
      new_currFeatures_tmp.push_back(new_currFeatures_tmp_tmp[i]);
      new_curr_points_map_tmp.push_back(new_curr_points_map_tmp_tmp[i]);
    }
    // if points are created more than 300, just erase excess points in the end.
    if (new_currFeatures_tmp.size()>max_feature_number){
      int new_currFeatures_tmp_size=new_currFeatures_tmp.size();
      for (int i=0;i<new_currFeatures_tmp_size-max_feature_number;i++){
        point_3d_map_tmp.pop_back();
      new_currFeatures_tmp.pop_back();
      new_curr_points_map_tmp.pop_back();
      }
    }

     point_3d_map=point_3d_map_tmp;
      prevFeatures = new_currFeatures_tmp;
      prev_points_map = new_curr_points_map_tmp;
      
      
     
       //**********************************************************************************************************************************************************************************************
       //draw new created points
       int prevFeatures_size=prevFeatures.size();
       currImage_c = imread(filename);
       cloud->points.resize (prevFeatures_size);

       int repro_sum=0;
    for(int i = 0; i < prevFeatures_size; i++) {
      
      int m = prevFeatures.at(i).x;
      int n = prevFeatures.at(i).y;
      //circle(currImage_c, Point(m, n) ,2, CV_RGB(255,0,0), 2);
      
      Mat point_3d(4,1,CV_64F);
      point_3d.at<double>(0)=point_3d_map[i].second.second.x;
      point_3d.at<double>(1)=point_3d_map[i].second.second.y;
      point_3d.at<double>(2)=point_3d_map[i].second.second.z;
      point_3d.at<double>(3)=1;
      
       three_to_p=Kd*Rt1*point_3d;
      //  cout<<point_3d<<"\n";
      //  cout<<Rt1<<"\n";
      //  cout<<three_to_p<<"\n";
      // waitKey();
        int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
        int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
      point = cloud->points[i];
      point.x = point_3d.at<double>(0);
      point.y = point_3d.at<double>(1);
      point.z = point_3d.at<double>(2);
      point.r=100;
      point.g=100;
      point.b=100;



         double dist_x = (point.x-t_solve_f.at<double>(0))*(point.x-t_solve_f.at<double>(0));
      double dist_y = (point.y-t_solve_f.at<double>(1))*(point.y-t_solve_f.at<double>(1));
      double dist_z = (point.z-t_solve_f.at<double>(2))*(point.z-t_solve_f.at<double>(2));
      //circle(currImage_c, Point(c, d) ,2, CV_RGB(0,255,0), 2);
      int point_diff_x = (m-c)*(m-c);
      int point_diff_y = (n-d)*(n-d);
      if ( 
             //&&(sqrt( dist_x+dist_y+dist_z )>5)
           (c>0)&&(d>0)&&(c<currImage.cols)&&(d<currImage.rows)
             &&(sqrt(point_diff_x+point_diff_y)<reprojectionError)
             &&(sqrt( dist_x+dist_y+dist_z )<max_distance)
            ){ /////////////here
            repro_sum+=sqrt(point_diff_x+point_diff_y);
            //&&(sqrt(point_diff_x+point_diff_y)<5)
        // cout<<sqrt(point_diff_x+point_diff_y)<<"\n";
        // waitKey();
        // three_to_p=Rt1*point_3d;
        // cout<<Rt1<<"\n";
        // cout<<point_3d<<"\n";
       
        // // cout<<three_to_p<<"\n";
        // waitKey();
         circle(currImage_c, Point(m, n) ,2, CV_RGB(255,0,0), 2);
        if (i<currFeatures.size()){
        circle(currImage_c, Point(c,d),2,CV_RGB(0,255,0),2);
        }
        else{
        circle(currImage_c, Point(c,d),2,CV_RGB(72,209,204),2);  
        }
        
        msg->points.push_back(point);
        // point_3d_map.push_back(make_pair(new_curr_points_map[i].first,make_pair(new_curr_points_map[i].second.first,Point3d(point.x,point.y,point.z))));
        // //point_3d_map_first.push_back(new_curr_points_map[i].first);
        // new_currFeatures.push_back(new_currFeatures[i]);
        // new_curr_points_map.push_back(new_curr_points_map[i]);
        point_3d_map.push_back(point_3d_map[i]);
      prevFeatures.push_back(prevFeatures[i]);
      prev_points_map.push_back(prev_points_map[i]);
      }
      
  }

 
  //imshow( "Road facing camera", currImage_c );
  image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", currImage_c).toImageMsg();
    image_pub.publish(image_msg);
  // waitKey(1000);
  point_3d_map.erase(point_3d_map.begin(),point_3d_map.begin()+prevFeatures_size);
  prevFeatures.erase(prevFeatures.begin(),prevFeatures.begin()+prevFeatures_size);
  prev_points_map.erase(prev_points_map.begin(),prev_points_map.begin()+prevFeatures_size);

   
      
      
      //cout<<"tracking point + new created point: "<<point_3d_map.size()<<"\n";
  
//***********************************************************************************************Add local ba points**************************************************************************************************************

if ((numFrame-numFrame_prev>=min_keyframe)||(rot_ang_diff>1.5)){
        keyframe_number++;
        if (number_of_3d_points.size()==local_ba_frame){
        BA_3d_points_map.erase(BA_3d_points_map.begin(),BA_3d_points_map.begin()+number_of_3d_points[0]);
        BA_2d_points_map.erase(BA_2d_points_map.begin(),BA_2d_points_map.begin()+number_of_3d_points[0]);
        number_of_3d_points.erase(number_of_3d_points.begin());
      
        Mat rvec_erase=rvec.clone();
        Mat tvec_erase=tvec.clone();
        rvec_erase.at<double>(0)=rvec_vec[0].x;
        rvec_erase.at<double>(1)=rvec_vec[0].y;
        rvec_erase.at<double>(2)=rvec_vec[0].z;
        tvec_erase.at<double>(0)=tvec_vec[0].x;
        tvec_erase.at<double>(1)=tvec_vec[0].y;
        tvec_erase.at<double>(2)=tvec_vec[0].z;
        Mat R_solve_tmp;
        Mat R_solve_inv_tmp;
        Mat t_solve_f_tmp;
        Rodrigues(rvec_erase,R_solve_tmp);
   
        

    R_solve_inv_tmp = R_solve_tmp.t();
    t_solve_f_tmp = -R_solve_inv_tmp*tvec_erase;

    
        rvec_vec.erase(rvec_vec.begin());
        tvec_vec.erase(tvec_vec.begin());
      }
      //traj_pub.publish(msg2);

      for (int i=0;i<point_3d_map.size();i++){
        BA_3d_points_map.push_back(point_3d_map[i]);
        BA_2d_points_map.push_back(prev_points_map[i]);

        BA_3d_points_map_loop.push_back(point_3d_map[i]);
        BA_2d_points_map_loop.push_back(prev_points_map[i]);
      }

      number_of_3d_points.push_back(point_3d_map.size());
      number_of_3d_points_loop.push_back(point_3d_map.size());
      
      BA_3d_points_map_tmp=BA_3d_points_map;
      int BA_3d_points_map_tmp_size=BA_3d_points_map_tmp.size();
      
      sort(BA_3d_points_map_tmp.begin(),BA_3d_points_map_tmp.end(),compare_point);
      BA_3d_points_map_tmp.erase(unique(BA_3d_points_map_tmp.begin(),BA_3d_points_map_tmp.end()),BA_3d_points_map_tmp.end());
      
      BA_3d_map_points.clear();
      vector <int>().swap(BA_3d_map_points);
      for (int i=0;i<BA_3d_points_map_tmp.size();i++){
        //cout<<i<<" "<<BA_3d_points_map_tmp[i].first<<" "<<BA_3d_points_map_tmp[i].second.first<<"\n";
        BA_3d_map_points.push_back(10000*BA_3d_points_map_tmp[i].first+BA_3d_points_map_tmp[i].second.first);
        //cout<<BA_3d_map_points.at(i)<<"\n";
      }
     
      
      rvec_vec.push_back(Point3d( rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2) ));
      
      tvec_vec.push_back(Point3d( tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2) ));
      
       
numFrame_prev=numFrame-1;

      //***************************************************************************************************local BA***************************************************************************************************************//
#if (local_ba>0)
if (number_of_3d_points.size()>=4){

  cout<<number_of_3d_points.size()<<"\n";

  cout<<"local BA start"<<"\n";
  //cout<<"before local ba tvec: "<<tvec.at<double>(0)<<" "<<tvec.at<double>(1)<<" "<<tvec.at<double>(2)<<"\n";

    int rvec_eig_local_size=rvec_vec.size();
   
    Eigen::MatrixXd rvec_eig_local(3,rvec_eig_local_size);
    Eigen::MatrixXd tvec_eig_local(3,rvec_eig_local_size);


    for (int i=0; i<rvec_eig_local_size;i++){
      rvec_eig_local(0,i)=rvec_vec[i].x;
      rvec_eig_local(1,i)=rvec_vec[i].y;
      rvec_eig_local(2,i)=rvec_vec[i].z;
      tvec_eig_local(0,i)=tvec_vec[i].x;
      tvec_eig_local(1,i)=tvec_vec[i].y;
      tvec_eig_local(2,i)=tvec_vec[i].z;
    }    
    
   
    Eigen::VectorXd BA_2d_points_eig(2);
    Eigen::MatrixXd BA_3d_points_eig(3,BA_3d_points_map_tmp.size());
    Eigen::VectorXi number_of_3d_points_eig(number_of_3d_points.size());
    
    int half_3d_points=0;
    for (int i=0;i<rvec_eig_local_size;i++){
      number_of_3d_points_eig[i]=number_of_3d_points[i];
      //cout<<number_of_3d_points_eig[i]<<"\n";
      if (i<rvec_eig_local_size/2){
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
    
    int index_vec=0;
    
    //cout<<half_3d_points<<"\n";
    //-number_of_3d_points[local_ba_frame-1]
    for (int i = 0; i < BA_2d_points_map.size(); i++) {
          index++;
          //cout<<BA_2d_points_map.at(i).first*10000+BA_2d_points_map.at(i).second.first<<"\n";
          auto it =find(BA_3d_map_points.begin(), BA_3d_map_points.end(), BA_2d_points_map.at(i).first*10000+BA_2d_points_map.at(i).second.first);
          
          //auto it = BA_3d_map_points.find(BA_2d_points_map[j].at(i).first*1000+BA_2d_points_map[j].at(i).second.first);
          BA_2d_points_eig[0]=(double)BA_2d_points_map.at(i).second.second.x;
          BA_2d_points_eig[1]=(double)BA_2d_points_map.at(i).second.second.y;
          

          int index_vec_num=0;
          for (int j = 0; j < number_of_3d_points.size(); j++)
            {
              index_vec_num += number_of_3d_points[j];
              if (i < index_vec_num)
              {
                index_vec = j;
                break;
              }
            }
          //cout<<it-BA_3d_map_points.begin()<<"\n";
          /*
          if (i<half_3d_points){
                // cout<<i<<"\n";
              ceres::CostFunction* cost_function2 = 
          new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local_pose_fixed, 2,3>(
            new SnavelyReprojectionError_Local_pose_fixed(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,i,number_of_3d_points_eig,rvec_eig_local,tvec_eig_local)
          );
       
    
    problem2.AddResidualBlock(cost_function2,
                             NULL ,
                             BA_3d_points_eig.col(it-BA_3d_map_points.begin()).data());

          }
          else{
            */
           ceres::CostFunction* cost_function2 = 
          new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3,3,3>(
            new SnavelyReprojectionError_Local(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,i,number_of_3d_points_eig)
          );
       
    
    problem2.AddResidualBlock(cost_function2,
                             NULL ,
                             rvec_eig_local.col(index_vec).data(),
                             tvec_eig_local.col(index_vec).data(),
                             BA_3d_points_eig.col(it-BA_3d_map_points.begin()).data());
          }
      
     // }
      
    //   cout<<"BA_3d_points_map_tmp size: "<<BA_3d_points_map_tmp.size()<<"\n";
    //  waitKey();
      
  //cout<<"local BA solver start"<<"\n";
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 12;
  options.max_num_iterations=100;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem2, &summary);
  
  
  rvec_vec.clear();
  vector <Point3d>().swap(rvec_vec);
  tvec_vec.clear();
  vector <Point3d>().swap(tvec_vec);
   for (int i=0;i<rvec_eig_local_size;i++){
     double rvec_eig_1=rvec_eig_local(0,i);
     double rvec_eig_2=rvec_eig_local(1,i);
     double rvec_eig_3=rvec_eig_local(2,i);
     double tvec_eig_1=tvec_eig_local(0,i);
     double tvec_eig_2=tvec_eig_local(1,i);
     double tvec_eig_3=tvec_eig_local(2,i);
     
     
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
      if (it==BA_3d_map_points.end()){
            cout<<"fail"<<"\n";
            
            waitKey();
          }
      int eig_index=it-BA_3d_map_points.begin();
      BA_3d_points_map.push_back(make_pair(BA_3d_points_map[i].first,make_pair(BA_3d_points_map[i].second.first,Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index)))));
      
      if (i>=BA_3d_points_map_size-number_of_3d_points[number_of_3d_points.size()-1]){
        point_3d_map.push_back(make_pair(BA_3d_points_map[i].first,make_pair(BA_3d_points_map[i].second.first,Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index)))));
      }
    }
    
BA_3d_points_map.erase(BA_3d_points_map.begin(),BA_3d_points_map.begin()+BA_3d_points_map_size);
point_3d_map.erase(point_3d_map.begin(),point_3d_map.begin()+point_3d_map_size);

     

  rvec.at<double>(0)=rvec_vec[rvec_eig_local_size-1].x;
  rvec.at<double>(1)=rvec_vec[rvec_eig_local_size-1].y;
  rvec.at<double>(2)=rvec_vec[rvec_eig_local_size-1].z;

  tvec.at<double>(0)=tvec_vec[rvec_eig_local_size-1].x;
  tvec.at<double>(1)=tvec_vec[rvec_eig_local_size-1].y;
  tvec.at<double>(2)=tvec_vec[rvec_eig_local_size-1].z;

  

Rodrigues(rvec,R_solve);
   
    R_solve_inv = R_solve.t();
    // cout<<R_solve_inv<<"\n";
    // waitKey(1000);
    t_solve_f = -R_solve_inv*tvec;
    

    
  //cout<<"after local ba tvec: "<<tvec.at<double>(0)<<" "<<tvec.at<double>(1)<<" "<<tvec.at<double>(2)<<"\n";
  
//************************************draw after local ba**********************************

      R_solve.copyTo(Rt1.rowRange(0,3).colRange(0,3));
      tvec.copyTo(Rt1.rowRange(0,3).col(3));
//currImage_c = imread(filename);
int repro_sum=0;
 for(int i = 0; i < prevFeatures.size(); i++) {
      
      int m = prevFeatures.at(i).x;
      int n = prevFeatures.at(i).y;
      //circle(currImage_c, Point(m, n) ,2, CV_RGB(255,0,0), 2);
      
      Mat point_3d(4,1,CV_64F);
      point_3d.at<double>(0)=point_3d_map[i].second.second.x;
      point_3d.at<double>(1)=point_3d_map[i].second.second.y;
      point_3d.at<double>(2)=point_3d_map[i].second.second.z;
      point_3d.at<double>(3)=1;

       three_to_p=Kd*Rt1*point_3d;
       
        int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
        int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
      circle(currImage_c, Point(c, d) ,2, CV_RGB(0,0,255), 2);
    int point_diff_x=(m-c)*(m-c);
    int point_diff_y=(n-d)*(n-d);
    repro_sum+=sqrt(point_diff_x+point_diff_y);
      
  }
  //cout <<"after local ba average reprojection error: "<<repro_sum/prevFeatures.size()<<"\n";
    //imshow( "Road facing camera", currImage_c );
    image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", currImage_c).toImageMsg();
    image_pub.publish(image_msg);
    
    //cout<<"local BA end"<<"\n";
}
#endif
//********************************************************************************************************localBA end******************************************************************************************************************//
#if (loopclosing)
      t_solve_f_vec.push_back(Point3d(t_solve_f.at<double>(0),t_solve_f.at<double>(1),t_solve_f.at<double>(2)));
      
    trajectory = cloud2->points[keyframe_number];
    trajectory.x=t_solve_f.at<double>(0);
    trajectory.y=t_solve_f.at<double>(1);
    trajectory.z=t_solve_f.at<double>(2);
    trajectory.r=0;
    trajectory.g=255.0f;
    trajectory.b=0;
    
    msg2->points.push_back(trajectory);
      numFrame_vec.push_back(numFrame);
      {
        Eigen::Matrix3d mat_eig;
        for (int i=0;i<3;i++){
            for(int j=0;j<3;j++){
              mat_eig(i,j)=R_solve_inv.at<double>(i,j);
            }
          }
        Eigen::Quaterniond quat(mat_eig);
        quat_vec.push_back(quat);
      }
      // for (int i=0;i<quat_vec.size();i++){
      //   cout<<quat_vec[i].w()<<" "<<quat_vec[i].x()<<" "<<quat_vec[i].y()<<" "<<quat_vec[i].z()<<"\n";
      //   waitKey(2000);
      // }
      
      
      rvec_vec_loop.push_back(Point3d(rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2)));
      tvec_vec_loop.push_back(Point3d(tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2)));



      vector<KeyPoint> keypoints_1;
      
      Mat descriptor_1;
      // KeyPoint::convert( prevFeatures, keypoints_1);
      //detector->compute(currImage,keypoints_1,descriptor_1);
      Mat mask2;
      detector->detectAndCompute(currImage, mask2, keypoints_1, descriptor_1);
      
       features.push_back(vector<cv::Mat >());
      changeStructure(descriptor_1, features.back());
      
      QueryResults ret;
      testDatabase(features,db,Isloopdetected,keyframe_prev,keyframe_curr);
      // &&once_loop_detected==0

      if ((features.size()-prev_traj_num)>20){
        once_loop_detected=0;
      }

      if (once_loop_detected==1){
        Isloopdetected=0;
      }

      if (Isloopdetected&&(once_loop_detected==0)){
        vector<g2o::SE3Quat> gt_poses;
        
        // cout<<"loop closing start"<<"\n";
        // cout<<"keyframe_prev id: "<<keyframe_prev<<"\n";
        // cout<<"keyframe_curr id: "<<keyframe_curr<<"\n";
        // cout<<"t_solve_f_vec size: "<<t_solve_f_vec.size()<<"\n";
        // cout<<"quat_vec size: "<<quat_vec.size()<<"\n";
        // cout<<"BA_3d_points_map_loop size: "<<BA_3d_points_map_loop.size()<<"\n";

        loop_keyframe_num.push_back(keyframe_prev);
        loop_keyframe_num.push_back(keyframe_curr);

        //g2o SE3
    
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linear_solver
            = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    std::unique_ptr<g2o::BlockSolverX> block_solver
            = g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
    g2o::OptimizationAlgorithm* algorithm
            = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
    // g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer;
    // optimizer->setAlgorithm(algorithm);
    // optimizer->setVerbose(true);
    
    //g2o sim3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>
      LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));



          g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer;
          optimizer->setAlgorithm(algorithm);
          optimizer->setVerbose(true);
          g2o::SparseOptimizer optimizer_sim3;
           g2o::ParameterSE3Offset* cameraOffset = new g2o::ParameterSE3Offset;
          cameraOffset->setId(0);
          optimizer_sim3.addParameter(cameraOffset);
          optimizer_sim3.setAlgorithm(solver);
          optimizer_sim3.setVerbose(true);
        //first, set fixed vertex
        //cout<<"first, set fixed vertex"<<"\n";
        {
          
          Eigen::Vector3d trans;
      
          trans[0]=t_solve_f_vec[0].x;
          trans[1]=t_solve_f_vec[0].y;
          trans[2]=t_solve_f_vec[0].z;
      

          g2o::SE3Quat pose0(quat_vec[0], trans);
          addPoseVertex(optimizer, pose0, true,0);
          gt_poses.push_back(pose0);
        }
        // set variable vertices
        //cout<<"set variable vertices"<<"\n";
        {
          // Eigen::Matrix3d mat_eig;
          Eigen::Vector3d trans;

          for (int k=1;k<=keyframe_curr;k++){
            trans[0]=t_solve_f_vec[k].x;
            trans[1]=t_solve_f_vec[k].y;
            trans[2]=t_solve_f_vec[k].z;


          // Eigen::Quaterniond quat(mat_eig);
          g2o::SE3Quat pose0(quat_vec[k], trans);
          addPoseVertex(optimizer, pose0, true,k);
          gt_poses.push_back(pose0);
          }
        }
        
        //find loop constraint
          {
          Ptr<Feature2D> detector_loop = ORB::create(4000);
          int prev_numFrame = numFrame_vec.at(keyframe_prev);
          int curr_numFrame = numFrame_vec.at(keyframe_curr);
          
          sprintf(filename1, path_to_image, prev_numFrame);
          sprintf(filename2, path_to_image, curr_numFrame);
          //cout<<"load image"<<"\n";
          Mat prev_image_c = imread(filename1);
          Mat curr_image_c = imread(filename2);

          Mat prev_image;
          Mat curr_image;

          cvtColor(prev_image_c, prev_image, COLOR_BGR2GRAY);
          cvtColor(curr_image_c, curr_image, COLOR_BGR2GRAY);
         
          vector<KeyPoint> prev_keypoints;
          vector<KeyPoint> curr_keypoints;
          
          Mat prev_desc;
          Mat curr_desc;
          detector_loop->detectAndCompute(prev_image,noArray(),prev_keypoints,prev_desc);
          detector_loop->detectAndCompute(curr_image,noArray(),curr_keypoints,curr_desc);
          
          //cout<<"BFMatcher"<<"\n";
          BFMatcher matcher(NORM_HAMMING);
          vector<DMatch> matches;
          matcher.match(prev_desc,curr_desc,matches);

          double minDist,maxDist;
          minDist=maxDist=matches[0].distance;
          for (int i=1;i<matches.size();i++){
          double dist = matches[i].distance;
          if (dist<minDist) minDist=dist;
          if (dist>maxDist) maxDist=dist;
          }
          vector<DMatch> goodMatches;
          double fTh= 4.0*minDist;
          for (int i=0;i<matches.size();i++){
            if (matches[i].distance <=max(fTh,0.02))
              goodMatches.push_back(matches[i]);
            }
          
          vector<Point2f> prev_features;
          vector<Point2f> curr_features;
          for(int i=0;i<goodMatches.size();i++){
            prev_features.push_back(prev_keypoints[goodMatches[i].queryIdx].pt);
            curr_features.push_back(curr_keypoints[goodMatches[i].trainIdx].pt);
          }

          E = findEssentialMat(curr_features, prev_features, focal, pp, RANSAC, 0.999, 1.0, mask);
          recoverPose(E, curr_features, prev_features, R, t, focal, pp, mask);

         

          Eigen::Vector3d trans;
          
          trans = Eigen::Vector3d(t.at<double>(0),t.at<double>(1),t.at<double>(2));
          
          Eigen::Matrix3d mat_eig;
          for (int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                  mat_eig(i,j)=R.at<double>(i,j);
                }
              }
          Eigen::Quaterniond quat(mat_eig);
          trans_vec_loop.push_back(trans);
          quat_vec_loop.push_back(quat);
          Mat img_match;
          drawMatches(prev_image, prev_keypoints, curr_image, curr_keypoints, goodMatches, img_match);
         
          imshow("Matches", img_match);
          waitKey();
          }


        // set edges between poses
        {
          g2o::SE3Quat relpose;
          for (int j=1;j<2;j++){
          for (int i=j;i<gt_poses.size();i++){
            // relpose: pose[i] w.r.t pose[i-1]
            relpose = gt_poses[i-j].inverse() * gt_poses[i];
            addEdgePosePose(optimizer, i-j, i, relpose);
          }
          }
        
        // the last  vertex is same with the keyframe_prev=>Using 2d-2d matching, find relative pose
          
          for (int i=0;i<loop_keyframe_num.size()/2;i++){
            g2o::SE3Quat pose0(quat_vec_loop[i], trans_vec_loop[i]);
            
            addEdgePosePose(optimizer, loop_keyframe_num[2*i], loop_keyframe_num[2*i+1], pose0);


          }
        }
        

        // Convert SE3 to Sim3
        //cout<<"Convert SE3 to Sim3"<<"\n";
        // Convert all vertices
        for (int i=0;i<gt_poses.size();i++){
          g2o::VertexSE3* v_se3 = static_cast<g2o::VertexSE3*>(optimizer->vertex(i));
          g2o::VertexSim3Expmap* v_sim3 = new g2o::VertexSim3Expmap();
          v_sim3->setId(i);
          v_sim3->setMarginalized(false);
          //cout<<"id: "<<i<<"\n";
          //cout<<"vertex t:" << v_se3->estimate().translation()[0]<<" "<<v_se3->estimate().translation()[1]<<" "<<v_se3->estimate().translation()[2]<<"\n";
          
            ToVertexSim3(*v_se3, v_sim3,1.0);
          
          // cout<<v_sim3->estimate().translation()<<"\n";

          optimizer_sim3.addVertex(v_sim3);
          if (i == 0) {
            v_sim3->setFixed(true);
          }
        }

          vector<int> number_of_3d_points_loop_tmp;
           for (int i=0;i<=keyframe_curr;i++){
              number_of_3d_points_loop_tmp.push_back(number_of_3d_points_loop[i]);
              
            }
            // cout<<"number_of_3d_points_loop size: "<<number_of_3d_points_loop.size()<<"\n";
            // cout<<"number of 3d points loop tmp size: "<<number_of_3d_points_loop_tmp.size()<<"\n";

          
          //relative scale by orb matching
          //cout<<"relative scale by orb matching"<<"\n";
          double rel_scale=1;
          {
          double scale;
          int prev_numFrame = numFrame_vec.at(keyframe_prev);
          int curr_numFrame = numFrame_vec.at(keyframe_curr);
          
          // KeyPoint::convert( prevFeatures, keypoints_1);

          sprintf(filename1, path_to_image, prev_numFrame);
          sprintf(filename2, path_to_image, curr_numFrame);
          //cout<<"load image"<<"\n";
          Mat prev_image_c = imread(filename1);
          Mat curr_image_c = imread(filename2);

          Mat prev_image;
          Mat curr_image;

          cvtColor(prev_image_c, prev_image, COLOR_BGR2GRAY);
          cvtColor(curr_image_c, curr_image, COLOR_BGR2GRAY);
          int prev_idx=0;
          for (int i=0;i<keyframe_prev;i++){
            prev_idx+=number_of_3d_points_loop_tmp[i];
          }
          //cout<<prev_idx<<"\n";
          vector<Point2f> prev_Features;
          vector<Point3d> prev_3d_points;
          for (int i=prev_idx;i<prev_idx+number_of_3d_points_loop_tmp[keyframe_prev];i++){
            prev_Features.push_back(BA_2d_points_map_loop[i].second.second);
            prev_3d_points.push_back(BA_3d_points_map_loop[i].second.second);
            //cout<<"prev 3d points: "<<prev_3d_points[i]<<"\n";
          }
          vector<KeyPoint> prev_keypoints;
          vector<KeyPoint> curr_keypoints;
          
          KeyPoint::convert( prev_Features, prev_keypoints);
          KeyPoint::convert( prevFeatures, curr_keypoints);

          Mat prev_desc;
          Mat curr_desc;
          detector->compute(prev_image,prev_keypoints,prev_desc);
          detector->compute(curr_image,curr_keypoints,curr_desc);
          //cout<<"BFMatcher"<<"\n";
          BFMatcher matcher(NORM_HAMMING);
          vector<DMatch> matches;
          matcher.match(prev_desc,curr_desc,matches);


          double minDist,maxDist;
          minDist=maxDist=matches[0].distance;
          for (int i=1;i<matches.size();i++){
          double dist = matches[i].distance;
          if (dist<minDist) minDist=dist;
          if (dist>maxDist) maxDist=dist;
          }
          vector<DMatch> goodMatches;
          double fTh= 4.0*minDist;
          for (int i=0;i<matches.size();i++){
            if (matches[i].distance <=max(fTh,0.02))
              goodMatches.push_back(matches[i]);
            }
          //cout<<goodMatches.size()<<"\n";
          
          // cout<<goodMatches.size()<<"\n";
          // cout<<matches.size()<<"\n";
          vector<double> dist_vec;
          for (int i=0;i<goodMatches.size();i++){
          int prev_gm_idx = goodMatches[i].queryIdx;
          int curr_gm_idx = goodMatches[i].trainIdx;
          
          double x_prev = (prev_3d_points[prev_gm_idx].x-t_solve_f_vec[keyframe_prev].x);
          double y_prev = (prev_3d_points[prev_gm_idx].y-t_solve_f_vec[keyframe_prev].y);
          double z_prev = (prev_3d_points[prev_gm_idx].z-t_solve_f_vec[keyframe_prev].z);

          double prev_dist=sqrt ( x_prev*x_prev + y_prev*y_prev +z_prev*z_prev );

          double x_curr = (point_3d_map[curr_gm_idx].second.second.x-t_solve_f_vec[keyframe_curr].x);
          double y_curr = (point_3d_map[curr_gm_idx].second.second.y-t_solve_f_vec[keyframe_curr].y);
          double z_curr = (point_3d_map[curr_gm_idx].second.second.z-t_solve_f_vec[keyframe_curr].z);

          double curr_dist= sqrt( x_curr*x_curr+y_curr*y_curr +z_curr*z_curr );
          
         
          scale=curr_dist/prev_dist;
          dist_vec.push_back(scale);

          }

          sort(dist_vec.begin(),dist_vec.end());
          rel_scale=dist_vec.at(goodMatches.size()/2);
          cout<<"rel scale: "<<rel_scale<<"\n";
          Mat img_match;
          drawMatches(prev_image, prev_keypoints, curr_image, curr_keypoints, goodMatches, img_match);
          imshow("Matches", img_match);
          waitKey();
          }



          // Convert all edges
        int edge_index = 0;
        for (auto& tmp : optimizer->edges()) {
          g2o::EdgeSE3* e_se3 = static_cast<g2o::EdgeSE3*>(tmp);
          int idx0 = e_se3->vertex(0)->id();
          int idx1 = e_se3->vertex(1)->id();
          g2o::EdgeSim3* e_sim3 = new g2o::EdgeSim3();
          if ( (idx0==keyframe_prev)&&(idx1==keyframe_curr)){
            ToEdgeSim3(*e_se3,e_sim3,rel_scale);
          }
          else{
          ToEdgeSim3(*e_se3, e_sim3,1.0);
          }
          e_sim3->setId(edge_index++);
          e_sim3->setVertex(0, optimizer_sim3.vertices()[idx0]);
          e_sim3->setVertex(1, optimizer_sim3.vertices()[idx1]);
          e_sim3->information() = Eigen::Matrix<double, 7, 7>::Identity();

          optimizer_sim3.addEdge(e_sim3);
        }

       // cout<<"optimizer_sim3 vertices size: "<<optimizer_sim3.vertices().size()<<"\n";
        //cout<<"optimizer_sim3 edge size: "<<optimizer_sim3.edges().size()<<"\n";





        //cout<<"initializing ..."<<"\n";
        optimizer_sim3.initializeOptimization();
       // cout << "optimizing ..." << endl;
        optimizer_sim3.optimize(100);
        



        //cout<<"t_solve_f_vec size: "<<t_solve_f_vec.size()<<"\n";
        t_solve_f_vec.clear();
        vector <Point3d>().swap(t_solve_f_vec);
        quat_vec.clear();
        vector <Eigen::Quaterniond>().swap(quat_vec);
        rvec_vec.clear();
        vector <Point3d>().swap(rvec_vec);
        tvec_vec.clear();
        vector <Point3d>().swap(tvec_vec);
        //cout<<"msg2 size: "<<msg2->points.size()<<"\n";
        msg2->points.clear();
        msg->points.clear();
        msg4->points.clear();
        // msg2->points.clear();
        vector<pair<int,pair<int,Point3d>>> BA_3d_points_map2 = BA_3d_points_map;
        vector<pair<int,pair<int,Point3d>>> BA_3d_points_map_loop2 = BA_3d_points_map_loop;


        BA_3d_points_map.clear();
        vector<pair<int,pair<int,Point3d>>>().swap(BA_3d_points_map);
        BA_3d_points_map_loop.clear();
        vector<pair<int,pair<int,Point3d>>>().swap(BA_3d_points_map_loop);
        point_3d_map.clear();
        vector<pair<int,pair<int,Point3d>>>().swap(point_3d_map);

        
      // msg4->points.clear();
      int number_of_points=0;
      int number_of_points_prev=0;
      cloud->points.resize(BA_3d_points_map_loop.size());

      vector<Point3d> rvec_vec_loop_tmp=rvec_vec_loop;
      vector<Point3d> tvec_vec_loop_tmp=tvec_vec_loop;

      rvec_vec_loop.clear();
      vector<Point3d>().swap(rvec_vec_loop);
      tvec_vec_loop.clear();
      vector<Point3d>().swap(tvec_vec_loop);

      //plot
      for (int i=0;i<gt_poses.size();i++){
         
        //g2o::VertexSE3* vtx = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(i));
        g2o::VertexSim3Expmap* vtx =
        static_cast<g2o::VertexSim3Expmap*>(optimizer_sim3.vertex(i));
        g2o::Sim3 sim3 = vtx->estimate().inverse(); //pose
        g2o::Sim3 sim3_inv = vtx->estimate(); // projection
        Eigen::Matrix3d r = sim3_inv.rotation().toRotationMatrix(); // projection
        Eigen::Vector3d t = sim3_inv.translation();
        
        double s = sim3_inv.scale();
        t *=(1./s);
        Eigen::Matrix3d r_inv = r.transpose();
        Eigen::Vector3d t_traj = -r_inv*t;
          
        tvec_vec_loop.push_back(Point3d(t[0],t[1],t[2]));

        Mat rvec_rot(3,3,CV_64F);
        for (int a=0;a<3;a++){
            for(int b=0;b<3;b++){
              rvec_rot.at<double>(a,b)=r(a,b);
            }
          }
        Mat rvec_tmp2;
        Rodrigues(rvec_rot,rvec_tmp2);

        rvec_vec_loop.push_back(Point3d(rvec_tmp2.at<double>(0),rvec_tmp2.at<double>(1),rvec_tmp2.at<double>(2)));

        Eigen::Matrix3d mat_eig_tmp;
      for (int a=0;a<3;a++){
            for(int b=0;b<3;b++){
              mat_eig_tmp(a,b)=r.inverse()(a,b);
            }
          }
      Eigen::Quaterniond quat(mat_eig_tmp);
      quat_vec.push_back(quat);

        //cout<<"After optimization"<<"\n";
        // cout<<r<<"\n";
        //cout<<t<<"\n";
        if (i>=gt_poses.size()-local_ba_frame){
          // cout<<i<<"\n";
          Mat rvec_tmp;
          Mat r_inverse(3,3,CV_64F);
          for (int a=0;a<3;a++){
            for(int b=0;b<3;b++){
              r_inverse.at<double>(a,b)=r(a,b);
            }
          }
          Rodrigues(r_inverse,rvec_tmp);
          rvec_vec.push_back(Point3d(rvec_tmp.at<double>(0),rvec_tmp.at<double>(1),rvec_tmp.at<double>(2)));
          tvec_vec.push_back(Point3d(t[0],t[1],t[2]));
        }


          t_solve_f_vec.push_back(Point3d(t_traj[0],t_traj[1],t_traj[2]));


          Mat mat_sim3 = Mat::eye(4, 4, CV_64FC1);
          Mat mat_prev = Mat::eye(4, 4, CV_64FC1);
          
          for (int m=0;m<3;m++){
            for (int n=0;n<3;n++){
              mat_sim3.at<double>(m,n)=r(m,n);
            }
          }

            mat_sim3.at<double>(0,3)=t(0);
            mat_sim3.at<double>(1,3)=t(1);
            mat_sim3.at<double>(2,3)=t(2);
          
          


          Mat rot_mat;
          Mat rvec_tmp(3,1,CV_64F);

          rvec_tmp.at<double>(0)=rvec_vec_loop_tmp[i].x;
          rvec_tmp.at<double>(1)=rvec_vec_loop_tmp[i].y;
          rvec_tmp.at<double>(2)=rvec_vec_loop_tmp[i].z;

          Rodrigues(rvec_tmp,rot_mat);

          rot_mat.copyTo(mat_prev.rowRange(0,3).colRange(0,3));

          Mat tvec_tmp(3,1,CV_64F);
          
          tvec_tmp.at<double>(0)=tvec_vec_loop_tmp[i].x;
          tvec_tmp.at<double>(1)=tvec_vec_loop_tmp[i].y;
          tvec_tmp.at<double>(2)=tvec_vec_loop_tmp[i].z;


          number_of_points+=number_of_3d_points_loop_tmp[i];

          tvec_tmp.copyTo(mat_prev.rowRange(0,3).col(3));
          //corrected 3d points add

          for (int j=number_of_points_prev;j<number_of_points;j++){
            Mat prev_3d_point(4,1,CV_64F);
            prev_3d_point.at<double>(0)=BA_3d_points_map_loop2[j].second.second.x;
            prev_3d_point.at<double>(1)=BA_3d_points_map_loop2[j].second.second.y;
            prev_3d_point.at<double>(2)=BA_3d_points_map_loop2[j].second.second.z;
            prev_3d_point.at<double>(3)=1;

            

            Mat opt_3d_point;
            Mat mat_sim3_inv = mat_sim3.inv();
            opt_3d_point = mat_sim3_inv* mat_prev  * prev_3d_point;
            
            //cout << Point3d(opt_3d_point.at<double>(0), opt_3d_point.at<double>(1), opt_3d_point.at<double>(2))<<"\n";

            point = cloud->points[j];
            point.x = opt_3d_point.at<double>(0);
            point.y = opt_3d_point.at<double>(1);
            point.z = opt_3d_point.at<double>(2);
            point.r=100;
            point.g=100;
            point.b=100;
            msg->points.push_back(point);

            BA_3d_points_map_loop.push_back(make_pair(BA_3d_points_map_loop2[j].first,make_pair(BA_3d_points_map_loop2[j].second.first,Point3d(point.x,point.y,point.z))));

            if (i>=gt_poses.size()-local_ba_frame){
              BA_3d_points_map.push_back(make_pair(BA_3d_points_map_loop2[j].first,make_pair(BA_3d_points_map_loop2[j].second.first,Point3d(point.x,point.y,point.z))));
            }

            if (i==gt_poses.size()-1){
              point_3d_map.push_back(make_pair(BA_3d_points_map_loop2[j].first,make_pair(BA_3d_points_map_loop2[j].second.first,Point3d(point.x,point.y,point.z))));
            }

          }

          number_of_points_prev=number_of_points;


        trajectory = cloud2->points[i];
        trajectory.x=t_traj[0];
        trajectory.y=t_traj[1];
        trajectory.z=t_traj[2];

        
        // cout<<trajectory.x<<" "<<trajectory.y<<" "<<trajectory.z<<"\n";
        trajectory.r=255;
        trajectory.g=165;
        trajectory.b=0;
        msg4->points.push_back(trajectory);
        trajectory.r=0;
        trajectory.g=255;
        trajectory.b=0;
        msg2->points.push_back(trajectory);
      }
      
        //cout<<"rvec_vec size: "<<rvec_vec.size()<<"\n";
        traj_pub.publish(msg2);
        keyframe_pub.publish(msg4);
        world_points_pub.publish(msg);
        //cout<<"keyframe pub publish"<<"\n";
        //cout<<"msg->points size: "<<msg->points.size()<<"\n";
        //cout<<"BA_3d_points_map_loop size: "<<BA_3d_points_map_loop.size()<<"\n";
        //remove same 3d point
        
       vector<pair<int,pair<int,Point3d>>> BA_3d_points_map_tmp_loop=BA_3d_points_map_loop;
        BA_3d_points_map_tmp_size=BA_3d_points_map_tmp_loop.size();
      
        sort(BA_3d_points_map_tmp_loop.begin(),BA_3d_points_map_tmp_loop.end(),compare_point);
        
        // for (int i=0;i<BA_3d_points_map_tmp_loop.size();i++){
        //   cout<<BA_3d_points_map_tmp_loop[i].first<<" "<<BA_3d_points_map_tmp_loop[i].second.first<<"\\";
        // }
        // waitKey();

        vector<pair<int,pair<int,Point3d>>> BA_3d_points_map_loop_rm;
        //cout<<"BA_3d_points_map_tmp_loop size: "<<BA_3d_points_map_tmp_loop.size()<<"\n";
          for (int i=0;i<BA_3d_points_map_tmp_loop.size();i++){
            if ((i>0)&&((BA_3d_points_map_tmp_loop[i-1].first!=BA_3d_points_map_tmp_loop[i].first)||(BA_3d_points_map_tmp_loop[i-1].second.first!=BA_3d_points_map_tmp_loop[i].second.first)))
            {
              BA_3d_points_map_loop_rm.push_back(BA_3d_points_map_tmp_loop[i]);
            } 
            else if(i==0){
              BA_3d_points_map_loop_rm.push_back(BA_3d_points_map_tmp_loop[i]);
              // cout<<BA_3d_points_map_tmp_loop[i].first<<" "<<BA_3d_points_map_tmp_loop[i].second.first<<"\n";
              // waitKey();
            }
          }
        // cout<<"\n";
        // cout<<"--------------------------------"<<"\n";
        // for (int i=0;i<BA_3d_points_map_loop_rm.size();i++){
        //   cout<<BA_3d_points_map_loop_rm[i].first<<" "<<BA_3d_points_map_loop_rm[i].second.first<<"\\";
        // }
        // waitKey();



        //cout<<"BA_3d_points_map_rm size: "<<BA_3d_points_map.size()<<"\n";
        //cout<<"BA_3d_points_map_loop_rm size: "<<BA_3d_points_map_loop_rm.size()<<"\n";
        //cout<<"point_3d_map size: "<<point_3d_map.size()<<"\n";
        

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

        //waitKey();



        cout<<"pgo end"<<"\n";
        

        //after loop closing bundle adjustment
//*********************************************************full ba***********************************************************
{ 
  //after loop closing full ba
  //cout<<"full BA start"<<"\n";
  //cout<<"before local ba tvec: "<<tvec.at<double>(0)<<" "<<tvec.at<double>(1)<<" "<<tvec.at<double>(2)<<"\n";

  int rvec_eig_local_size=rvec_vec_loop.size();
    // Eigen::MatrixXd rvec_eig_local(1,3*local_ba_frame);
    // Eigen::MatrixXd tvec_eig_local(1,3*local_ba_frame);
    // Eigen::VectorXd rvec_eig_local(rvec_eig_local_size);
    // Eigen::VectorXd tvec_eig_local(rvec_eig_local_size);
    Eigen::MatrixXd rvec_eig_local(3,rvec_eig_local_size);
    Eigen::MatrixXd tvec_eig_local(3,rvec_eig_local_size);


    for (int i=0; i<rvec_vec_loop.size();i++){
      rvec_eig_local(0,i)=rvec_vec_loop[i].x;
      rvec_eig_local(1,i)=rvec_vec_loop[i].y;
      rvec_eig_local(2,i)=rvec_vec_loop[i].z;
      tvec_eig_local(0,i)=tvec_vec_loop[i].x;
      tvec_eig_local(1,i)=tvec_vec_loop[i].y;
      tvec_eig_local(2,i)=tvec_vec_loop[i].z;

    }    
    
    // for (int i=0;i<rvec_vec_loop.size();i++){
    //   cout<<rvec_vec_loop[i]<<"\n";
    //   cout<<tvec_vec_loop[i]<<"\n";
    // }
    // waitKey();


   
    Eigen::VectorXd BA_2d_points_eig(2);
    Eigen::MatrixXd BA_3d_points_eig(3,BA_3d_points_map_loop_rm.size());
    Eigen::VectorXi number_of_3d_points_eig(number_of_3d_points_loop.size());
    
    for (int i=0;i<rvec_vec_loop.size();i++){
      number_of_3d_points_eig[i]=number_of_3d_points_loop[i];
    }


    for (int i=0;i<BA_3d_points_map_loop_rm.size();i++){
      BA_3d_points_eig(0,i)=BA_3d_points_map_loop_rm[i].second.second.x;
      BA_3d_points_eig(1,i)=BA_3d_points_map_loop_rm[i].second.second.y;
      BA_3d_points_eig(2,i)=BA_3d_points_map_loop_rm[i].second.second.z;
    }
    
    vector<int> BA_3d_map_points_loop;
    for (int i=0;i<BA_3d_points_map_loop_rm.size();i++){
        //cout<<i<<" "<<BA_3d_points_map_tmp[i].first<<" "<<BA_3d_points_map_tmp[i].second.first<<"\n";
        BA_3d_map_points_loop.push_back(10000*BA_3d_points_map_loop_rm[i].first+BA_3d_points_map_loop_rm[i].second.first);
        //cout<<BA_3d_map_points.at(i)<<"\n";
      }

    

    ceres::Problem problem2;

    int index_vec=0;
    //-number_of_3d_points[local_ba_frame-1]
    //cout<<"BA 2d points map loop size: "<<BA_2d_points_map_loop.size()<<"\n";
    for (int i = number_of_3d_points_loop[0]; i < BA_2d_points_map_loop.size(); i++) {
          
          int index_vec_num=0;
          for (int j = 0; j < number_of_3d_points_loop.size(); j++)
            {
              index_vec_num += number_of_3d_points_loop[j];
              if (i < index_vec_num)
              {
                index_vec = j;
                break;
              }
            }
            
            
          //cout<<BA_2d_points_map.at(i).first*10000+BA_2d_points_map.at(i).second.first<<"\n";
          auto it =find(BA_3d_map_points_loop.begin(), BA_3d_map_points_loop.end(), BA_2d_points_map_loop.at(i).first*10000+BA_2d_points_map_loop.at(i).second.first);
          //cout<<"index_vec: "<<index_vec<<" index number: "<<BA_2d_points_map_loop.at(i).first*10000+BA_2d_points_map_loop.at(i).second.first<<" it: "<<it-BA_3d_map_points_loop.begin()<<"\n";
          //auto it = BA_3d_map_points.find(BA_2d_points_map[j].at(i).first*1000+BA_2d_points_map[j].at(i).second.first);
          BA_2d_points_eig[0]=(double)BA_2d_points_map_loop.at(i).second.second.x;
          BA_2d_points_eig[1]=(double)BA_2d_points_map_loop.at(i).second.second.y;
          
          if (it==BA_3d_map_points_loop.end()){
            cout<<"fail"<<"\n";
            // cout<<i<<"\n";
            // cout<<index_vec<<"\n";
            // cout<<BA_2d_points_map_loop.at(i).first*10000+BA_2d_points_map_loop.at(i).second.first<<"\n";
            waitKey();
          }



          //cout<<BA_2d_points_eig[0]<<" "<<BA_2d_points_eig[1]<<"\n";
          
           ceres::CostFunction* cost_function2 = 
          new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local_pose_fixed, 2,3>(
            new SnavelyReprojectionError_Local_pose_fixed(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,i,number_of_3d_points_eig,
                                                    rvec_eig_local.col(index_vec),tvec_eig_local.col(index_vec))
          );
       
    
    problem2.AddResidualBlock(cost_function2,
                             NULL ,
                             BA_3d_points_eig.col(it-BA_3d_map_points_loop.begin()).data());
          }
      
      
  //cout<<"full BA solver start"<<"\n";
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 12;
  options.max_num_iterations=200;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem2, &summary);
  // int num_threads=summary.num_threads_used;
  // cout<<num_threads<<"\n";
  // waitKey();
  //std::cout << summary.FullReport() << "\n";
  //**********************8
  
  //*************************
  


  
  rvec_vec.clear();
  vector <Point3d>().swap(rvec_vec);
  tvec_vec.clear();
  vector <Point3d>().swap(tvec_vec);

  int whole_point=rvec_vec_loop.size();

  rvec_vec_loop.clear();
  vector<Point3d>().swap(rvec_vec_loop);
  tvec_vec_loop.clear();
  vector<Point3d>().swap(tvec_vec_loop);

   for (int i=0;i<whole_point;i++){
     double rvec_eig_1=rvec_eig_local(0,i);
     double rvec_eig_2=rvec_eig_local(1,i);
     double rvec_eig_3=rvec_eig_local(2,i);
     double tvec_eig_1=tvec_eig_local(0,i);
     double tvec_eig_2=tvec_eig_local(1,i);
     double tvec_eig_3=tvec_eig_local(2,i);
     
     rvec_vec_loop.push_back(Point3d(rvec_eig_1,rvec_eig_2,rvec_eig_3));
     tvec_vec_loop.push_back(Point3d(tvec_eig_1,tvec_eig_2,tvec_eig_3));
     
     if (i>=whole_point-local_ba_frame){
      rvec_vec.push_back(Point3d(rvec_eig_1,rvec_eig_2,rvec_eig_3));
      tvec_vec.push_back(Point3d(tvec_eig_1,tvec_eig_2,tvec_eig_3));
     }
    }
    
    int local_index=0;
    for (int i=number_of_3d_points.size()-local_ba_frame;i<number_of_3d_points.size();i++){
      local_index+=number_of_3d_points[i];
    }
    
    vector <pair<int,pair<int,Point3d>>> BA_3d_points_map_loop_tmp;
    // vector <pair<int,pair<int,Point3d>>>().swap(BA_3d_points_map);
    int point_3d_map_size=point_3d_map.size();
    int BA_3d_points_map_size = BA_3d_points_map.size();
  for (int i=0;i<BA_3d_points_map_loop.size();i++){
      int map_first=BA_3d_points_map_loop[i].first;
      int map_second_first=BA_3d_points_map_loop[i].second.first;
      
      auto it =find(BA_3d_map_points_loop.begin(), BA_3d_map_points_loop.end(), map_first*10000+map_second_first);
      
      int eig_index=it-BA_3d_map_points_loop.begin();
      BA_3d_points_map_loop_tmp.push_back(make_pair(map_first,make_pair(map_second_first,Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index)))));
      
      if (i>=BA_3d_points_map_loop.size()-local_index){
        BA_3d_points_map.push_back(make_pair(map_first,make_pair(map_second_first,Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index)))));
      }

      if (i>=BA_3d_points_map_loop.size()-number_of_3d_points[number_of_3d_points.size()-1]){
        point_3d_map.push_back(make_pair(map_first,make_pair(map_second_first,Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index)))));
      }
    }
  BA_3d_points_map_loop=BA_3d_points_map_loop_tmp;

BA_3d_points_map.erase(BA_3d_points_map.begin(),BA_3d_points_map.begin()+BA_3d_points_map_size);
point_3d_map.erase(point_3d_map.begin(),point_3d_map.begin()+point_3d_map_size);

 //remove local ba 3d point
        {
          
          BA_3d_points_map_tmp=BA_3d_points_map;
        sort(BA_3d_points_map_tmp.begin(),BA_3d_points_map_tmp.end(),compare_point);
        
        vector<pair<int,pair<int,Point3d>>> BA_3d_points_map_rm;
        
          for (int i=0;i<BA_3d_points_map_tmp.size();i++){
            if ( (i>0)&&((BA_3d_points_map_tmp[i-1].first!=BA_3d_points_map_tmp[i].first)||(BA_3d_points_map_tmp[i-1].second.first!=BA_3d_points_map_tmp[i].second.first)))
            {
              BA_3d_points_map_rm.push_back(BA_3d_points_map_tmp[i]);
            }
            else if(i==0){
              BA_3d_points_map_rm.push_back(BA_3d_points_map_tmp[i]);
            }
          }
          BA_3d_points_map_tmp=BA_3d_points_map_rm;
        }

  rvec.at<double>(0)=rvec_vec[local_ba_frame-1].x;
  rvec.at<double>(1)=rvec_vec[local_ba_frame-1].y;
  rvec.at<double>(2)=rvec_vec[local_ba_frame-1].z;

  tvec.at<double>(0)=tvec_vec[local_ba_frame-1].x;
  tvec.at<double>(1)=tvec_vec[local_ba_frame-1].y;
  tvec.at<double>(2)=tvec_vec[local_ba_frame-1].z;

  
  Rodrigues(rvec,R_solve);

  R_solve_inv = R_solve.t();
  t_solve_f = -R_solve_inv*tvec;



    t_solve_f_vec.clear();
    vector <Point3d>().swap(t_solve_f_vec);
    quat_vec.clear();
    vector <Eigen::Quaterniond>().swap(quat_vec);
    msg2->points.clear();
    msg->points.clear();
    
    
    

//-----------------------------------------------------------------------------------
  for(int i=0;i<rvec_vec_loop.size();i++){
    trajectory = cloud2->points[i];
        

        Mat rvec_tmp(3,1,CV_64F);
        Mat tvec_tmp(3,1,CV_64F);
        
        rvec_tmp.at<double>(0)=rvec_vec_loop[i].x;
        rvec_tmp.at<double>(1)=rvec_vec_loop[i].y;
        rvec_tmp.at<double>(2)=rvec_vec_loop[i].z;

        tvec_tmp.at<double>(0)=tvec_vec_loop[i].x;
        tvec_tmp.at<double>(1)=tvec_vec_loop[i].y;
        tvec_tmp.at<double>(2)=tvec_vec_loop[i].z;

        Mat R_solve_tmp(3,3,CV_64F);
        Rodrigues(rvec_tmp,R_solve_tmp);
        
        Mat R_solve_inv_tmp=R_solve_tmp.t(); //pose
        Mat t_solve_f_tmp=-R_solve_inv_tmp*tvec_tmp;//pose
        
        Eigen::Matrix3d mat_eig_tmp;
        for (int a=0;a<3;a++){
            for(int b=0;b<3;b++){
              mat_eig_tmp(a,b)=R_solve_inv_tmp.at<double>(a,b);
            }
        }
      Eigen::Quaterniond quat(mat_eig_tmp);
      quat_vec.push_back(quat);

      

        trajectory.x=t_solve_f_tmp.at<double>(0);
        trajectory.y=t_solve_f_tmp.at<double>(1);
        trajectory.z=t_solve_f_tmp.at<double>(2);
        trajectory.r=0;
        trajectory.g=255;
        trajectory.b=0;

        t_solve_f_vec.push_back(Point3d(trajectory.x,trajectory.y,trajectory.z));
        
        msg2->points.push_back(trajectory);
        
        
  }

  cloud->points.resize(BA_3d_points_map_loop.size());
  for (int i=0;i<BA_3d_points_map_loop.size();i++){
    point = cloud->points[i];
            point.x = BA_3d_points_map_loop[i].second.second.x;
            point.y = BA_3d_points_map_loop[i].second.second.y;
            point.z = BA_3d_points_map_loop[i].second.second.z;
            point.r=100;
            point.g=100;
            point.b=100;
            msg->points.push_back(point);
  }
    
    traj_pub.publish(msg2);
    world_points_pub.publish(msg);
}

        

        Isloopdetected=0;
        once_loop_detected=1;

        prev_traj_num=rvec_vec_loop.size();
        
      }
      #endif




}

//************************************loop closing end*******************************************************************************//
     diagonal = R_solve.at<double>(0,0)+R_solve.at<double>(1,1)+R_solve.at<double>(2,2);

    rot_ang = acos( (diagonal-1.0)/2);
    rot_ang = rot_ang*(180/CV_PI);
      prev_rot_ang=rot_ang;

      t_solve_f_prev=t_solve_f.clone();
      // imshow( "Road facing camera", currImage_c );
      // waitKey();
  
      //if (abs(rot_ang_diff)<3.0){
       //cout<<"detect next feature"<<"\n";
        featureDetection(prevImage, new_prevFeatures,new_prev_points_map,keyframe_num,MAX_CORNERS);
      
      // else{
      //   featureDetection(prevImage, new_prevFeatures,new_prev_points_map,2000);
      // }
      // new_tri_prevFeatures=new_prevFeatures;
      
      new_tri_prev_points_map=new_prev_points_map;
      //cout<<"tracking next feature"<<"\n";
      vector<uchar> status2;
      featureTracking(prevImage, currImage, new_prevFeatures, new_currFeatures,new_prev_points_map,new_curr_points_map, status2,points2_tmp);
      erase_int_point2f(prevImage,points2_tmp,new_tri_prev_points_map,status2);
      //cout<<"new tracking feature number: "<<new_currFeatures.size()<<"\n";
      // new_prevFeatures = new_currFeatures;
      //  new_prev_points_map = new_curr_points_map;
      tracking_number_last=new_currFeatures.size();
       R_tri = R_solve_prev.clone();
       t_tri = t_solve_prev.clone();
       //cout<<"keyframe insert end!!"<<"\n";
 	  }
    else{
       
       prevFeatures = currFeatures;
       prev_points_map = curr_points_map;

      //  new_prevFeatures = new_currFeatures;
      //  new_prev_points_map = new_curr_points_map;
     }
      
      new_prevFeatures = new_currFeatures;
       new_prev_points_map = new_curr_points_map;
      prevImage = currImage.clone();
      

    R_solve_prev=R_solve.clone();
    t_solve_prev=tvec.clone();
    
    
    trajectory.x=gt_x;
    trajectory.y=gt_y;
    trajectory.z=gt_z;
    trajectory.r=0;
    trajectory.g=0;
    trajectory.b=255.0f;
    gt_msg->points.push_back(trajectory);



    world_points_pub.publish(msg);
    traj_pub.publish(msg2);
    tracking_pub.publish(msg3);
    gt_traj_pub.publish(gt_msg);
         


    waitKey(1);

        
    //cout<<"Frame end"<<"\n";
    }
    ros::spinOnce();
    loop_rate.sleep();
    
    return 0;
   }

}
    
    