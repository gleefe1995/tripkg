#include <iostream>
#include <fstream>

using namespace std;

double gt_x = 0, gt_y = 0, gt_z = 0;
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