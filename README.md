# mono vo

```bash
cd catkin_ws/src
git clone https://github.com/gleefe1995/tripkg.git
cd tripkg/Thirdparty
// git clone DBoW2
cd DBoW2
// CMakefile에서 set lib path 
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ../../
// git clone g2o
cd g2o
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
roslaunch tripkg triangulation.launch
rosrun tripkg monovo scene_number
```
![kitti01](https://user-images.githubusercontent.com/67038853/160248628-a4565f41-119a-4ece-9692-e455fd57c8a6.png)


![kitti05](https://user-images.githubusercontent.com/67038853/132988450-419ad8e1-c5c1-42fc-9186-ef1367d4bb59.gif)

![kitti06](https://user-images.githubusercontent.com/67038853/132988664-4c215e96-9141-4549-ab1b-3063e6bac866.gif)



### reference

<https://github.com/avisingh599/mono-vo>

