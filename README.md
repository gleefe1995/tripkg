# mono vo

```
cd catkin_ws/src
git clone https://github.com/gleefe1995/tripkg.git
cd tripkg/Thirdparty
cd DBoW2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ../../
cd g2o
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
roslaunch tripkg triangulation.launch
rosrun tripkg monovo
```

![kitti05](https://user-images.githubusercontent.com/67038853/132988450-419ad8e1-c5c1-42fc-9186-ef1367d4bb59.gif)

![kitti06](https://user-images.githubusercontent.com/67038853/132988664-4c215e96-9141-4549-ab1b-3063e6bac866.gif)



### reference

<https://github.com/avisingh599/mono-vo>

