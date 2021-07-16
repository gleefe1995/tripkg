# mono_vo_triangulationpoints

```
cd catkin_ws/src
git clone https://github.com/gleefe1995/mono_vo_triangulationpoints.git
cd ..
catkin_make
source devel/setup.bash
rosrun tripkg triangulation
```

![image](https://user-images.githubusercontent.com/67038853/116507323-02f78b80-a8fa-11eb-895b-742b70a73dda.png)

red points : feature

blue points : triangulation 계산 후 camera coordinate로 옮긴 points

3d points를 cloud point msg로 전달

### reference

<https://github.com/avisingh599/mono-vo>

