## Credits

This code is modified by taking a snapshot from the open-source code of deepracer available [here](https://github.com/aws-robotics/aws-robomaker-sample-application-deepracer). The changes include the track color, the captured camera sampling rate, and the removal of the AWS dependencies.

## Requirements

- ROS Kinetic.
- Gazebo.

## Building the code

```bash
source /opt/ros/kinetic/setup.bash
cd simulation_ws
rosws update
rosdep install --from-paths src --ignore-src -r -y
colcon build
colcon bundle
```

### Known issue:
- change permissions of simulation_ws/bundle/bundle_staging/etc/sudoers.d/README , if colcon bundle fails.

## Starting Deepracer Gazebo simulator locally

```bash
source simulation_ws/install/setup.sh
roslaunch deepracer_simulation racetrack_with_racecar.launch world_name:=hard_track
```
