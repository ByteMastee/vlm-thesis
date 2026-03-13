<?xml version="1.0" encoding="UTF-8"?>
<launch>
  <include file="$(find robot_manager)/launch/robot_manager.launch">
      <arg name="default_mode_automatic" value="true"/>
  </include>
  <node type="rviz" name="rviz" pkg="rviz" args="-d $(find robot_model)/rviz/path_finder_two.rviz" />
  <!-- <include file="$(find slam)/launch/path_finder_two.launch" /> -->
  <node pkg="rosbag" type="play" name="player" output="screen" args="--clock /home/robi/path_finder_two_test.bag"/>
</launch>
<!--  
  Bag visszajatszas a path_finder_two tesztelesehez
  Kotelezo topicok:
    "/robot/odometry"
    "/d435/depth/color/points"

  Rosbag felvetele:
    rosbag record - -node=path_finder_two

 -->
