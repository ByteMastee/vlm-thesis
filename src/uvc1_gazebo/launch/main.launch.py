<?xml version="1.0" encoding="UTF-8"?>
<launch>
<arg name="world" default="$(find robot_model)/worlds/hospital.world"/>
    <arg name="paused" default="false"/>
    <arg name="use_sim_time" default="true"/>
    <arg name="gui" default="true"/>
    <arg name="headless" default="false"/>
    <arg name="debug" default="false"/>
    <arg name="simulator" default="true"/>

<include file="$(find vf_camera_tf_publisher)/launch/publish_tfs.launch"/>
  <include file="$(find robot_manager)/launch/robot_manager.launch">
      <arg name="default_mode_automatic" value="true"/>
      <arg name="simulator" value="$(arg simulator)"/>
  </include>
  <node type="rviz" name="rviz" pkg="rviz" args="-d $(find robot_model)/rviz/path_finder_two.rviz" />
  <include file="$(find rosbridge_server)/launch/rosbridge_websocket.launch" />
  <include file="$(find clientmanager)/launch/clientmanager.launch" />
  <!-- <include file="$(find slam)/launch/path_finder_two.launch" /> -->
  <node pkg="robot_model" name="ultrasound" type="ultrasound" />

  <env name="GAZEBO_MODEL_PATH" value="$(find robot_model)/models:$(optenv GAZEBO_MODEL_PATH)"/>

<!--   We resume the logic in empty_world.launch, changing only the name of the world to be launched -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(arg world)"/>
    <arg name="paused" value="$(arg paused)"/>
    <arg name="use_sim_time" value="$(arg use_sim_time)"/>
    <arg name="gui" value="$(arg gui)"/>
    <arg name="headless" value="$(arg headless)"/>
    <arg name="debug" value="$(arg debug)"/>
  </include>

  <!-- Spawn the robot into Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-file $(find robot_model)/models/UVC_robot_new/urdf/robot.urdf -urdf -model robot" />


<!--   <node type="teleop_twist_keyboard.py" name="app_emulator" pkg="app_emulator"/> -->
</launch>