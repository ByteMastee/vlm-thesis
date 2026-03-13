#!/bin/bash

sed -i '$ d' robot.urdf
sed -i 's,\.\.,model://UVC_robot_new,g' robot.urdf
cat plugins.txt >> robot.urdf
