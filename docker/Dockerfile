# comac_painter_ur5:dev
# Downloads the comac and other source code for development


### ROS and Moveit develope environment
FROM jchen0213/ros_dev_env
MAINTAINER Jinfa Chen and Xiang Zheng


# Setup ROS workspace
WORKDIR /root/ws_emo


### Universal Robots ROS pkg
COPY emo_ur5/ src/emo_ur5/


### comac painter demo pkg
COPY painter_demo/ src/painter_demo/

### install gazebo
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python-pip \
    ros-melodic-gazebo-* \
    ros-melodic-rqt \
    ros-melodic-rqt-common-plugins \
    ros-melodic-rqt-joint-trajectory-controller \
    ros-melodic-soem* \
    ros-melodic-socketcan-interface* \
    ros-melodic-joint-state-publisher-gui \
    gedit && \
    rm -rf /var/lib/apt/lists/*

### Build all the ROS pkgs. Will not build unneccesary pkg in the future
ENV PYTHONIOENCODING UTF-8
RUN catkin config --extend /opt/ros/$ROS_DISTRO --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    # Status rate is limited so that just enough info is shown to keep Docker from timing out,
    # but not too much such that the Docker log gets too long (another form of timeout)
    catkin build ur_msgs --limit-status-rate 0.001 --no-notify && \
    catkin build household_objects_database_msgs --limit-status-rate 0.001 --no-notify &&  \
    catkin build manipulation_msgs  --limit-status-rate 0.001 --no-notify && \
    catkin build

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

### ROS Industrial

COPY docker/ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

