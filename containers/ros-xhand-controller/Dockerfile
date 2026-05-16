FROM osrf/ros:noetic-desktop-full

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libcrypto++-dev \
    libcrypto++-doc \
    libcrypto++-utils \
    ros-noetic-rosbridge-suite \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pandas nptyping

# Source the ROS workspace on container start
RUN echo "source /mnt/xhand1/dexterous_hands/devel/setup.bash" >> /root/.bashrc

# CMD can still be overriden
CMD ["/bin/bash", "-c", "source /mnt/xhand1/dexterous_hands/devel/setup.bash && roslaunch xhand_control_ros xhand_control.launch"]