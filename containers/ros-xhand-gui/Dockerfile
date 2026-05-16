FROM osrf/ros:noetic-desktop-full

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-tk \
    ros-noetic-rosbridge-server \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install matplotlib pandas nptyping

# Set up XDG runtime dir for root user
RUN mkdir -p -m 0700 /run/user/0
ENV XDG_RUNTIME_DIR=/run/user/0

# Qt display fix
ENV QT_X11_NO_MITSHM=1

# Source the ROS workspace on container start
RUN echo "source /mnt/xhand1/dexterous_hands/devel/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]