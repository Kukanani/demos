# The Picky Robot

This demo uses a UR5 robot and a depth camera to detect objects
and push them off the table. It is a vision-based manipulation demo
that does not require a gripper.

This demo requires some dependencies:
  - python-urx: a ROS 2 fork (see the `ros2` branch) is available at
    https://github.com/Kukanani/python-urx.
  - A suitable depth camera library: this demo uses defaults that work
    with the Orbbec Astra, and the included intra-process launcher file
    depends on and uses the Astra driver. It should be possible to switch
    out to a different depth camera by adding dependencies, changing camera
    parameters, and writing a new intra-process launcher (which is a short
    file). A ROS 2 fork of the astra camera driver is available at
    https://github.com/ros2/ros_astra_camera.
  - The URX driver also requires some python dependencies - `numpy`, `yaml`, and
    `math3d`. `math3d` must be installed from pip - `pip3 install math3d`. This
    can be installed using `sudo` or in a virtual environment, but be advised
    that you must avoid pitfalls when using virtual environments with ros2 - the
    namely, the python interpreter used for a node launcher is always the same
    as the active interpreter when it was built.