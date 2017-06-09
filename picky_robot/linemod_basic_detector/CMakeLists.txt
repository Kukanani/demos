cmake_minimum_required(VERSION 3.5)
project(linemod_basic_detector)

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Wpedantic")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(astra_camera REQUIRED)
find_package(ork_renderer REQUIRED)
find_package(GLUT REQUIRED)
find_package(ASSIMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(intra_process_demo REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

###############################################################################
# Hack to allow usage of Orbbec Astra library
option(FILTER "OB Filter library" ON)

if( ${FILTER} STREQUAL "ON" )
set(obfilter "On")
else()
set(obfilter "Off")
endif()

include(ExternalProject)

ExternalProject_Add(astra_openni2
  PREFIX astra_openni2
  # SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/astra_openni2/OpenNI2
  GIT_REPOSITORY https://github.com/orbbec/OpenNI2.git
  GIT_TAG orbbec_ros
  CONFIGURE_COMMAND echo "no need to configure"
  #${CMAKE_CURRENT_SOURCE_DIR}/libantlr/configure --prefix=<INSTALL_DIR>
  BUILD_IN_SOURCE 1
  BUILD_COMMAND make release FILTER=${obfilter}
  INSTALL_DIR openni2
  INSTALL_COMMAND tar -xjf <SOURCE_DIR>/Packaging/Final/OpenNI-Linux-2.3.tar.bz2 -C <INSTALL_DIR> --strip 1 && mkdir -p <INSTALL_DIR>/include && ln -fs <INSTALL_DIR>/Include <INSTALL_DIR>/include/openni2
)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/openni2/Redist)
MESSAGE(WARNING "${CMAKE_CURRENT_BINARY_DIR}/openni2/Redist")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/openni2/include)
###############################################################################


include_directories(
  include
  # ${astra_camera_INCLUDE_DIRS}
  # ${Boost_INCLUDE_DIRS}
  # ${ork_renderer_INCLUDE_DIRS}
  # ${intra_process_demo_INCLUDE_DIRS}
  # ${vision_msgs_INCLUDE_DIRS}
  # ${tf2_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED src/linemod.cpp src/linemod_icp.cpp)
ament_target_dependencies(${PROJECT_NAME}
  "OpenCV"
  "ork_renderer")
# target_link_libraries(${PROJECT_NAME} "freeimage" ${OPENGL_LIBRARIES})

add_executable(linemod_node
  src/linemod_node.cpp)
ament_target_dependencies(linemod_node
  "rclcpp"
  "sensor_msgs"
  "vision_msgs"
  "OpenCV"
  "${PROJECT_NAME}"
  "ork_renderer"
  # "GLUT"
  # "ASSIMP"
  # "OPENGL"
  "intra_process_demo"
  "tf2"
  "tf2_ros"
  "tf2_geometry_msgs"
)
# target_link_libraries(linemod_node ${PROJECT_NAME} "freeimage" ${OPENGL_LIBRARIES})
target_link_libraries(linemod_node ${PROJECT_NAME})

add_executable(linemod_pipeline
  src/linemod_pipeline.cpp)
ament_target_dependencies(linemod_pipeline
  "rclcpp"
  "sensor_msgs"
  "vision_msgs"
  "astra_camera"
  "astra_driver_lib"
  "astra_wrapper"
  "astra_openni2"
  "OpenCV"
  "Boost"
  "${PROJECT_NAME}"
  "ork_renderer"
  # "GLUT"
  # "ASSIMP"
  # "OPENGL"
  "intra_process_demo"
  "tf2"
  "tf2_ros"
  "tf2_geometry_msgs"
)
add_dependencies(linemod_pipeline astra_openni2)
# target_link_libraries(linemod_pipeline ${PROJECT_NAME} -lOpenNI2Orbbec ${Boost_LIBRARIES} "freeimage")
target_link_libraries(linemod_pipeline ${PROJECT_NAME} -lOpenNI2Orbbec)

add_executable(linemod_train_standalone_kinect
  src/linemod_train_standalone_kinect.cpp)
ament_target_dependencies(linemod_train_standalone_kinect
  "OpenCV"
  "Boost")

add_executable(linemod_train_virtual
  src/linemod_train_virtual.cpp)
ament_target_dependencies(linemod_train_virtual
  "OpenCV"
  "Boost"
  "ork_renderer"
  # "GLUT"
  # "ASSIMP"
  # "OPENGL"
)
# target_link_libraries(linemod_train "freeimage")
target_link_libraries(linemod_train_virtual ${ork_renderer_LIBRARIES})

install(TARGETS
  linemod_node
  linemod_pipeline
  linemod_train_standalone_kinect
  linemod_train_virtual
  DESTINATION bin)

ament_package()
