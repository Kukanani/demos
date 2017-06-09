cmake_minimum_required(VERSION 3.5)
project(ork_renderer)

find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(ASSIMP REQUIRED)

find_package(GLUT REQUIRED)
add_definitions(-DUSE_RENDERER_GLUT=1)

################################################################################
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${ASSIMP_INCLUDE_DIRS}
                    ${OPENGL_INCLUDE_DIR}
                    ${Boost_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${GLUT_INCLUDE_DIR}
)
link_directories(${ASSIMP_LIBRARY_DIRS})

################################################################################
add_library(${PROJECT_NAME}_2d SHARED
  src/renderer2d.cpp
)
ament_target_dependencies(${PROJECT_NAME}_2d
  "OpenCV"
)
install(
  TARGETS ${PROJECT_NAME}_2d
  DESTINATION lib
  RUNTIME DESTINATION bin
)

################################################################################
add_library(${PROJECT_NAME}_3d SHARED
  src/model.cpp
  src/renderer3d.cpp
  src/utils.cpp
  src/renderer3d_impl_glut.cpp
)
ament_target_dependencies(${PROJECT_NAME}_3d
   GLUT freeimage ASSIMP
)
target_link_libraries(${PROJECT_NAME}_3d
  freeimage
  ${OPENGL_LIBRARIES}
)
install(
  TARGETS ${PROJECT_NAME}_3d
  DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

################################################################################
add_executable(view_generator
  src/main.cpp
)
ament_target_dependencies(view_generator
  OPENGL
  GLUT
  OpenCV
  ASSIMP
  freeimage
)
target_link_libraries(view_generator
  ${PROJECT_NAME}_2d
  ${PROJECT_NAME}_3d
  freeimage
  ${OPENGL_LIBRARIES}
)
# set_property(
#   TARGET view_generator
#   PROPERTY DEBUG_POSTFIX ${ASSIMP_DEBUG_POSTFIX}
# )
install(
  TARGETS view_generator
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)


################################################################################
add_executable(trainer_planar
  src/apps/trainer_planar.cpp
)
# set_property(
#   TARGET trainer_planar
#   PROPERTY DEBUG_POSTFIX ${ASSIMP_DEBUG_POSTFIX}
# )
target_link_libraries(trainer_planar ${PROJECT_NAME}_2d)
set_target_properties(trainer_planar PROPERTIES
                                     OUTPUT_NAME trainer_planar
)
install(TARGETS trainer_planar
        DESTINATION lib RUNTIME DESTINATION bin
)


################################################################################
install(
  DIRECTORY include/
  DESTINATION include
)
ament_export_libraries(${PROJECT_NAME}_3d)
ament_export_include_directories(include)
ament_package()