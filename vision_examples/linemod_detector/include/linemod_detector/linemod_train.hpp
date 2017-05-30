#ifndef _LINEMOD_TRAINER_HPP_
#define _LINEMOD_TRAINER_HPP_

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

// #include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if CV_MAJOR_VERSION == 3
#include <opencv2/rgbd.hpp>
namespace cv {
using namespace cv::rgbd;
}
#else
#include <opencv2/objdetect/objdetect.hpp>
#endif

#include <ork_renderer/utils.h>
#include <ork_renderer/renderer3d.h>
#include <opencv2/highgui/highgui.hpp>

class Trainer
{
public:
  /** True or False to output debug image */
  bool visualize_ = true;
  /** The DB parameters as a JSON string */
  std::string json_db_;
  /** The id of the object to generate a trainer for */
  std::string object_id_;
  cv::linemod::Detector detector_;
  std::vector<cv::Mat>  Rs_;
  std::vector<cv::Mat>  Ts_;
  std::vector<float>  distances_;
  std::vector<cv::Mat>  Ks_;

  constexpr static int param_n_points_ = 150;
  constexpr static int param_angle_step_ = 10;
  constexpr static double param_radius_min_ = 0.6;
  constexpr static double param_radius_max_ = 1.1;
  constexpr static double param_radius_step_ = 0.4;
  constexpr static int param_width_ = 640;
  constexpr static int param_height_ = 480;
  constexpr static double param_near_ = 0.1;
  constexpr static double param_far_ = 1000.0;
  constexpr static double param_focal_length_x_ = 525;
  constexpr static double param_focal_length_y_ = 525;

  int renderer_n_points_;
  int renderer_angle_step_;
  double renderer_radius_min_;
  double renderer_radius_max_;
  double renderer_radius_step_;
  int renderer_width_;
  int renderer_height_;
  double renderer_near_;
  double renderer_far_;
  double renderer_focal_length_x_;
  double renderer_focal_length_y_;

  void writeLinemod(const std::string& filename);

  void train(std::string filename);
};

#endif