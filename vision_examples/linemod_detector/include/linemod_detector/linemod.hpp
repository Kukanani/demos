#ifndef _LINEMOD_H_
#define _LINEMOD_H_

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/rgbd.hpp>

#include <vector>

#include "ork_renderer/utils.h"

struct LinemodDetection {
  cv::Matx33f rotation;
  cv::Vec3f translation;
  std::string class_name;
};

class Linemod {
public:
  Linemod();

  std::vector<LinemodDetection> detect(cv::Mat& color_in, cv::Mat& depth_in, cv::Mat& display);

private:
  void drawResponse(const std::vector<cv::linemod::Template>& templates,
                    int num_modalities, cv::Mat& dst, cv::Point offset, int T);

  cv::linemod::Detector readLinemod(const std::string& filename);

  void inferDepth(cv::Mat_<cv::Vec3f>& depth_real_ref_raw, cv::linemod::Match& match, cv::Vec3f& T_real_icp, cv::Matx33f& R_real_icp);

  cv::linemod::Detector detector_;
  // Various settings and flags
  bool show_match_result = true;
  bool learn_online = false;
  int num_classes = 0;
  int matching_threshold = 1;
  int learning_lower_bound = 90;
  int learning_upper_bound = 95;
  cv::Mat display;

  std::map<std::string, std::vector<cv::Mat> > Ts;
  std::map<std::string, std::vector<cv::Mat> > Rs;
  std::map<std::string, std::vector<cv::Mat> > Ks;
  std::map<std::string, std::vector<float> > distances;
  std::map<std::string, RendererIterator*> renderer_iterators;
};

#endif