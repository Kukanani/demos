#include "linemod_detector/linemod.hpp"

#include <iterator>
#include <cstdio>
#include <iostream>
#include <set>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "linemod_detector/linemod_train.hpp"
#include "linemod_detector/linemod_icp.hpp"

typedef std::vector<cv::linemod::Template> TemplatePyramid;
typedef std::map<cv::String, std::vector<TemplatePyramid> > TemplatesMap;

// Functions to store detector and templates in single XML/YAML file
cv::linemod::Detector Linemod::readLinemod(const std::string& filename)
{
  cv::linemod::Detector detector;
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector.read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
  {
    detector.readClass(*i);
    cv::String class_id = (*i)["class_id"];

    cv::FileNode tps_fn = (*i)["template_pyramids"];
    cv::FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();

    // std::map<std::string, std::vector<cv::Mat> >::value_type T_val(class_id, cv::Mat());
    // cv::Mat& T_ref = T_val.second;

    // std::map<std::string, std::vector<cv::Mat> >::value_type R_val(class_id, cv::Mat());
    // cv::Mat& R_ref = R_val.second;

    // std::map<std::string, std::vector<float> >::value_type distance_val(class_id, float);
    // float& distance_ref = distance_val.second;
    int expected_id = 0;

    // cv::FileNode tps_fn = fn["template_pyramids"];
    for ( ; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
      int template_id = (*tps_it)["template_id"];

      cv::Mat T_new, R_new, K_new;

      cv::FileNodeIterator fni = (*tps_it)["T"].begin();
      fni >> T_new;
      Ts[class_id].push_back(T_new);

      fni = (*tps_it)["R"].begin();
      fni >> R_new;
      Rs[class_id].push_back(R_new);

      fni = (*tps_it)["K"].begin();
      fni >> K_new;
      Ks[class_id].push_back(K_new);

      distances[class_id].push_back((*tps_it)["distance"]);
    }

    Renderer3d *renderer_ = new Renderer3d("/home/adam/osrf/g7/src/adamfiles/testobj/mug.dae");
    renderer_->set_parameters(
      Trainer::param_width_,
      Trainer::param_height_,
      Trainer::param_focal_length_x_,
      Trainer::param_focal_length_y_,
      Trainer::param_near_,
      Trainer::param_far_);

    RendererIterator *renderer_iterator_ = new RendererIterator(renderer_, Trainer::param_n_points_);
    renderer_iterator_->angle_step_ = Trainer::param_angle_step_;
    renderer_iterator_->radius_min_ = float(Trainer::param_radius_min_);
    renderer_iterator_->radius_max_ = float(Trainer::param_radius_max_);
    renderer_iterator_->radius_step_ = float(Trainer::param_radius_step_);

    renderer_iterators.insert(std::pair<std::string,RendererIterator*>(class_id, renderer_iterator_));
  }
  return detector;
}

Linemod::Linemod()
{
  // not initializer list because it also fills the T, R and distance vectors.
  detector_ = readLinemod("linemod_ros2.yml");
  cv::namedWindow("normals");

  std::vector<cv::String> ids = detector_.classIds();
  num_classes = detector_.numClasses();
  std::cout << "Loaded linemod templates with " << num_classes <<
      " classes, " << detector_.numTemplates() << " templates, and " <<
      detector_.getModalities().size() << " modalities." << std::endl;
  if (!ids.empty())
  {
    printf("Class ids:\n");
    std::copy(ids.begin(), ids.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
  }
}


void Linemod::detect(cv::Mat& color_in, cv::Mat& depth_in, cv::Mat& display)
{

  int num_modalities = detector_.getModalities().size();
  std::cout << "performing linemod with " << num_modalities << " modalities..." << std::endl;

  std::vector<cv::Mat> sources;
  sources.push_back(color_in);
  sources.push_back(depth_in);
  display = color_in.clone();
  cv::Mat depth_proc = depth_in.clone();

  // Perform matching
  std::vector<cv::linemod::Match> matches;
  std::vector<cv::String> class_ids;
  std::vector<cv::Mat> quantized_images;
  float matching_threshold = 91.6;
  detector_.match(sources, (float)matching_threshold, matches, class_ids, quantized_images);

  int classes_visited = 0;
  std::set<std::string> visited;

  cv::Mat_<cv::Vec3f> depth_real_ref_raw;
  cv::Mat_<float> K(3,3,CV_32F);
  cv::rgbd::depthTo3d(depth_in, K, depth_real_ref_raw);

  for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
  {
    cv::linemod::Match m = matches[i];

    inferDepth(depth_real_ref_raw, m);

    if (visited.insert(m.class_id).second)
    {
      ++classes_visited;

      // Draw matching template
      const std::vector<cv::linemod::Template>& templates = detector_.getTemplates(m.class_id, m.template_id);

      drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector_.getT(0));

      if (show_match_result)
      {
        std::cout << "Similarity: " << m.similarity << ", x: " << m.x << ", y: "
                  << m.y << ", class: " << m.class_id.c_str() << ", template: "
                  << m.template_id << std::endl;

        float distance = distances.at(m.class_id).at(m.template_id);
        std::cout << "distance " << distance << std::endl;
        // also could print T and R but that will be very verbose, so skip it
        // for now
      }
    }
  }

  if (show_match_result && matches.empty())
    printf("No matches found...\n");

  if (show_match_result)
    printf("------------------------------------------------------------\n");

  cv::imshow("normals", quantized_images[1]);
  cv::waitKey(1);
}

void Linemod::inferDepth(cv::Mat_<cv::Vec3f>& depth_real_ref_raw, cv::linemod::Match& match)
{
  cv::Matx33d R_match = Rs.at(match.class_id)[match.template_id].clone();
  cv::Vec3d T_match = Ts.at(match.class_id)[match.template_id].clone();
  float D_match = distances.at(match.class_id)[match.template_id];
  cv::Mat K_match = Ks.at(match.class_id)[match.template_id];

  cv::Mat mask;
  cv::Rect rect;
  cv::Matx33d R_temp(R_match.inv());
  cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));
  RendererIterator* it_r = renderer_iterators.at(match.class_id);
  cv::Mat depth_ref_;
  it_r->renderDepthOnly(depth_ref_, mask, rect, -T_match, up);

  cv::Mat_<cv::Vec3f> depth_real_model_raw;
  cv::depthTo3d(depth_ref_, K_match, depth_real_model_raw);
  //prepare the bounding box for the model and reference point clouds
  cv::Rect_<int> rect_model(0, 0, depth_real_model_raw.cols, depth_real_model_raw.rows);
  //prepare the bounding box for the reference point cloud: add the offset
  cv::Rect_<int> rect_ref(rect_model);
  rect_ref.x += match.x;
  rect_ref.y += match.y;

  rect_ref = rect_ref & cv::Rect(0, 0, depth_real_ref_raw.cols, depth_real_ref_raw.rows);
  if ((rect_ref.width < 5) || (rect_ref.height < 5))
  {
    std::cerr << "error: rect_ref is too small: " << rect_ref.width << "x" << rect_ref.height << std::endl;
  }
  //adjust both rectangles to be equal to the smallest among them
  if (rect_ref.width > rect_model.width)
    rect_ref.width = rect_model.width;
  if (rect_ref.height > rect_model.height)
    rect_ref.height = rect_model.height;
  if (rect_model.width > rect_ref.width)
    rect_model.width = rect_ref.width;
  if (rect_model.height > rect_ref.height)
    rect_model.height = rect_ref.height;

  //prepare the reference data: from the sensor : crop images
  cv::Mat_<cv::Vec3f> depth_real_ref = depth_real_ref_raw(rect_ref);
  //prepare the model data: from the match
  cv::Mat_<cv::Vec3f> depth_real_model = depth_real_model_raw(rect_model);

  //initialize the translation based on reference data
  cv::Vec3f T_crop = depth_real_ref(depth_real_ref.rows / 2.0f, depth_real_ref.cols / 2.0f);
  //add the object's depth
  T_crop(2) += D_match;

  if (!cv::checkRange(T_crop))
  {
    std::cerr << "error: T_crop out of range" << std::endl;
  }
  cv::Vec3f T_real_icp(T_crop);

  //initialize the rotation based on model data
  if (!cv::checkRange(R_match))
  {
    std::cerr << "error: R_match out of range" << std::endl;
  }
  cv::Matx33f R_real_icp(R_match);

  //get the point clouds (for both reference and model)
  std::vector<cv::Vec3f> pts_real_model_temp;
  std::vector<cv::Vec3f> pts_real_ref_temp;
  float px_ratio_missing = matToVec(depth_real_ref, depth_real_model, pts_real_ref_temp, pts_real_model_temp);
  float px_match_min_ = 0.25;
  if (px_ratio_missing > (1.0f-px_match_min_))
  {
    std::cerr << "ratio missing too high!" << std::endl;
  }

  //perform the first approximate ICP
  float px_ratio_match_inliers = 0.0f;
  float icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp, T_real_icp, px_ratio_match_inliers, 1);
  //reject the match if the icp distance is too big

  float icp_dist_min_ = 0.05;
  if (icp_dist > icp_dist_min_)
  {
    std::cerr << "icp dist too high!" << std::endl;
  }

  //perform a finer ICP
  icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp, T_real_icp, px_ratio_match_inliers, 2);

  std::cout << "final R_real_icp:" << R_real_icp << std::endl;
  std::cout << "final T_real_icp:" << T_real_icp << std::endl;
}

void Linemod::drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
  static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < num_modalities; ++m)
  {
    // NOTE: Original demo recalculated max response for each feature in the TxT
    // box around it and chose the display color based on that response. Here
    // the display color just depends on the modality.
    cv::Scalar color = COLORS[m];

    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
    }
  }
}
