#include "linemod_detector/linemod.hpp"

#include <iterator>
#include <cstdio>
#include <iostream>
#include <set>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


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

      cv::Mat T_new, R_new;

      cv::FileNodeIterator fni = (*tps_it)["T"].begin();
      fni >> T_new;
      Ts[class_id].push_back(T_new);

      fni = (*tps_it)["R"].begin();
      fni >> R_new;
      Rs[class_id].push_back(R_new);

      distances[class_id].push_back((*tps_it)["distance"]);
    }
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

  // Perform matching
  std::vector<cv::linemod::Match> matches;
  std::vector<cv::String> class_ids;
  std::vector<cv::Mat> quantized_images;
  detector_.match(sources, (float)matching_threshold, matches, class_ids, quantized_images);

  int classes_visited = 0;
  std::set<std::string> visited;

  for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
  {
    cv::linemod::Match m = matches[i];
    // TODO: infer depth

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
