#include "linemod.hpp"

#include <iterator>
#include <cstdio>
#include <iostream>
#include <set>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

// Functions to store detector and templates in single XML/YAML file
cv::linemod::Detector Linemod::readLinemod(const std::string& filename)
{
  cv::linemod::Detector detector;
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector.read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector.readClass(*i);

  return detector;
}

Linemod::Linemod() :
  detector_(readLinemod("linemod_templates.yml"))
{
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

      if (show_match_result)
      {
        printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
               m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
      }

      // Draw matching template
      const std::vector<cv::linemod::Template>& templates = detector_.getTemplates(m.class_id, m.template_id);

      drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector_.getT(0));
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
