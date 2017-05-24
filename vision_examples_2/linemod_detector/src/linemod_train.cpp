/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

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

// using ecto::tendrils;
// using ecto::spore;

// namespace ecto_linemod
// {
  struct Trainer
  {

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

    int param_n_points_ = 150;
    int param_angle_step_ = 10;
    double param_radius_min_ = 0.6;
    double param_radius_max_ = 1.1;
    double param_radius_step_ = 0.4;
    int param_width_ = 640;
    int param_height_ = 480;
    double param_near_ = 0.1;
    double param_far_ = 1000.0;
    double param_focal_length_x_ = 525;
    double param_focal_length_y_ = 525;

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

    void writeLinemod(const std::string& filename)
    {
      cv::FileStorage fs(filename, cv::FileStorage::WRITE);
      detector_.write(fs);

      std::cout << "total # of templates: " << detector_.numTemplates() << std::endl;
      std::cout << "# of stored translations: " << Ts_.size() << std::endl;

      std::vector<cv::String> ids = detector_.classIds();
      fs << "classes" << "[";
      int template_counter = 0;
      for (int i = 0; i < (int)ids.size(); ++i)
      {
        fs << "{";

        fs << "class_id" << detector_.classIds()[i];
        fs << "modalities" << "[:";
        for (size_t i = 0; i < detector_.getModalities().size(); ++i)
          fs << detector_.getModalities()[i]->name();
        fs << "]"; // modalities
        fs << "pyramid_levels" << detector_.pyramidLevels();
        fs << "template_pyramids" << "[";
        for (size_t k = 0; k < detector_.numTemplates(ids[i]); ++k)
        {
          const auto& tp = detector_.getTemplates(ids[i], k);
          fs << "{";
          fs << "template_id" << int(k); //TODO is this cast correct? won't be good if rolls over...

          // detector_.writeClass(ids[i], fs);
          fs << "distance" << distances_[template_counter];
          fs << "T" << Ts_[template_counter];
          fs << "R" << Rs_[template_counter];
          template_counter++;

          fs << "templates" << "[";
          for (size_t j = 0; j < tp.size(); ++j)
          {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
          }
          fs << "]"; // templates
          fs << "}"; // current pyramid
        }
        fs << "]"; // pyramids

        fs << "}"; // current class
      }
      fs << "]"; // classes
    }

    void train(std::string filename)
    {
      // Get the document for the object_id_ from the DB
      // object_recognition_core::db::ObjectDbPtr db =
      //     object_recognition_core::db::ObjectDbParameters(*json_db_).generateDb();
      // object_recognition_core::db::Documents documents =
      //     object_recognition_core::db::ModelDocuments(db,
      //         std::vector<object_recognition_core::db::ObjectId>(1, *object_id_),
      //         "mesh");
      // if (documents.empty()) {
      //   std::cerr << "Skipping object id \"" << *object_id_ << "\" : no mesh in the DB" << std::endl;
      //   return ecto::OK;
      // }

      // Get the list of _attachments and figure out the original one
      // object_recognition_core::db::Document document = documents[0];
      // std::vector<std::string> attachments_names = document.attachment_names();
      // std::string mesh_path;
      std::string mesh_path = filename;
      // std::vector<std::string> possible_names(2);
      // possible_names[0] = "original";
      // possible_names[1] = "mesh";
      // for (size_t i = 0; i < possible_names.size() && mesh_path.empty(); ++i) {
      //   BOOST_FOREACH(const std::string& attachment_name, attachments_names) {
      //     if (attachment_name.find(possible_names[i]) != 0)
      //       continue;
      //     // Create a temporary file
      //     char mesh_path_tmp[L_tmpnam];
      //     tmpnam(mesh_path_tmp);
      //     mesh_path = std::string(mesh_path_tmp) + attachment_name.substr(possible_names[i].size());

      //     // Load the mesh and save it to the temporary file
      //     std::ofstream mesh_file;
      //     mesh_file.open(mesh_path.c_str());
      //     document.get_attachment_stream(attachment_name, mesh_file);
      //     mesh_file.close();
      //   }
      // }

      cv::Ptr<cv::linemod::Detector> detector_ptr = cv::linemod::getDefaultLINEMOD();
      detector_ = *detector_ptr;

      // Define the display
      //assign the parameters of the renderer
      renderer_n_points_ = param_n_points_;
      renderer_angle_step_ = param_angle_step_;
      renderer_radius_min_ = param_radius_min_;
      renderer_radius_max_ = param_radius_max_;
      renderer_radius_step_ = param_radius_step_;
      renderer_width_ = param_width_;
      renderer_height_ = param_height_;
      renderer_near_ = param_near_;
      renderer_far_ = param_far_;
      renderer_focal_length_x_ = param_focal_length_x_;
      renderer_focal_length_y_ = param_focal_length_y_;

      // the model name can be specified on the command line.
      if (mesh_path.empty())
      {
        std::remove(mesh_path.c_str());
        std::cerr << "The mesh path is empty for the object id \"" << object_id_<< std::endl;
        return;
      }

      Renderer3d renderer = Renderer3d(mesh_path);
      renderer.set_parameters(renderer_width_, renderer_height_, renderer_focal_length_x_,
                              renderer_focal_length_y_, renderer_near_, renderer_far_);

      RendererIterator renderer_iterator = RendererIterator(&renderer, renderer_n_points_);
      //set the RendererIterator parameters
      renderer_iterator.angle_step_ = renderer_angle_step_;
      renderer_iterator.radius_min_ = float(renderer_radius_min_);
      renderer_iterator.radius_max_ = float(renderer_radius_max_);
      renderer_iterator.radius_step_ = float(renderer_radius_step_);

      cv::Mat image, depth, mask;
      cv::Matx33d R;
      cv::Vec3d T;
      cv::Matx33f K;
      for (size_t i = 0; !renderer_iterator.isDone(); ++i, ++renderer_iterator)
      {
        std::stringstream status;
        status << "Loading images " << (i+1) << "/"
            << renderer_iterator.n_templates();
        std::cout << status.str();

        cv::Rect rect;
        renderer_iterator.render(image, depth, mask, rect);

        R = renderer_iterator.R_obj();
        T = renderer_iterator.T();
        float distance = renderer_iterator.D_obj() - float(depth.at<ushort>(depth.rows/2.0f, depth.cols/2.0f)/1000.0f);
        K = cv::Matx33f(float(renderer_focal_length_x_), 0.0f, float(rect.width)/2.0f, 0.0f, float(renderer_focal_length_y_), float(rect.height)/2.0f, 0.0f, 0.0f, 1.0f);

        std::vector<cv::Mat> sources(2);
        sources[0] = image;
        sources[1] = depth;

        // Display the rendered image
        if (visualize_)
        {
          cv::namedWindow("Rendering");
          if (!image.empty()) {
            cv::imshow("Rendering", image);
            cv::waitKey(1);
          }
        }

        int template_in = detector_.addTemplate(sources, "object1", mask);
        if (template_in == -1)
        {
          // Delete the status
          for (size_t j = 0; j < status.str().size(); ++j)
            std::cout << '\b';
          continue;
        }

        // Also store the pose of each template
        Rs_.push_back(cv::Mat(R));
        Ts_.push_back(cv::Mat(T));
        distances_.push_back(distance);
        Ks_.push_back(cv::Mat(K));

        // Delete the status
        for (size_t j = 0; j < status.str().size(); ++j)
          std::cout << '\b';
      }
      return;
    }
  };
// } // namespace ecto_linemod

// ECTO_CELL(ecto_linemod, ecto_linemod::Trainer, "Trainer", "Train the LINE-MOD object detection algorithm.")

int main(int argc, char** argv)
{
  Trainer t;
  t.train("/home/adam/osrf/g7/src/adamfiles/testobj/mug.dae");

  // save the data in a form easily readable by linemod.
  // For now we're using a flat file representation, but we could be using
  // a database, like ORK does. This is just more straightforward
  t.writeLinemod("linemod_ros2.yml");
}