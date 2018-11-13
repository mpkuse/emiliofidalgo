/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>

#include "ibow-lcd/lcdetector.h"

void getFilenames(const std::string& directory,
                  std::vector<std::string>* filenames) {
    using namespace boost::filesystem;

    filenames->clear();
    path dir(directory);

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".jpg" ||
            ext == ".ppm" || ext == ".jpeg") {
            filenames->push_back(it->string());
        }
    }
}


#include <json.hpp>
using json = nlohmann::json;
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
using namespace Eigen;

#include "utils/RawFileIO.h"
#include "utils/ElapsedTime.h"

// const std::string BASE = "/Bulk_Data/_tmp_cerebro/bb4_long_lab_traj/";
const std::string BASE = "/Bulk_Data/_tmp_cerebro/bb4_multiple_loops_in_lab/";

void mpkuse_getFilenames( std::vector<std::string>& filenames )
{
    std::string json_fname = BASE+"/log.json";
    std::cout <<  "Open file: " << json_fname <<  std::endl;
    std::ifstream json_fileptr(json_fname);
    json json_obj;
    json_fileptr >> json_obj;
    std::cout << "Successfully opened file "<< json_fname << std::endl;


    // Collect all poses
    std::cout << "json_obj[\"DataNodes\"].size()  " << json_obj["DataNodes"].size() << std::endl;
    // std::vector< std::string > list_of_imfiles;
    std::vector< Matrix4d > list_of_vio_poses;
    // int n_max = 1500;
    int n_max = json_obj["DataNodes"].size();
    for( int i=0 ; i<n_max ; i++ ) {
        if(  json_obj["DataNodes"][i]["isPoseAvailable"] == 1 ) {
            std::cout << i << " isPoseAvailable: " <<  json_obj["DataNodes"][i]["isPoseAvailable"] << std::endl;
            // Matrix4d w_T_c_from_file;
            // RawFileIO::read_eigen_matrix( BASE+"/"+to_string(i)+".wTc", w_T_c_from_file );
            // cout << "w_T_c_from_file "<< w_T_c_from_file << endl;


            Matrix4d w_T_c;
            std::vector<double> raa = json_obj["DataNodes"][i]["w_T_c"]["data"];
            RawFileIO::read_eigen_matrix( raa, w_T_c );//loads as row-major
            cout << "w_T_c "<< w_T_c << endl;

            std::string im_fname = BASE+"/"+std::to_string(i)+".jpg";
            std::cout << i << " " << im_fname << std::endl;
            filenames.push_back( im_fname );
            // list_of_vio_poses.push_back( w_T_c );
        }
    }
    cout << "Done reading "<< json_fname << endl;
}

int main(int argc, char** argv) {
  // Creating feature detector and descriptor
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(500);  // Default params

  // Loading image filenames
  std::vector<std::string> filenames;
  // getFilenames(argv[1], &filenames);
  mpkuse_getFilenames( filenames );
  ElapsedTime timer;
  // return 0;
  unsigned nimages = filenames.size();

  json jsonout_obj;

#if 0
  // Creating the loop closure detector object
  ibow_lcd::LCDetectorParams params;  // Assign desired parameters
  // params.purge_descriptors = false;
  // params.min_feat_apps = 5;
  ibow_lcd::LCDetector lcdet(params);

  // Processing the sequence of images
  for (unsigned i = 0; i < nimages; i++) {
    // Processing image i
    std::cout << "--- Processing image " << i << " of " << nimages << ": " << filenames[i] << std::endl;

    // Loading and describing the image
    timer.tic();
    cv::Mat img = cv::imread(filenames[i]);
    std::cout << "cv::imread in (ms) " << timer.toc_milli() << std::endl;

    timer.tic();
    std::vector<cv::KeyPoint> kps;
    detector->detect(img, kps);
    cv::Mat dscs;
    detector->compute(img, kps, dscs);
    std::cout << "detector->detectAndCompute in (ms) " << timer.toc_milli() << std::endl;

    timer.tic();
    ibow_lcd::LCDetectorResult result;
    lcdet.process(i, kps, dscs, &result);
    std::cout << "lcdet.process (ms) " << timer.toc_milli() << std::endl;

    switch (result.status) {
      case ibow_lcd::LC_DETECTED:
        std::cout << "--- Loop detected!!!: " << result.train_id <<
                     " with " << result.inliers << " inliers" << std::endl;
        {
        json _cur_obk;
        _cur_obk["a"] = (int) i;
        _cur_obk["b"] = (int) result.train_id;
        _cur_obk["inliers"] = result.inliers;
        _cur_obk["fname_a"] = filenames[i];
        _cur_obk["fname_b"] = filenames[result.train_id];
        jsonout_obj.push_back( _cur_obk );
        }

        break;
      case ibow_lcd::LC_NOT_DETECTED:
        std::cout << "No loop found" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_IMAGES:
        std::cout << "Not enough images to found a loop" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_ISLANDS:
        std::cout << "Not enough islands to found a loop" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_INLIERS:
        std::cout << "Not enough inliers" << std::endl;
        break;
      case ibow_lcd::LC_TRANSITION:
        std::cout << "Transitional loop closure" << std::endl;
        break;
      default:
        std::cout << "No status information" << std::endl;
        break;
    }
  }

#else
  // Read json file
  std::string json_out_fname = BASE+"/loopcandidates_ibow_lcd.json";
  std::cout << "Open file: "<< json_out_fname << endl;
  std::ifstream fptr(json_out_fname);
  fptr >> jsonout_obj;

  for( int i=0 ; i<jsonout_obj.size() ; i++ ) {
      cout << i << " of " << jsonout_obj.size() << " ] ";
      cout << jsonout_obj[i]["a"] << "<" << jsonout_obj[i]["inliers"] << ">" << jsonout_obj[i]["a"] << endl;
      if( jsonout_obj[i]["inliers"] < 20 )
        continue;
      string fname_a = jsonout_obj[i]["fname_a"] ;
      string fname_b = jsonout_obj[i]["fname_b"] ;
      cv::Mat image_a = cv::imread( fname_a);
      cv::Mat image_b = cv::imread( fname_b );
      cv::imshow( "image_a", image_a );
      cv::imshow( "image_b", image_b );
      char key = cv::waitKey( 0 );
      if( key == 'q' )
        break;
  }
#endif


#if 1
  // Print out the loop candidates
  std::string json_out_fname_org = BASE+"/loopcandidates_ibow_lcd.json";
  std::cout << "Write File: " << json_out_fname_org << endl;
  std::ofstream out( json_out_fname_org );
  out << jsonout_obj.dump(4);
  out.close();
#endif

  return 0;
}
