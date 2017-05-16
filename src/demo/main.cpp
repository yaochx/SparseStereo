#include "sparse_stereo.h"
#include <bitset>
#include <iostream>
#include <stdint.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

static void help(char **argv) {
  std::cout << "\nUsage: " << argv[0]
            << "[path/to/image1] [path/to/image2] [Max Hamming Dist] [Max "
               "Disparity] [Epipolar Range]\n"
            << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    help(argv);
    return -1;
  }

  // Load images
  Mat imgL = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  if (!imgL.data) {
    std::cout << " --(!) Error reading image " << argv[1] << std::endl;
    return -1;
  }

  Mat imgR = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  if (!imgR.data) {
    std::cout << " --(!) Error reading image " << argv[2] << std::endl;
    return -1;
  }

  int maxDist = atoi(argv[3]);
  int maxDisparity = atoi(argv[4]);
  int epiRange = atoi(argv[5]);

  std::vector<cv::KeyPoint> keypointsL, keypointsR;
  std::vector<cv::Point> pointsL, pointsR;
  std::vector<cv::DMatch> matches;
  std::vector<int> disparity;
  SparseStereo census(imgL.step, maxDist, maxDisparity, epiRange);

  double t = (double)getTickCount();
  cv::FAST(imgL, keypointsL, 51, true);
  t = ((double)getTickCount() - t) / getTickFrequency();
  std::cout << "detection size Left: " << keypointsL.size() << std::endl;
  std::cout << "detection time [s]: " << t << std::endl;

  for (auto p : keypointsL) {
    pointsL.push_back(cv::Point(p.pt.x, p.pt.y));
  }
  
  // census.update(pointsL, imgL, imgR, disparity);

  t = (double)getTickCount();
  census.update(pointsL, pointsR, imgL, imgR, matches);
  t = ((double)getTickCount() - t) / getTickFrequency();

  std::cout << "matching time [s]: " << t << std::endl;
  std::cout << "Number of matches: " << matches.size() << std::endl;

  for (auto p : pointsR) {
    cv::KeyPoint kp;
    kp.pt.x = p.x;
    kp.pt.y = p.y;
    keypointsR.push_back(kp);
  }

  // Draw matches
  Mat imgMatch;
  std::vector<char> mask;
  // drawKeypoints(imgL, keypointsL, imgL, cv::Scalar::all(-1));
  // drawKeypoints(imgR, keypointsR, imgR, cv::Scalar::all(-1));
  drawMatches(imgL, keypointsL, imgR, keypointsR, matches, imgMatch,
              cv::Scalar::all(-1), cv::Scalar::all(-1), mask, 2);

  // namedWindow("matches", CV_WINDOW_KEEPRATIO);
  imshow("matches", imgMatch);
  waitKey(0);
}
