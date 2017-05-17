#include "sparse_stereo.h"
#include <bitset>
#include <iostream>

SparseStereo::SparseStereo(const cv::MatStep &step, const int filterDist,
                           const int validDisparity, const int epipolarRange)
    : m_filterDist(filterDist), m_maxDisparity(validDisparity),
      m_epipolarRange(epipolarRange) {
  m_sampleOffsets_9x9.resize(16);
  m_sampleOffsets_9x9[0] = -4 * step[0];
  m_sampleOffsets_9x9[1] = -3 * step[0] - 2 * step[1];
  m_sampleOffsets_9x9[2] = -3 * step[0] + 2 * step[1];
  m_sampleOffsets_9x9[3] = -2 * step[0] - 3 * step[1];
  m_sampleOffsets_9x9[4] = -2 * step[0] - 2 * step[1];
  m_sampleOffsets_9x9[5] = -2 * step[0] + 2 * step[1];
  m_sampleOffsets_9x9[6] = -2 * step[0] + 3 * step[1];
  m_sampleOffsets_9x9[7] = -4 * step[1];
  m_sampleOffsets_9x9[8] = 4 * step[1];
  m_sampleOffsets_9x9[9] = 2 * step[0] - 3 * step[1];
  m_sampleOffsets_9x9[10] = 2 * step[0] - 2 * step[1];
  m_sampleOffsets_9x9[11] = 2 * step[0] + 2 * step[1];
  m_sampleOffsets_9x9[12] = 2 * step[0] + 3 * step[1];
  m_sampleOffsets_9x9[13] = 3 * step[0] - 2 * step[1];
  m_sampleOffsets_9x9[14] = 3 * step[0] + 2 * step[1];
  m_sampleOffsets_9x9[15] = 4 * step[0];
}

SparseStereo::~SparseStereo() {}

void SparseStereo::extractSparse(const cv::Mat &img,
                                 std::vector<cv::Point> &points,
                                 TransformData &result) {
  if (result.transfmImg.cols != img.cols || result.transfmImg.rows != img.rows)
    return;

  std::sort(points.begin(), points.end(),
            [](const cv::Point &a, const cv::Point &b) {
              return ((a.y == b.y && a.x < b.x) || (a.y < b.y));
            });

  std::vector<KpLite> _kps;
  _kps.reserve(points.size());

  SubVector initVal;
  initVal.isEmpty = true;
  result.rowBuckets.resize(img.rows, initVal);

  int newIdx = 0;
  // Reject points that are too close to the edge of the image
  for (size_t i = 0; i < points.size(); ++i) {
    cv::Point pixelLoc(static_cast<int>(points[i].x),
                       static_cast<int>(points[i].y));

    if (pixelLoc.x < 4 || pixelLoc.x > (img.cols - 4) || pixelLoc.y < 4 ||
        pixelLoc.y > (img.rows - 4)) {
      continue;
    }

    uint32_t *pixelResultLoc =
        result.transfmImg.unsafeAt(pixelLoc.x, pixelLoc.y);
    uchar *pixelSrcLoc =
        img.data + pixelLoc.y * img.step[0] + pixelLoc.x * img.step[1];

    for (size_t k = 0; k < m_sampleOffsets_9x9.size(); ++k) {
      // Check if the value at this location has already been calculated
      if (!(*(pixelResultLoc + m_sampleOffsets_9x9[k]) & 0xFF000000))
        continue;

      *(pixelResultLoc + m_sampleOffsets_9x9[k]) =
          transform9x9(img, pixelSrcLoc + m_sampleOffsets_9x9[k]);
    }

    _kps.push_back(KpLite(pixelLoc.x, pixelLoc.y, newIdx, i));

    if (result.rowBuckets[pixelLoc.y].isEmpty) // First keypoint in its row
    {
      result.rowBuckets[pixelLoc.y].isEmpty = false;
      result.rowBuckets[pixelLoc.y].start = _kps.begin() + newIdx;
      result.rowBuckets[pixelLoc.y].stop = _kps.begin() + newIdx;
    } else { // Last point(so far) in its row
      result.rowBuckets[pixelLoc.y].stop = _kps.begin() + newIdx;
    }

    ++newIdx;
  }
  // std::cout << result.kps.size() << std::endl;
  // std::cout << _kps.size() << std::endl;
  result.kps.swap(_kps);
  // result.kps = _kps;
  // std::cout << result.kps.size() << std::endl;
  // std::cout << _kps.size() << std::endl;
}

void SparseStereo::match(TransformData &tfmData1, TransformData &tfmData2,
                         std::vector<cv::DMatch> &matches) {
  matches.clear();
  // check that left and right images have the same dimensions
  if (tfmData1.transfmImg.rows != tfmData2.transfmImg.rows ||
      tfmData1.transfmImg.cols != tfmData2.transfmImg.cols) {
    // TODO throw an exception here.
    return;
  }

  matches.reserve(tfmData1.kps.size());

  // Find best match in img2 for each feature in img1, by Hamming Distance
  for (size_t i = 0; i < tfmData1.kps.size(); ++i) // For each left feature
  {
    // Create submat to give to findBestMatch.  Epipolar + Disparity
    // constraints.
    int x, y, width, height;

    int yMax; // Bottom of epipolarRegion

    x = tfmData1.kps[i].x - m_maxDisparity;
    y = tfmData1.kps[i].y - m_epipolarRange / 2;
    yMax = tfmData1.kps[i].y + m_epipolarRange / 2;

    if (x < 0) // TODO make dynamic with (correlation window size / 2 +
               // transform window size / 2)
      x = 0;
    if (y < 0)
      y = 0;
    if (yMax >= tfmData1.transfmImg.rows)
      yMax = tfmData1.transfmImg.rows - 1;

    width = tfmData1.kps[i].x - x + 1;
    height = yMax - y + 1;

    std::vector<SubVector> potentialMatches;
    CensusMat searchRegion(tfmData2.transfmImg, x, y, width, height);

    loadDescriptors(tfmData2.rowBuckets, searchRegion, potentialMatches);

    // Find best match in the submat
    if (potentialMatches.empty())
      continue;

    cv::DMatch match = matchSparse(tfmData1.transfmImg, tfmData2.transfmImg,
                                   tfmData1.kps[i], potentialMatches);

    if (match.distance < m_filterDist && match.distance >= 0)
      matches.push_back(match);
  } // end for loop
}

uint16_t SparseStereo::transform9x9(const cv::Mat &img, uchar *pixelLoc) {
  uint16_t result = 0;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[0]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[1]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[2]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[3]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[4]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[5]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[6]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[7]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[8]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[9]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[10]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[11]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[12]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[13]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[14]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[15]) > *pixelLoc));

  // std::cout << std::bitset<sizeof(int)*8>( result ) << std::endl;
  return result;
}

void SparseStereo::loadDescriptors(const std::vector<SubVector> &descriptors,
                                   const CensusMat &resultImg,
                                   std::vector<SubVector> &resultVector) {
  /* NOTE: If no subpixel interpolation has been done to the
     keypoints/descriptors, then they should remain
     sorted, as they come from FAST.  Otherwise this will fail at boundaries
  */
  resultVector.clear();

  int lastRow = resultImg.rows + resultImg.offset.y;

  // For each row(element) of descriptors
  for (int i = resultImg.offset.y; i < lastRow; ++i) {
    if (descriptors[i].isEmpty) // make sure that this row has descriptors in it
      continue;
    std::vector<KpLite>::iterator k = descriptors[i].start;

    do {
      SubVector tmp;
      if (resultImg.isWithin(k->x, k->y)) {
        if (tmp.isEmpty) {
          tmp.isEmpty = false;
          tmp.start = k;
          tmp.stop = k;
          resultVector.push_back(tmp);
        } else {
          resultVector.back().stop = k;
        }
      }
    } while ((k++) != descriptors[i].stop);
  } // end for loop (rows)
}

uint32_t SparseStereo::computeSHD(CensusMat &transformedL,
                                  CensusMat &transformedR, const KpLite descL,
                                  const KpLite descR) {
  /* SSE VERSION (TODO - with memory reorganization)
    descriptors in contiguous memory

    Load 8x LDesc into a register  mm_load_128si(*LDesc)
    Load 8x RDesc into a register  mm_load_128si(*RDesc)

    XOR the registers
    PSHUFB-lookup the 4-bit table
    Add results
    return value
    */

  uint32_t result = 0;

  uint32_t *pixelOffsetL = transformedL.unsafeAt(descL.x, descL.y);
  uint32_t *pixelOffsetR = transformedR.unsafeAt(descR.x, descR.y);
  for (size_t i = 0; i < m_sampleOffsets_9x9.size(); ++i) {
    result +=
        SparseStereo::calcHammingDist(*(pixelOffsetL + m_sampleOffsets_9x9[i]),
                                      *(pixelOffsetR + m_sampleOffsets_9x9[i]));
  }

  return result;
}

cv::DMatch SparseStereo::matchSparse(CensusMat &imgToMatch,
                                     CensusMat &imgPotMatches,
                                     const KpLite &toMatch,
                                     const std::vector<SubVector> &potMatches) {
  // TODO ensure only unique matches
  // queryIdx is the right img, train idx is the left img
  int trainIdx = toMatch.kpIdx;
  int queryIdx = potMatches[0].start->kpIdx; // Should always be overwritten
  int bestDist = 512; // greater than any possible distance, so it will be
                      // reset to a real distance immediately
  int dist = bestDist;

  for (auto row : potMatches) {
    if (!row.isEmpty) {
      std::vector<KpLite>::iterator iter = row.start;
      do {
        // Then calculate the Hamming Distance between it and the toMatch
        dist = computeSHD(imgToMatch, imgPotMatches, toMatch, *iter);

        if (dist < bestDist) {
          bestDist = static_cast<int>(dist);
          queryIdx = iter->kpIdx;
          if (bestDist == 0)
            break; // simple WTA
        }
      } while ((iter++) != row.stop);
    }
  }

  cv::DMatch match(trainIdx, queryIdx, static_cast<float>(bestDist));
  return match;
}

void SparseStereo::prepareRightPointList(const std::vector<cv::Point> &pointsL,
                                         std::vector<cv::Point> &pointsR,
                                         const int width, const int height,
                                         const int maxDisparity,
                                         const int epipolarRange) {
  if (!pointsR.empty()) {
    pointsR.clear();
  }

  cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
  for (auto p : pointsL) {
    int x = p.x - maxDisparity;
    int y = p.y - epipolarRange / 2;
    int yMax = p.y + epipolarRange / 2;
    // check range
    if (x < 0) {
      x = 0;
    }
    if (y < 0) {
      y = 0;
    }
    if (yMax >= height) {
      yMax = height - 1;
    }
    // put back the potential points
    for (int i = y; i <= yMax; ++i) {
      for (int j = x; j <= p.x; ++j) {
        if (img.at<unsigned char>(i, j) == 0) {
          img.at<unsigned char>(i, j) = 1;
          pointsR.push_back(cv::Point(j, i));
        }
      }
    }
  }
}

void SparseStereo::update(std::vector<cv::Point> &pointsL,
                          std::vector<cv::Point> &pointsR, const cv::Mat &imgL,
                          const cv::Mat &imgR,
                          std::vector<cv::DMatch> &matches) {
  prepareRightPointList(pointsL, pointsR, imgL.cols, imgL.rows, m_maxDisparity,
                        m_epipolarRange);

  static TransformData transfmData1(imgL.rows, imgL.cols);
  static TransformData transfmData2(imgR.rows, imgR.cols);

  transfmData1.reset();
  transfmData2.reset();

  extractSparse(imgL, pointsL, transfmData1);
  extractSparse(imgR, pointsR, transfmData2);

  match(transfmData1, transfmData2, matches);
}

void SparseStereo::update(std::vector<cv::Point> &pointsL, const cv::Mat &imgL,
                          const cv::Mat &imgR, std::vector<int> &disparity) {
  std::vector<cv::Point> pointsR;
  std::vector<cv::DMatch> matches;
  prepareRightPointList(pointsL, pointsR, imgL.cols, imgL.rows, m_maxDisparity,
                        m_epipolarRange);

  // match
  static TransformData transfmData1(imgL.rows, imgL.cols);
  static TransformData transfmData2(imgR.rows, imgR.cols);

  transfmData1.reset();
  transfmData2.reset();

  extractSparse(imgL, pointsL, transfmData1);
  extractSparse(imgR, pointsR, transfmData2);

  match(transfmData1, transfmData2, matches);
  for (auto m : matches) {
    disparity.push_back((pointsL[m.queryIdx].x - pointsR[m.trainIdx].x));
  }
}
