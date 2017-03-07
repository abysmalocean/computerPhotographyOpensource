/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <opencv2/opencv_modules.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include <stdio.h>

#ifndef HAVE_OPENCV_NONFREE

int main(int, char**)
{
    printf("The sample requires nonfree module that is not available in your OpenCV distribution.\n");
    return -1;
}

#else

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include <sys/time.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;
void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }
  GpuMat img_1;
  GpuMat img_2;
  img_1.upload(imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE ));
  img_2.upload(imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE ));

  if( !img_1.data || !img_2.data )
  { printf(" --(!) Error reading images \n"); return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;
  struct timeval start, end;
  long seconds, useconds;
  gettimeofday(&start, NULL);

  //SurfFeatureDetector detector( minHessian );
  SURF_GPU surf;
  GpuMat keypoints1GPU ;
  GpuMat descriptors1GPU ;
  GpuMat keypoints2GPU ;
  GpuMat descriptors2GPU ;
  surf(img_1, GpuMat(), keypoints1GPU, descriptors1GPU);
  surf(img_2, GpuMat(), keypoints2GPU, descriptors2GPU);

  Mat descriptorsMat2;
  Mat descriptorsMat1;
  vector<float> descriptors1;
  vector<float> descriptors2;
  descriptors2GPU.download(descriptorsMat2);
  surf.downloadDescriptors(descriptors2GPU, descriptors2);
  descriptors1GPU.download(descriptorsMat1);
  surf.downloadDescriptors(descriptors1GPU, descriptors1);

  //-- Step 2: Calculate descriptors (feature vectors)
  std::vector<DMatch> matches;
  BFMatcher_GPU matcher(NORM_L2);
  Mat descriptors_1, descriptors_2;
  GpuMat img1;
  GpuMat img2;
  img2.upload(descriptors_2);
  img1.upload(descriptors_1);
  GpuMat trainIdx, distance;
  matcher.matchSingle(img1, img2, trainIdx, distance);
  BFMatcher_GPU::matchDownload(trainIdx, distance, matches);
  //-- Show detected matches
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;

  if (useconds < 0) {
    seconds -= 1;
  }
  long total_micro_seconds = (seconds * 1000000) + abs(useconds);
  printf(
          "This Program is the MultiGUP Version  \n");
  printf("total micro seconds is ----->[%ld]\n", total_micro_seconds);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ printf(" Usage: ./SURF_FlannMatcher <img1> <img2>\n"); }

#endif
