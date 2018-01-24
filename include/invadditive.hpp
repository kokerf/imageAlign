#ifndef _INVADDITIVE_HPP_
#define _INVADDITIVE_HPP_

#include <opencv2/opencv.hpp>

void inverseAdditiveImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect& omega, cv::Mat& affine, bool report = false);

#endif