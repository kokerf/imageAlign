#ifndef _ADDITIVE_HPP_
#define _ADDITIVE_HPP_

#include <opencv2/opencv.hpp>

void forwardAdditiveImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect& omega, cv::Mat& affine, bool report = false);

#endif