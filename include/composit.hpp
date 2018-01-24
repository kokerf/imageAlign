#ifndef _COMPOSIT_HPP_
#define _COMPOSIT_HPP_ 

#include <opencv2/opencv.hpp>

void forwardCompositionalImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect omega, cv::Mat& affine, bool report = false);

#endif