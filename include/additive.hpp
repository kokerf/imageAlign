#ifndef _ADDITIVE_HPP_
#define _ADDITIVE_HPP_

#include <opencv2/opencv.hpp>
#include <list>
#include <string>

//! log_level 1: only log last, 0 : log each iteration
void forwardAdditiveImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect& omega, cv::Mat& affine, int log_level = 1, std::list<std::string> * log_str = nullptr);

#endif