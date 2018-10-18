#ifndef _IMVCOMPOSIT_HPP_
#define _IMVCOMPOSIT_HPP_

#include <list>
#include <string>
#include <opencv2/opencv.hpp>

void inverseCompositionalImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect omega, cv::Mat& affine, int log_level = 1, std::list<std::string> * log_str = nullptr);

#endif