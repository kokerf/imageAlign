#ifndef _VISIONKIT_HPP_
#define _VISIONKIT_HPP_ 

#include <opencv2/opencv.hpp>

void intAffine(cv::Mat& A, float a11, float a12, float a21, float a22, float tx, float ty);
void warpAffine(const cv::Mat& A, const cv::Mat& img_ref, cv::Mat& img_out);

float interpolateMat_8u(const cv::Mat& mat, float u, float v);

void gradient(const cv::Mat& img, cv::Mat& grad, int dx, int dy);

#endif