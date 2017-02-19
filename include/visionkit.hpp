#ifndef _VISIONKIT_HPP_
#define _VISIONKIT_HPP_ 

#include <opencv2/opencv.hpp>

void intAffine(cv::Mat& A, float a11, float a12, float a21, float a22, float tx, float ty);
void warpAffine(const cv::Point2d& pt_src,cv::Point2d& pt_dest,const cv::Mat& A,cv::Point2d O);
void warpAffine(const cv::Mat& img_ref, cv::Mat& img_out, const cv::Mat& A, cv::Point2d O);
void warpAffine(const cv::Mat& img_ref, cv::Mat& img_out, const cv::Mat& A, cv::Rect& omega, cv::Point2d O = cv::Point2d(0, 0), bool flag = false);
void warpAffineback(const cv::Mat& img_ref, cv::Mat& img_out, const cv::Mat& A, cv::Rect& omega, cv::Point2d O = cv::Point2d(0, 0), bool flag = false);
void warpAffine_float(const cv::Mat& img_ref, cv::Mat& img_out, const cv::Mat& A, cv::Rect& omega, cv::Point2d O = cv::Point2d(0, 0), bool flag = false);
void warpAffineback_float(const cv::Mat& img_ref, cv::Mat& img_out, const cv::Mat& A, cv::Rect& omega, cv::Point2d O = cv::Point2d(0, 0), bool flag = false);

void gradient(const cv::Mat& img, cv::Mat& grad, int dx, int dy);

float interpolateMat_8u(const cv::Mat& mat, float u, float v);
float interpolateMat_32f(const cv::Mat& mat, float u, float v);

#endif