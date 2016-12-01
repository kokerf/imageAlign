#include <iostream>
#include <opencv2/opencv.hpp>

#include "visionkit.hpp"
#include "invcomp.hpp"

int main(int argc, char const *argv[])
{
    cv::Mat origin,image,templet;

    origin = cv::imread("../data/photo.jpg");

    cvtColor(origin, image, CV_BGR2GRAY);

    std::cout<<"img:"<<image.size()<<std::endl;

    cv::Rect omega = cv::Rect(110, 100, 200, 150);
    image(omega).copyTo(templet);
    cv::imshow("templet",templet);

    cv::Mat A;
    intAffine(A,1.1,-0.7,0.1,1.1,0,0);
    cv::Mat imgAff;
    imgAff.create(image.size(),CV_8U);
    warpAffine(A,image,imgAff);

    std::cout<<"A:"<<A<<std::endl;

    cv::Mat I;
    imgAff(omega).copyTo(I);

    cv::imshow("I",I);
    cv::waitKey(0);

    inverseCompositionalAlign(templet,I);

    cv::waitKey(0);

    return 0;
}