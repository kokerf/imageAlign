#include <iostream>
#include <opencv2/opencv.hpp>

#include "visionkit.hpp"
#include "invcomp.hpp"
#include "additive.hpp"

int main(int argc, char const *argv[])
{
    //! Creat images
    cv::Mat origin,image,imgAff;
    cv::Mat T,I;
    cv::Mat A;

    //! Load origin image
    origin = cv::imread("../data/photo.jpg");
    image.create(origin.size(),CV_8U);
    imgAff.create(origin.size(),CV_8U);
    //! Convert to gray image
    cvtColor(origin, image, CV_BGR2GRAY);

    std::cout<<"img:"<<origin.size()<<std::endl;

    //! Set Affine Model
	cv::Rect omega = cv::Rect(110, 100, 200, 150);
	intAffine(A, 1.0, 0.3, 0, 1.0, 0, 0);
    //intAffine(A,1.2,-0.2,-0.1,1.2,0,0);
	cv::Point2d O(omega.x + 0.5*omega.width, omega.y + 0.5*omega.height);
    warpAffine(A,image,imgAff,O);
    std::cout<<"A:"<<A<<std::endl;

    //! Get image I form gary image, T from imgAff
    //cv::Rect omega = cv::Rect(110, 100, 200, 150);
    image.copyTo(T);
    imgAff.copyTo(I);

    //! Show templet & image
    cv::imshow("templet",T);
    cv::imshow("image",I);
    cv::waitKey(0);

    //! Inverse Compositional Image Alignment 
	additiveImageAlign(T,I,omega);

    cv::waitKey(0);

    return 0;
}