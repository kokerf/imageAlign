#include <iostream>
#include <opencv2/opencv.hpp>

#include "visionkit.hpp"
#include "invadditive.hpp"

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
    intAffine(A, 1.2, 0.3, 0, 1.0, 0, 0);
    cv::Point2d O(omega.x + 0.5*omega.width, omega.y + 0.5*omega.height);
    warpAffine(image, imgAff, A, O);

    //! Get image I form gary image, T from imgAff
    image.copyTo(I);
    imgAff.copyTo(T);

    //! Show templet & image
	cv::rectangle(image, omega, cv::Scalar(255, 0, 0));
    cv::imshow("templet", imgAff(omega));
    cv::imshow("image", image);
    cv::waitKey(0);

    //! The Affine Matrix
    std::cout << std::endl << "A:" << std::endl << A << std::endl;

    //! Forward Compositional Image Alignment
    inverseAdditiveImageAlign(T, I, omega);

    cv::waitKey(0);

    return 0;
}