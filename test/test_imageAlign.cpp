#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "visionkit.hpp"
#include "additive.hpp"
#include "composit.hpp"
#include "invadditive.hpp"
#include "invcomposit.hpp"

int main(int argc, char const *argv[])
{
    //! Choose the type of algorithm
    int type = 0;
    std::string typeStr;
    if(argc <=1)
    {
        std::cout << "Usage: ./test_imageAlign algorithm_name (algorithm_name: FAIA, FCIA, IAIA, ICIA)" << std::endl;
        return 0;
    }
    else
    {
        typeStr = argv[1];
        if(typeStr == "FAIA"){
            type = 1;
        }
        else if(typeStr == "FCIA"){
            type = 2;
        }
        else if(typeStr == "IAIA"){
            type = 3;
        }
        else if(typeStr == "ICIA"){
            type = 4;
        }
        else{
            std::cout<<"No such algorithm_name:"<<typeStr<<"\nPlease choose: FAIA, FCIA, IAIA, ICIA"<<std::endl;
            return 0;
        }
    }

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
    intAffine(A, 1.0, 0.2, 0, 1.0, 0, 0);
    cv::Point2d O(omega.x + 0.5*omega.width, omega.y + 0.5*omega.height);
    warpAffine(image, imgAff, A, O);

    //! Get image T form gary image, I from imgAff
    image.copyTo(T);
    imgAff.copyTo(I);

    //! Show templet & image
    //! Draw the area of warped image
    cv::Point2d pt[4], wpt[4]; 
    pt[0].x = omega.x; pt[0].y = omega.y;
    pt[1].x = omega.x + omega.width; pt[1].y = omega.y;
    pt[2].x = omega.x + omega.width; pt[2].y = omega.y + omega.height;
    pt[3].x = omega.x; pt[3].y = omega.y + omega.height;
    for(int i = 0; i < 4; ++i)
    {
        warpAffine(pt[i], wpt[i], A, O);
        //std::cout<<"wpt"<<i<<":"<<wpt[i]<<std::endl;
    }
    cv::line(imgAff, wpt[0], wpt[1], cv::Scalar::all(255), 1, 8, 0 );
    cv::line(imgAff, wpt[1], wpt[2], cv::Scalar::all(255), 1, 8, 0 );
    cv::line(imgAff, wpt[2], wpt[3], cv::Scalar::all(255), 1, 8, 0 );
    cv::line(imgAff, wpt[3], wpt[0], cv::Scalar::all(255), 1, 8, 0 );

    cv::imshow("templet", image(omega));
    cv::imshow("image", imgAff);
    cv::waitKey(0);

    //! The Affine Matrix
    std::cout << std::endl << "A:" << std::endl << A << std::endl;

    cv::Mat A_estimate;

    std::list<std::string> logs;

    switch(type)
    {
        case 1: forwardAdditiveImageAlign(T, I, omega, A_estimate, 1, &logs); break;      //! Forward Additive Image Alignment Algorithm
        case 2: forwardCompositionalImageAlign(T, I, omega, A_estimate, 1, &logs); break; //! Forward Compositional Image Alignment Algorithm
        case 3: inverseAdditiveImageAlign(T, I, omega, A_estimate, 1, &logs); break;      //! Inverse Additive Image Alignment Algorithm
        case 4: inverseCompositionalImageAlign(T, I, omega, A_estimate, 1, &logs); break; //! Inverse Compositional Image Alignment Algorithm
        default: break;
    }

    cv::Mat IW;
    warpAffineback(I, IW, A_estimate, omega);
    cv::imshow("warp back", IW);

    cv::waitKey(0);

    std::ofstream fout(typeStr+".txt");
    std::for_each(logs.begin(), logs.end(), [&](const std::string &s){fout << s << std::endl;});
    fout.close();

    return 0;
}