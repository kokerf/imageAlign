﻿#include <iostream>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "visionkit.hpp"
#include "invcomposit.hpp"

void inverseCompositionalImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect omega, cv::Mat& affine, int log_level, std::list<std::string> * log_str)
{
    const float EPS = 1E-5f; // Threshold value for termination criteria.
    const int MAX_ITER = 100;  // Maximum iteration count.

    const int cols = omega.width;
    const int rows = omega.height;

    std::cout << std::endl << "Start Inverse Compositional Algorithm!\n" << std::endl;

    clock_t start_time = clock();

    /*
     *  Pre-computation stage.
     */

    //! 1. [Step-3]Evaluate the gradient of the template
    //! the function gradient expansions the image with border then get gradient
    //! it is better to calculate the gradient of imgT then get the gradient of T
    cv::Mat T = imgT(omega).clone();
    cv::Mat gradTx, gradTy;
    gradient(T, gradTx, 1, 0);
    gradient(T, gradTy, 0, 1);

    cv::Mat jac;
    cv::Mat dxy;
    cv::Mat J;
    cv::Mat Jac_cache = cv::Mat::zeros(cols*rows, 6, CV_32F);
    cv::Mat H = cv::Mat::zeros(6, 6, CV_32FC1);

    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < cols; ++x)
        {
            //! 2. [Step-4]Evaluate the Jacobin ∂W/∂p at(x;0)
            jac = (cv::Mat_<float>(2, 6) << x, y, 0, 0, 1, 0, 0, 0, x, y, 0, 1);

            //! 3.[Step-5]Calculate steepest descent image ▽T*∂W/∂p
            dxy = (cv::Mat_<float>(1, 2) << gradTx.at<float>(y, x), gradTy.at<float>(y, x));
            J = dxy*jac;
            
            J.copyTo(Jac_cache.row(x + y*cols));
        }
    }
    clock_t percompute_time = clock();
    double per_time = (double)(percompute_time - start_time) / CLOCKS_PER_SEC;

    //! 4. [Step-6]Calculate Hessian Matrix H = ∑x[▽T*∂W/∂p]^T*[▽T*∂W/∂p]
    H = Jac_cache.t()*Jac_cache;

    //! 5. Get Invert Hessian Matrix
    cv::Mat Hinv = H.inv();

    /*
     *   Iteration stage.
     */

    cv::Mat A = cv::Mat::eye(3, 3, CV_32FC1);

    int iter = 0;
    float mean_error = 0;

    while(iter < MAX_ITER)
    {
        iter++;

        mean_error = 0;

        cv::Mat IW;
        cv::Mat Jres = cv::Mat::zeros(6, 1, CV_32FC1);
        cv::Mat dp = cv::Mat::zeros(6, 1, CV_32FC1);
        cv::Mat dA;

        //! 6. [Step-1]Get the Warp Image of I: I(W(x;p))
        warpAffineback(imgI, IW, A, omega);

        for(int y = 0; y < rows; ++y)
        {
            uint8_t* pIW = IW.ptr<uint8_t>(y);
            uint8_t* pT = T.ptr<uint8_t>(y);
            for(int x = 0; x < cols; ++x)
            {
                //! 7. [Step-2]Compute the error image: Res = I(W(x;p)) - T(x)
                float res = pIW[x] * 1.0 - pT[x];

                mean_error += res*res;

                //! 8. [Step-7]Compute Jres = ∑x[▽T*∂W/∂p]^T*[I(W(x;p))-T(x)]
                cv::Mat JT = Jac_cache.row(x+y*cols).t();
                Jres += JT * res;
            }
        }

        mean_error /= rows*cols;

        //! 9. [Step-8]Compute Parameter Increment: dp = H^(-1) * Jres
        dp = Hinv * Jres;

        //! 10. [Step-9]Invert increment of Warp and update Warp: W(x,p) = W(x;p) * W(x;dp)^-1 
        float dA11 = dp.at<float>(0,0);
        float dA12 = dp.at<float>(1,0);
        float dA21 = dp.at<float>(2,0);
        float dA22 = dp.at<float>(3,0);
        float dtx = dp.at<float>(4,0);
        float dty = dp.at<float>(5,0);
        intAffine(dA, dA11 + 1, dA12, dA21, dA22 + 1, dtx, dty);
        A = A * dA.inv();

        if(log_level <= 0)
        {
            std::cout << "A:" << A << std::endl;
            std::cout << "Iter:" << iter << ", Mean Error:" << mean_error << std::endl;
        }

        if(log_str)
        {
            std::string log = std::to_string(iter) + ", " + std::to_string(mean_error);
            log_str->push_back(log);
        }

        if(fabs(dA11) < EPS && fabs(dA12) < EPS && fabs(dA21) < EPS && fabs(dA22) < EPS && fabs(dtx) < EPS && fabs(dty) < EPS)
        {break;}
    }
    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

    affine = A;

    if(log_level > 1)
        return;

    //! Print summary.
    std::cout << "===============================================" << std::endl;
    std::cout << "Algorithm: Inverse Compositional" << std::endl;
    std::cout << "A:" << std::endl << A << std::endl;
    std::cout << "Mean Error:" << mean_error << std::endl;
    std::cout << "Iteration:" << iter << std::endl;
    std::cout << "Per-compute Time:" << per_time << std::endl;
    std::cout << "Total Time:" << total_time << std::endl;
    std::cout << "===============================================" << std::endl;
}