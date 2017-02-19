#include <iostream>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "visionkit.hpp"
#include "invadditive.hpp"

//for warp affine, we evaluate jacobian
// x' = (1+a11)*x+a12*y+tx;
// y' = a21*x+(1+a22)*y+ty;
// p = (a11, a12, a21, a22, tx, ty)
// 6 Dof
// Jp = | x y 0 0 1 0|
//      | 0 0 x y 0 1|
//      
// Jx = | 1+a11   a12|  Jx^(-1) = 1/det*| 1+a22 -a12|  det = (1+a11)*(1+a22) - a12*a21
//      | a21   1+a22|                  | -a21 1+a11|
// 
// Jx^(-1)*Jp = Γ(x) * ∑(p)
// = 1/det * | x y 0 0 1 0| * | 1+a22   0    -a12     0     0      0  |
//           | 0 0 x y 0 1|   |   0   1+a22    0    -a12    0      0  |
//                            | -a21    0    1+a22    0     0      0  |
//                            |   0    -a21    0    1+a22   0      0  |
//                            |   0     0      0      0   1+a22  -a12 |
//                            |   0     0      0      0   -a21   1+a22|

void inverseAdditiveImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect& omega)
{
    const float EPS = 1E-5f; // Threshold value for termination criteria.
    const int MAX_ITER = 100;  // Maximum iteration count.

    const int cols = omega.width;
    const int rows = omega.height;

    std::cout << std::endl << "Start Inverse Additive Algorithm!\n" << std::endl;

    clock_t start_time = clock();

    /*
    *  Pre-computation stage.
    */
    //! 1. [Step-3]Evaluate the gradient of the template ▽T
    //! the function gradient expansions the image with border then get gradient
    //! it is better to calculate the gradient of imgT then get the gradient of T
    cv::Mat T = imgT(omega).clone();
    cv::Mat gradTx,gradTy;
    gradient(T, gradTx, 1, 0);
    gradient(T, gradTy, 0, 1);

    cv::Mat stdesctImage = cv::Mat::zeros(cols*rows, 6, CV_32FC1);
    cv::Mat H_ = cv::Mat::zeros(6, 6, CV_32FC1);

    cv::Mat jac_x;
    cv::Mat dxy;
    cv::Mat J;
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < cols; ++x)
        {
            //! 2: [Step-4]Evaluate Γ(x) = Jx
            jac_x = (cv::Mat_<float>(2, 6) << x, y, 0, 0, 1, 0, 0, 0, x, y, 0, 1);

            //! 3. [Step-5]Calculate modified steepest descent image ▽TΓ(x)
            dxy = (cv::Mat_<float>(1, 2) << gradTx.at<float>(y, x), gradTy.at<float>(y, x));
            J = dxy*jac_x;
            
            J.copyTo(stdesctImage.row(x + y*cols));
        }
    }

    //! 4. [Step-6]Calculate modified Hessian Matrix H* = ∑x[▽I*Γ(x)]^T*[▽I*Γ(x)]
    H_ = stdesctImage.t() * stdesctImage;

    /*
    *   Iteration stage.
    */
   
    //! Evaluate Model's Parameter in Warp: p
    cv::Mat p = cv::Mat::zeros(6, 1, CV_32FC1);
    cv::Mat A = cv::Mat::eye(3, 3, CV_32FC1);

    int iter = 0;
    float mean_error = 0;
    float last_error = 999999;

    while(iter < MAX_ITER)
    {
        iter++;
        mean_error = 0;

        cv::Mat IW;
        cv::Mat Jres = cv::Mat::zeros(6, 1, CV_32FC1);
        cv::Mat dp = cv::Mat::zeros(6, 1, CV_32FC1);

        //! 5. [Step-1]Get the Warp Image of I: I(W(x;p))
        warpAffineback(imgI, IW, A, omega);

        for(int y = 0; y < rows; ++y)
        {
            uint8_t* pIW = IW.ptr<uint8_t>(y);
            uint8_t* pT = T.ptr<uint8_t>(y);
            for(int x = 0; x < cols; ++x)
            {
                //! 6. [Step-2]Compute the error image: Res = I(W(x;p)) - T(x)
                float res = pIW[x] * 1.0 - pT[x];

                mean_error += res*res;

                //! 7. [Step-7]Compute Jres = ∑x[▽I*Γ(x)]^T*[T(x)-I(W(x;p))]
                cv::Mat JT = stdesctImage.row(x + y*cols).t();
                Jres += JT * res;
            }
        }

        mean_error /= rows*cols;

        //! 8. [Step-8]Compute Parameter Increment: Δp = ∑(p)^(-1) * H*^(-1) * Jres, Δp* = H*^(-1) * Jres
        float a11 = A.at<float>(0, 0) - 1;
        float a12 = A.at<float>(1, 0);
        float a21 = A.at<float>(0, 1);
        float a22 = A.at<float>(1, 1) - 1;
        float det = (1 + a11)*(1 + a22) - a12*a21;
        cv::Mat jac_p = (cv::Mat_<float>(6, 6) <<
            1 + a22, 0, -a12, 0, 0, 0,
            0, 1 + a22, 0, -a12, 0, 0,
            -a21, 0, 1 + a22, 0, 0, 0,
            0, -a21, 0, 1 + a22, 0, 0,
            0, 0, 0, 0, 1 + a22, -a12,
            0, 0, 0, 0, -a21, 1 + a22);

        //! the following two lines can modify to: dp = det * (H_ * jac_p).inv() * Jres;
        jac_p /= det;
        dp = jac_p.inv() * H_.inv() * Jres;

        //! 9. [Step-9]Update the parameters p = p - △p
        p -= dp;
        float* pp = p.ptr<float>(0);
        A = (cv::Mat_<float>(3, 3) << 1 + *pp, *(pp + 1), *(pp + 4), *(pp + 2), 1 + *(pp + 3), *(pp + 5), 0, 0, 1);

#ifdef  DEBUG_INF_OUT
        std::cout << "A:" << A << std::endl;
        std::cout << "Iter:" << iter << "  ";
        std::cout << "Mean Error:" << mean_error << std::endl;
#endif // DEBUG_INF_OUT

        if(fabs(dp.at<float>(0, 0)) < EPS && fabs(dp.at<float>(1, 0)) < EPS && fabs(dp.at<float>(2, 0)) < EPS && fabs(dp.at<float>(3, 0)) < EPS && fabs(dp.at<float>(4, 0)) < EPS && fabs(dp.at<float>(5, 0)) < EPS)
        {break;}
    }

    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

    //! Print summary.
    std::cout << "===============================================" << std::endl;
    std::cout << "Algorithm: Inverse Additive" << std::endl;
    std::cout << "A:" << std::endl << A << std::endl;
    std::cout << "Mean Error:" << mean_error << std::endl;
    std::cout << "Iteration:" << iter << std::endl;
    std::cout << "Total Time:" << total_time << std::endl;
    std::cout << "===============================================" << std::endl;
}