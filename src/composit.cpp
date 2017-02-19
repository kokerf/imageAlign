#include <iostream>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "visionkit.hpp"
#include "composit.hpp"

void forwardCompositionalImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect omega)
{
    const float EPS = 1E-5f; // Threshold value for termination criteria.
    const int MAX_ITER = 100;  // Maximum iteration count.

    const int cols = omega.width;
    const int rows = omega.height;

    std::cout << std::endl << "Start Forward Compositional Algorithm!" << std::endl;

    clock_t start_time = clock();

    cv::Mat T = imgT(omega).clone();
    
    /*
    *   Iteration stage.
    */

    cv::Mat A = cv::Mat::eye(3, 3, CV_32FC1);

    float mean_error = 0;
    int iter = 0;

    while(iter < MAX_ITER)
    {
        iter++;
        mean_error = 0;

        cv::Mat IW;
        cv::Mat gradIx_W, gradIy_W;
        cv::Mat H = cv::Mat::zeros(6, 6, CV_32FC1);
        cv::Mat Jres = cv::Mat::zeros(6, 1, CV_32FC1);
        cv::Mat dp = cv::Mat::zeros(6, 1, CV_32FC1);
        cv::Mat dA;

        //! 1. [Step-1]Get the Warp Image of I: I(W(x;p))
        warpAffineback(imgI, IW, A, omega);

        //! 2. [Step-3]Get gradient ▽I(W)
        //! it is better to get IW with border, then calculate the gradient of IW
        cv::Mat gradIWx, gradIWy;
        gradient(IW, gradIWx, 1, 0);
        gradient(IW, gradIWy, 0, 1);

        cv::Mat jac;
        cv::Mat dxy;
        cv::Mat J;
        for(int y = 0; y < rows; ++y)
        {
            uint8_t* pIW = IW.ptr<uint8_t>(y);
            uint8_t* pT = T.ptr<uint8_t>(y);
            for(int x = 0; x < cols; ++x)
            {
                //! 3. [Step-4]Evaluate the Jacobin ∂W/∂p at (x;p)
                jac = (cv::Mat_<float>(2, 6) << x, y, 0, 0, 1, 0, 0, 0, x, y, 0, 1);

                //! 4. [Step-5]Calculate steepest descent image ▽I*∂W/∂p
                dxy = (cv::Mat_<float>(1, 2) << gradIWx.at<float>(y, x), gradIWy.at<float>(y, x));
                J = dxy*jac;

                //! 5. [Step-6]Calculate Hessian Matrix H = ∑x[▽I*∂W/∂p]^T*[▽I*∂W/∂p]
                H += J.t() * J;

                //! 6. [Step-2]Compute the error image T(x) - I(W(x:p))
                float res = pT[x] * 1.0 - pIW[x];
                mean_error += res*res;

                //! 7. [Step-7]Compute Jres = ∑x[▽I*∂W/∂p]^T*[T(x)-I(W(x;p))]
                Jres += J.t() * res;
            }
        }

        mean_error /= rows*cols;

        //! 8. [Step-8]Compute △p = H^(-1) * Jres
        dp = H.inv() * Jres;

        //! Step9: Update the parameters p = p * △p
        float dA11 = dp.at<float>(0,0);
        float dA12 = dp.at<float>(1,0);
        float dA21 = dp.at<float>(2,0);
        float dA22 = dp.at<float>(3,0);
        float dtx = dp.at<float>(4,0);
        float dty = dp.at<float>(5,0);
        dA = (cv::Mat_<float>(3, 3) << dA11 + 1, dA12, dtx, dA21, dA22 + 1, dty, 0, 0, 1);
        A *= dA;

#ifdef  DEBUG_INF_OUT
        std::cout << "A:" << A << std::endl;
        std::cout << "Iter:" << iter << "  ";
        std::cout << "Mean Error:" << mean_error << std::endl;
#endif // DEBUG_INF_OUT

        if(fabs(dA11) < EPS && fabs(dA12) < EPS && fabs(dA21) < EPS && fabs(dA22) < EPS && fabs(dtx) < EPS && fabs(dty) < EPS)
        {break;}
    }
    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

    //! Print summary.
    std::cout << "===============================================" << std::endl;
    std::cout << "Algorithm: Forward Compositional" << std::endl;
    std::cout << "A:" << std::endl << A << std::endl;
    std::cout << "Mean Error:" << mean_error << std::endl;
    std::cout << "Iteration:" << iter << std::endl;
    std::cout << "Total Time:" << total_time << std::endl;
    std::cout << "===============================================" << std::endl;

}