#include <iostream>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "visionkit.hpp"
#include "invcomp.hpp"

//for warp affine, we evaluate jacobian
// x' = (1+a11)*x+a12*y+tx;
// y' = a21*x+(1+a22)*y+ty;
// p = (a11, a12, a21, a22, tx, ty)
// 6 Dof
// J = | x y 0 0 1 0|
//     | 0 0 x y 0 1|
void inverseCompositionalImageAlign(cv::Mat& imgT, cv::Mat& imgI, cv::Rect omega)
{
    const float EPS = 1E-5f; // Threshold value for termination criteria.
    const int MAX_ITER = 100;  // Maximum iteration count.

    const int cols = omega.width;
    const int rows = omega.height;

    /*
     *  Precomputation stage.
     */

    //! Step1: Evaluate the gradient of the templet
    cv::Mat T = imgT(omega).clone();
    cv::Mat gradTx,gradTy;
    gradient(T,gradTx,1,0);
    gradient(T,gradTy,0,1);

    cv::Mat jac;
    cv::Mat dxy;
    cv::Mat Jac_cache = cv::Mat::zeros(cols*rows, 6, CV_32F);
    cv::Mat H = cv::Mat::zeros(6, 6, CV_32FC1);

    std::cout << std::endl << "Start Inverse Compositional Algorithm!" << std::endl;
    clock_t start_time = clock();
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < cols; ++x)
        {
			//! Step2: Evaluate the Jacobin at(x; p)
            jac =  (cv::Mat_<float>(2,6) << x, y, 0, 0, 1, 0, 0, 0, x, y, 0, 1);

			//! Step3: Calculate steepest descent image
            dxy =  (cv::Mat_<float>(1,2) << gradTx.at<float>(y,x),gradTy.at<float>(y,x));
            cv::Mat J = dxy*jac;
            
            J.copyTo(Jac_cache.row(x+y*cols));
        }
    }
    clock_t percompute_time = clock();
    double per_time = (double)(percompute_time-start_time)/CLOCKS_PER_SEC;

    //! Step4: Calculate Hessian Matrix
    H = Jac_cache.t()*Jac_cache;

    //! Get Invert Hessian Matrix
    cv::Mat Hinv = H.inv();

    /*
     *   Iteration stage.
     */

    //! Evaluate Model's Parameter in Warp: p
    cv::Mat A = (cv::Mat_<float>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

    int iter = 0;
    float mean_error = 0;
    float last_error = 999999;

    cv::Mat dA;
    while(iter < MAX_ITER)
    {
        iter++;

        mean_error = 0;

        cv::Mat IW;
        cv::Mat Jres = cv::Mat::zeros(6,1, CV_32FC1);
        cv::Mat dp = cv::Mat::zeros(6,1, CV_32FC1);

        //! Step5: Get the Warp Image of I: I(W(x;p))
        warpAffine(imgI, IW, A, omega);

        for(int y = 0; y < rows; ++y)
        {
            uint8_t* pIW = IW.ptr<uint8_t>(y);
            uint8_t* pT = T.ptr<uint8_t>(y);
            for(int x = 0; x < cols; ++x)
            {
                //! Step6: Compute the error image: Res = I(W(x;p)) - T(x)
                float res = pIW[x]*1.0 - pT[x];

                mean_error += res*res;

				//! Step7: Compute Jres
                cv::Mat JT = Jac_cache.row(x+y*cols).t();
                Jres += JT * res;
            }
        }

        mean_error /= rows*cols;

        /*if(mean_error > last_error)
        {
            A = A * dA;
        }*/
        last_error = mean_error;

        //! Step8: Compute Parameter Increment: dp = H^(-1) * Jres 
        dp = Hinv * Jres;

        //! Step9: Invert increment of Warp and update Warp: W(x,p) = W(x;p) * W(x;dp)^-1 
        float dA11 = dp.at<float>(0,0);
        float dA12 = dp.at<float>(1,0);
        float dA21 = dp.at<float>(2,0);
        float dA22 = dp.at<float>(3,0);
        float dtx = dp.at<float>(4,0);
        float dty = dp.at<float>(5,0);
        intAffine(dA,dA11+1,dA12,dA21,dA22+1,dtx,dty);
        A = A * dA.inv();

#ifdef  DEBUG_INF_OUT
		std::cout << "A:" << A << std::endl;
		std::cout << "Iter:" << iter << "  ";
		std::cout << "Mean Error:" << mean_error << std::endl;
#endif // DEBUG_INF_OUT

        if(fabs(dA11) < EPS && fabs(dA12) < EPS && fabs(dA21) < EPS && fabs(dA22) < EPS && fabs(dtx) < EPS && fabs(dty) < EPS)
        {break;}
    }
    clock_t finish_time = clock();
    double total_time = (double)(finish_time-start_time)/CLOCKS_PER_SEC;

    //! Print summary.
    std::cout << "===============================================" << std::endl;
    std::cout << "Algorithm: inverse compositional" << std::endl;
    std::cout << "A:" << std::endl << A << std::endl;
    std::cout << "Mean Error:" << mean_error << std::endl;
    std::cout << "Iteration:" << iter << std::endl;
    std::cout << "Percompute Time:" << per_time << std::endl;
    std::cout << "Total Time:" << total_time << std::endl;
    std::cout << "===============================================" << std::endl;
}