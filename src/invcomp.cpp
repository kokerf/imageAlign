#include <iostream>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "invcomp.hpp"
#include "visionkit.hpp"

//for warp affine, we evaluate jacobian
// x' = (1+a11)*x+a12*y+tx;
// y' = a21*x+(1+a22)*y+ty;
// p = (a11, a12, a21, a22, tx, ty)
// 6 Dof
// J = | x y 0 0 1 0|
//     | 0 0 x y 0 1|
void inverseCompositionalAlign(cv::Mat& imgT, cv::Mat& imgI)
{
    const float EPS = 1E-5f; // Threshold value for termination criteria.
    const int MAX_ITER = 200;  // Maximum iteration count.

    const int cols = imgT.cols;
    const int rows = imgT.rows;
    const int cx = imgT.cols/2;
    const int cy = imgT.rows/2;

    /*
     *  Precomputation stage.
     */

    //! Evaluate gradient of T
    cv::Mat gradDx,gradDy;
    gradient(imgT,gradDx,1,0);
    gradient(imgT,gradDy,0,1);

    cv::Mat jac;
    cv::Mat dxy;
    cv::Mat Jac_cache = cv::Mat::zeros(cols*rows, 6, CV_32F);
    cv::Mat H;

    std::cout<<"Start Inverse Compositional Algorithm!"<<std::endl;
    clock_t start_time = clock();
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < cols; ++x)
        {

            //! Calculate steepest descent image.
            jac =  (cv::Mat_<float>(2,6) << x, y, 0, 0, 1, 0, 0, 0, x, y, 0, 1);
            dxy =  (cv::Mat_<float>(1,2) << gradDx.at<float>(y,x),gradDy.at<float>(y,x));
            cv::Mat J = dxy*jac;
            //Jac_cache.row(x+y*rows) = dxy*jac;
            
            J.copyTo(Jac_cache.row(x+y*rows));
            //compute H in here will contribute to larger error
            //H = H + J.t() * J;

        }
    }
    clock_t percompute_time = clock();
    double per_time = (double)(percompute_time-start_time)/CLOCKS_PER_SEC;

    //! Calculate Hessian Matrix
    H = Jac_cache.t()*Jac_cache;
    std::cout<<"H:"<<H<<std::endl;

    //! Get Invert Hessian Matrix
    cv::Mat Hinv = H.inv();
    std::cout<<Hinv<<std::endl;

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
        cv::Mat dP = cv::Mat::zeros(6,1, CV_32FC1);

        //! Get the Warp Image of I: I(W(x;p))
        warpAffine(A,imgI,IW);

        for(int y = 0; y < rows; ++y)
        {
            uint8_t* pIW = IW.ptr<uint8_t>(y);
            uint8_t* pT = imgT.ptr<uint8_t>(y);
            for(int x = 0; x < cols; ++x)
            {
                //! Compute the error image: Res = I(W(x;p)) - T(x)
                float res = pIW[x]*1.0 - pT[x];
                //std::cout<<"res:"<<res<<std::endl;
                mean_error += res*res;

                cv::Mat JT = Jac_cache.row(x+y*rows).t();

                Jres += JT * res;
                //std::cout<<"Jac_cache:"<<std::endl<<Jac_cache.row(820)<<std::endl;
                //std::cout<<"Jres "<<x+y*rows<<":"<<std::endl<<Jres<<std::endl;
            }
        }

        mean_error /= rows*cols;

        if(mean_error > last_error)
        {
            A = A * dA;
        }
        last_error = mean_error;

        //! Compute Parameter Increment: dP = H^(-1) * Jres 
        dP = Hinv * Jres;
        std::cout<<"dP"<<std::endl<<dP<<std::endl;

        //! Invert increment of Warp and update Warp: W(x,p) = W(x;p) * W(x;dp)^-1 
        float dA11 = dP.at<float>(0,0);
        float dA12 = dP.at<float>(1,0);
        float dA21 = dP.at<float>(2,0);
        float dA22 = dP.at<float>(3,0);
        float dtx = dP.at<float>(4,0);
        float dty = dP.at<float>(5,0);
        intAffine(dA,dA11+1,dA12,dA21,dA22+1,dtx,dty);
        A = A * dA.inv();

        std::cout<<"Ai:"<<A<<std::endl;
        std::cout<<"Time:"<<iter<<"  ";
        std::cout<<"Mean Error:"<<mean_error<<std::endl;

        if(fabs(dA11) < EPS && fabs(dA12) < EPS && fabs(dA21) < EPS && fabs(dA22) < EPS && fabs(dtx) < EPS && fabs(dty) < EPS)
        {break;}
    }
    clock_t finish_time = clock();
    double total_time = (double)(finish_time-start_time)/CLOCKS_PER_SEC;

    //! Print summary.
    std::cout<<"==============================================="<<std::endl;
    std::cout<<"Algorithm: inverse compositional"<<std::endl;
    std::cout<<"A:"<<std::endl<<A.inv()<<std::endl;
    std::cout<<"Percompute Time:"<<per_time<<std::endl;
    std::cout<<"Total Time:"<<total_time<<std::endl;
    std::cout<<"==============================================="<<std::endl;
}