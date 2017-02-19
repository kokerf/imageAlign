#include <iostream>
#include <stdint.h>

#include "visionkit.hpp"

// the Affine
// | a11 a12 tx |
// | a21 a22 ty |
// |  0   0  1  |
void intAffine(cv::Mat& A, float a11, float a12, float a21, float a22, float tx, float ty)
{
    A = (cv::Mat_<float>(3, 3) << a11, a12, tx, a21, a22, ty, 0, 0, 1);
}

// the origin of axis of the Affine is on the centre of the picture
void warpAffine(
    const cv::Mat& img_ref,
    cv::Mat& img_out,
    const cv::Mat& A,
    cv::Rect& omega,
    cv::Point2d O,
    bool flag)
{
    assert(img_ref.type()==CV_8UC1);
    assert(A.rows== 3 && A.cols==3);

    if(!img_out.empty())
    {
        img_out.release();
    }
    img_out = cv::Mat::zeros(omega.height, omega.width, CV_8UC1);
    
    const int refCols = img_ref.cols;
    const int refRows = img_ref.rows;
    const int destCols = img_out.cols;
    const int destRows = img_out.rows;
    const float cx = (!flag) ? (omega.width*0.5) : (O.x);
    const float cy = (!flag) ? (omega.height*0.5) : (O.y);

    float a11 = A.at<float>(0, 0);
    float a12 = A.at<float>(0, 1);
    float a21 = A.at<float>(1, 0);
    float a22 = A.at<float>(1, 1);
    float tx = A.at<float>(0, 2);
    float ty = A.at<float>(1, 2);

    uint8_t* im_ptr = img_out.data;
    //! (u,v) is warp from (x,y) in img_out
    float u, v;
    for(int y = 0; y < destRows; ++y)
    {
        for(int x = 0; x < destCols; ++x, ++im_ptr)
        {
            u = a11 * (x - cx) + a12 * (y - cy) + tx;
            v = a21 * (x - cx) + a22 * (y - cy) + ty;
            u += cx;
            v += cy;

            float u1, v1;
            u1 = u + omega.x;
            v1 = v + omega.y;

            if(u1 < 0 || v1 < 0 || u1 > refCols -1 || v1 > refRows-1)
                continue;
            else
            {
                *im_ptr = (uint8_t)interpolateMat_8u(img_ref, u1, v1);
            }
        }
    }

}

void warpAffine(
    const cv::Mat& img_ref,
    cv::Mat& img_out,
    const cv::Mat& A,
    cv::Point2d O)
{
    assert(img_ref.type() == CV_8UC1);
    assert(A.rows == 3 && A.cols == 3);

    if (!img_out.empty())
    {
        img_out.release();
    }
    img_out = cv::Mat::zeros(img_ref.size(), CV_8UC1);

    const int refCols = img_ref.cols;
    const int refRows = img_ref.rows;
    const float cx = O.x;
    const float cy = O.y;
    const int destCols = img_out.cols;
    const int destRows = img_out.rows;

    float a11 = A.at<float>(0, 0);
    float a12 = A.at<float>(0, 1);
    float a21 = A.at<float>(1, 0);
    float a22 = A.at<float>(1, 1);
    float tx = A.at<float>(0, 2);
    float ty = A.at<float>(1, 2);

    uint8_t* im_ptr = img_out.data;
    //! (u,v) is in img_out
    float u, v;
    for (int y = 0; y < destRows; ++y)
    {
        for (int x = 0; x < destCols; ++x, ++im_ptr)
        {
            u = a11 * (x - cx) + a12 * (y - cy) + tx;
            v = a21 * (x - cx) + a22 * (y - cy) + ty;
            u += cx;
            v += cy;

            float u1, v1;
            u1 = u;
            v1 = v;

            if (u1 < 0 || v1 < 0 || u1 > refCols - 1 || v1 > refRows - 1)
                continue;
            else
            {
                *im_ptr = (uint8_t)interpolateMat_8u(img_ref, u1, v1);
            }
        }
    }

}

void warpAffine_float(
    const cv::Mat& img_ref,
    cv::Mat& img_out,
    const cv::Mat& A,
    cv::Rect& omega,
    cv::Point2d O,
    bool flag)
{
    assert(img_ref.type() == CV_32FC1);
    assert(A.rows == 3 && A.cols == 3);

    if (!img_out.empty())
    {
        img_out.release();
    }
    img_out = cv::Mat::zeros(omega.height, omega.width, CV_32FC1);

    const int refCols = img_ref.cols;
    const int refRows = img_ref.rows;
    const int destCols = img_out.cols;
    const int destRows = img_out.rows;
    const float cx = (!flag) ? (omega.width*0.5) : (O.x);
    const float cy = (!flag) ? (omega.height*0.5) : (O.y);

    float a11 = A.at<float>(0, 0);
    float a12 = A.at<float>(0, 1);
    float a21 = A.at<float>(1, 0);
    float a22 = A.at<float>(1, 1);
    float tx = A.at<float>(0, 2);
    float ty = A.at<float>(1, 2);

    //float* im_ptr = (float*)img_out.data;
    float* im_ptr = img_out.ptr<float>(0);
    //! (u,v) is warp from (x,y) in img_out
    float u, v;
    for (int y = 0; y < destRows; ++y)
    {
        for (int x = 0; x < destCols; ++x, ++im_ptr)
        {
            u = a11 * (x - cx) + a12 * (y - cy) + tx;
            v = a21 * (x - cx) + a22 * (y - cy) + ty;
            u += cx;
            v += cy;

            float u1, v1;
            u1 = u + omega.x;
            v1 = v + omega.y;

            if (u1 < 0 || v1 < 0 || u1 > refCols - 1 || v1 > refRows - 1)
                continue;
            else
            {
                *im_ptr = interpolateMat_32f(img_ref, u1, v1);
            }
        }
    }
}

void gradient(const cv::Mat& I, cv::Mat& G, int dx, int dy)
{
    assert(I.type()==CV_8UC1);

    //! use Sobel gradient
    int8_t Sobel_dx[9] = {-1, 0, 1,-2, 0, 2,-1, 0, 1};
    int8_t Sobel_dy[9] = {-1,-2,-1, 0, 0, 0, 1, 2, 1};
    int8_t* sobel;
    if(dx == 1 && dy == 0)
    {
        sobel = Sobel_dx;
    }
    else if(dx == 0 && dy == 1)
    {
        sobel = Sobel_dy;
    }
    else
    {
        std::cout << "!!!Error in gradient(): dx = " << dx << " dy = " << dy << std::endl;
        return;
    }

    //! Get T with border
    cv::Mat T = cv::Mat(I.rows + 2, I.cols + 2, CV_8UC1);
    I.copyTo(T.rowRange(1, T.rows - 1).colRange(1, T.cols - 1));
    I.row(0).copyTo(T.row(0).colRange(1, T.cols - 1));
    I.col(0).copyTo(T.col(0).rowRange(1, T.rows - 1));
    I.row(I.rows - 1).copyTo(T.row(T.rows - 1).colRange(1, T.cols - 1));
    I.col(I.cols - 1).copyTo(T.col(T.cols - 1).rowRange(1, T.rows - 1));
    I.row(0).col(0).copyTo(T.row(0).col(0));
    I.row(0).col(I.cols - 1).copyTo(T.row(0).col(T.cols - 1));
    I.row(I.rows - 1).col(0).copyTo(T.row(T.rows - 1).col(0));
    I.row(I.rows - 1).col(I.cols - 1).copyTo(T.row(T.rows - 1).col(T.cols - 1));

    //! Creat Gradient Image
    if (G.empty())
        G = cv::Mat(I.size(), CV_32FC1);
    else
        assert(G.type() == CV_32FC1 && G.size() == I.size());

    int strideT = T.step[0];

    int rows = I.rows;
    int cols = I.cols;
    int u, v;
    for(int y = 0; y < rows; ++y)
    {
        v = y + 1;
        float* G_ptr = G.ptr<float>(y);
        uint8_t* T_ptr = (uint8_t*)T.ptr<uint8_t>(v);
        for(int x = 0; x < cols; ++x)
        {
            u = x + 1;
            G_ptr[x] = T_ptr[u - 1 - strideT] * sobel[0]
                + T_ptr[u - strideT] * sobel[1]
                + T_ptr[u + 1 - strideT] * sobel[2]
                + T_ptr[u - 1] * sobel[3]
                + T_ptr[u] * sobel[4]
                + T_ptr[u + 1] * sobel[5]
                + T_ptr[u - 1 + strideT] * sobel[6]
                + T_ptr[u + strideT] * sobel[7]
                + T_ptr[u + 1 + strideT] * sobel[8];

            G_ptr[x] /= 8.0;
        }
    }
}

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! Return value between 0 and 255
//! WARNING This function does not check whether the x/y is within the border
float interpolateMat_8u(const cv::Mat& mat, float u, float v)
{
    assert(mat.type() == CV_8UC1);
    int x = floor(u);
    int y = floor(v);
    float subpix_x = u - x;
    float subpix_y = v - y;

    float w00 = (1.0f - subpix_x)*(1.0f - subpix_y);
    float w01 = (1.0f - subpix_x)*subpix_y;
    float w10 = subpix_x*(1.0f - subpix_y);
    float w11 = 1.0f - w00 - w01 - w10;

    //! addr(Mij) = M.data + M.step[0]*i + M.step[1]*j
    const int stride = mat.step.p[0];
    unsigned char* ptr = mat.data + y*stride + x;
    return w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride + 1] + 0.5;//! add 0.5 to round off!
}

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
float interpolateMat_32f(const cv::Mat& mat, float u, float v)
{
    assert(mat.type() == CV_32F);
    float x = floor(u);
    float y = floor(v);
    float subpix_x = u - x;
    float subpix_y = v - y;
    float wx0 = 1.0 - subpix_x;
    float wx1 = subpix_x;
    float wy0 = 1.0 - subpix_y;
    float wy1 = subpix_y;

    float val00 = mat.at<float>(y, x);
    float val10 = mat.at<float>(y, x + 1);
    float val01 = mat.at<float>(y + 1, x);
    float val11 = mat.at<float>(y + 1, x + 1);
    return (wx0*wy0)*val00 + (wx1*wy0)*val10 + (wx0*wy1)*val01 + (wx1*wy1)*val11;
}