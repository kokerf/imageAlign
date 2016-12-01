#include <iostream>
#include <stdint.h>

#include "visionkit.hpp"

// the Affine
// | a11 a12 tx |
// | a21 a22 ty |
// |  0   0  1  |
void intAffine(cv::Mat& A, float a11, float a12, float a21, float a22, float tx, float ty)
{
    A = (cv::Mat_<float>(3,3) << a11, a12, tx, a21, a22, ty, 0, 0, 1);
}

// the origin of axis is on the centre of the picture
void warpAffine(
    const cv::Mat& A,
    const cv::Mat& img_ref,
    cv::Mat& img_out)
{
    assert(img_ref.type()==CV_8UC1);
    assert(A.rows== 3 && A.cols==3);

    if(img_out.empty())
        img_out.create(img_ref.size(),CV_8U);
    else
		assert(img_out.type()==CV_8UC1 && img_out.size()==img_ref.size());
    
    int refCols = img_ref.cols;
    int refRows = img_ref.rows;
    float cx = img_ref.cols/2.0;
    float cy = img_ref.rows/2.0;
    int destCols = img_out.cols;
    int destRows = img_out.rows;

    float a11 = A.at<float>(0,0);
    float a12 = A.at<float>(0,1);
    float a21 = A.at<float>(1,0);
    float a22 = A.at<float>(1,1);
    float tx = A.at<float>(0,2);
    float ty = A.at<float>(1,2);
    //std::cout<<"A:"<<a11<<","<<a12<<","<<a21<<","<<a22<<","<<tx<<","<<ty<<std::endl;

    uint8_t* im_ptr = img_out.data;
    for(int y = 0; y < refRows; ++y)
    {
        for(int x = 0; x < refCols; ++x, ++im_ptr)
        {
            float x1 = a11 * (x-cx) + a12 * (y-cy) + tx;
            float y1 = a21 * (x-cx) + a22 * (y-cy) + ty;
            x1 += cx;
            y1 += cy;

            if(x1 < 0 || y1 < 0 || x1 > destCols-1 || y1 > destRows-1)
                continue;
            else
            {
                *im_ptr = (uint8_t)interpolateMat_8u(img_ref,x1,y1);
            }
        }
    }

}

void gradient(const cv::Mat& I, cv::Mat& G, int dx, int dy)
{
    assert(I.type()==CV_8UC1);

    // use Sobel gradient
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
        std::cout<<"!!!Error in gradient(): dx = "<<dx<<" dy = "<<dy<<std::endl;
        return;
    }

    // Get T with border
    cv::Mat T = cv::Mat(I.rows+2,I.cols+2,CV_8UC1);
    I.copyTo(T.rowRange(1,T.rows-1).colRange(1,T.cols-1));
    I.row(0).copyTo(T.row(0).colRange(1,T.cols-1));
    I.col(0).copyTo(T.col(0).rowRange(1,T.rows-1));
    I.row(I.rows-1).copyTo(T.row(T.rows-1).colRange(1,T.cols-1));
    I.col(I.cols-1).copyTo(T.col(T.cols-1).rowRange(1,T.rows-1));
    I.row(0).col(0).copyTo(T.row(0).col(0));
    I.row(0).col(I.cols-1).copyTo(T.row(0).col(T.cols-1));
    I.row(I.rows-1).col(0).copyTo(T.row(T.rows-1).col(0));
    I.row(I.rows-1).col(I.cols-1).copyTo(T.row(T.rows-1).col(T.cols-1));

    // Creat Gradient Image
    if(G.empty())
        G = cv::Mat(I.size(),CV_32FC1);
    else
		assert(G.type()==CV_32FC1 && G.size()==I.size());

    int strideI = I.step[0];

    int rows = I.rows;
    int cols = I.cols;
    for(int y = 0; y < rows; ++y)
    {
        float* G_ptr = G.ptr<float>(y);
        uint8_t* I_ptr = (uint8_t*) I.ptr<uint8_t>(y);
        for(int x = 0; x < cols; ++x)
        {
            G_ptr[x] = I_ptr[x-1-strideI] * sobel[0]
                     + I_ptr[x-strideI] * sobel[1]
                     + I_ptr[x+1-strideI] * sobel[2]
                     + I_ptr[x-1] * sobel[3]
                     + I_ptr[x] * sobel[4]
                     + I_ptr[x+1] * sobel[5]
                     + I_ptr[x-1+strideI] * sobel[6]
                     + I_ptr[x+strideI] * sobel[7]
                     + I_ptr[x+1+strideI] * sobel[8];

            G_ptr[x] /= 8.0;
        }
    }
}


float interpolateMat_8u(const cv::Mat& mat, float u, float v)
{
  assert(mat.type()==CV_8U);
  int x = floor(u);
  int y = floor(v);
  float subpix_x = u-x;
  float subpix_y = v-y;

  float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
  float w01 = (1.0f-subpix_x)*subpix_y;
  float w10 = subpix_x*(1.0f-subpix_y);
  float w11 = 1.0f - w00 - w01 - w10;

  // addr(Mij) = M.data + M.step[0]*i + M.step[1]*j
  const int stride = mat.step.p[0];
  unsigned char* ptr = mat.data + y*stride + x;
  return w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride+1];
}