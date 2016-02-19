#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//原始的LBP算法
//使用模板参数

template <typename _Tp> static
void olbp(InputArray _src, OutputArray _dst) {
    // get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2, src.cols-2, CV_8UC1);
    Mat dst = _dst.getMat();
    // zero the result matrix
    dst.setTo(0);
    
    cout<<"rows "<<src.rows<<" cols "<<src.cols<<endl;
    cout<<"channels "<<src.channels();
    getchar();
    // calculate patterns
    for(int i=1;i<src.rows-1;i++) {
        cout<<endl;
        for(int j=1;j<src.cols-1;j++) {
            
            _Tp center = src.at<_Tp>(i,j);
            //cout<<"center"<<(int)center<<"  ";
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
            code |= (src.at<_Tp>(i-1,j  ) >= center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
            code |= (src.at<_Tp>(i  ,j+1) >= center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
            code |= (src.at<_Tp>(i+1,j  ) >= center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
            code |= (src.at<_Tp>(i  ,j-1) >= center) << 0;
            
            dst.at<unsigned char>(i-1,j-1) = code;
            //cout<<(int)code<<" ";
            //cout<<(int)code<<endl;
        }
    }
}



void elbp(Mat& src, Mat &dst, int radius, int neighbors)
{
    
    neighbors = max(min(neighbors,31),1); // set bounds...
    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8UC1);
    
    for(int n=0; n<neighbors; n++)
    {
        // 采样点的计算
        float x = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // 上取整和下取整的值
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // 小数部分
        float ty = y - fy;
        float tx = x - fx;
        // 设置插值权重
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // 循环处理图像数据
        for(int i=radius; i < src.rows-radius;i++)
        {
            for(int j=radius;j < src.cols-radius;j++)
            {
                // 计算插值
                float t = static_cast<float>(w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx));
                // 进行编码
                dst.at<uchar>(i-radius,j-radius) += ((t > src.at<uchar>(i,j)) || (std::abs(t-src.at<uchar>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}





//基于旧版本的opencv的LBP算法opencv1.0
void LBP (IplImage *src,IplImage *dst)
{
    int tmp[8]={0};
    CvScalar s;
    
    IplImage * temp = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U,1);
    uchar *data=(uchar*)src->imageData;
    int step=src->widthStep;
    
    cout<<"step"<<step<<endl;
    
    for (int i=1;i<src->height-1;i++)
        for(int j=1;j<src->width-1;j++)
        {
            int sum=0;
            if(data[(i-1)*step+j-1]>data[i*step+j])
                tmp[0]=1;
            else
                tmp[0]=0;
            if(data[i*step+(j-1)]>data[i*step+j])
                tmp[1]=1;
            else
                tmp[1]=0;
            if(data[(i+1)*step+(j-1)]>data[i*step+j])
                tmp[2]=1;
            else
                tmp[2]=0;
            if (data[(i+1)*step+j]>data[i*step+j])
                tmp[3]=1;
            else
                tmp[3]=0;
            if (data[(i+1)*step+(j+1)]>data[i*step+j])
                tmp[4]=1;
            else
                tmp[4]=0;
            if(data[i*step+(j+1)]>data[i*step+j])
                tmp[5]=1;
            else
                tmp[5]=0;
            if(data[(i-1)*step+(j+1)]>data[i*step+j])
                tmp[6]=1;
            else
                tmp[6]=0;
            if(data[(i-1)*step+j]>data[i*step+j])
                tmp[7]=1;
            else
                tmp[7]=0;
            //计算LBP编码
            s.val[0]=(tmp[0]*1+tmp[1]*2+tmp[2]*4+tmp[3]*8+tmp[4]*16+tmp[5]*32+tmp[6]*64+tmp[7]*128);
            
            cvSet2D(dst,i,j,s);//写入LBP图像
        }
}


Mat getHistImg(const MatND& hist)
{
    double maxVal=0;
    double minVal=0;
    
    //找到直方图中的最大值和最小值
    minMaxLoc(hist,&minVal,&maxVal,0,0);
    int histSize=hist.rows;
    Mat histImg(histSize,histSize,CV_8U,Scalar(255));
    // 设置最大峰值为图像高度的90%
    int hpt=static_cast<int>(0.9*histSize);
    
    for(int h=0;h<histSize;h++)
    {
        float binVal=hist.at<float>(h);
        int intensity=static_cast<int>(binVal*hpt/maxVal);
        line(histImg,Point(h,histSize),Point(h,histSize-intensity),Scalar::all(0));
    }
    
    return histImg;
}


int main()
{
    IplImage* face = cvLoadImage("/Users/hyy/Dropbox/CPPworkspace/LBP2/image.jpg",CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    
    IplImage* Gray_face = cvCreateImage( cvSize( face->width,face->height ), face->depth, 1);//先分配图像空间
    cvCvtColor(face, Gray_face ,CV_BGR2GRAY);//把载入图像转换为灰度图
    IplImage* lbp_face =   cvCreateImage(cvGetSize(Gray_face), IPL_DEPTH_8U,1);//先分配图像空间
    
    cvNamedWindow("Gray Image",1);
    cvShowImage("Gray Image",Gray_face);
 
    
    Mat imgmat = imread("/Users/hyy/Dropbox/CPPworkspace/LBP2/image.jpg");
    
    Mat Gray;
    
    cvtColor(imgmat, Gray, CV_BGR2GRAY);
    
    GaussianBlur(Gray, Gray, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    
    Mat lbpMAT;
   
    elbp(Gray,lbpMAT,1,8);

    //normalize(lbpMAT, lbpMAT, 0, 255, NORM_MINMAX, CV_8UC1);
    namedWindow("lbp",CV_WINDOW_AUTOSIZE);
    imshow("lbp", lbpMAT);

    
    //显示原始的输入图像
    cvNamedWindow("Src Image",CV_WINDOW_AUTOSIZE);
    cvShowImage("Src Image",face);


    //计算输入图像的LBP纹理特征
    LBP(Gray_face,lbp_face);

    //显示第一幅图像的LBP纹理特征图
    cvNamedWindow("LBP Image",CV_WINDOW_AUTOSIZE);
    cvShowImage("LBP Image",lbp_face);

    // Use the o-th and 1-st channels
    int channels[] = { 0 };
    
    int histSize[] = {256};
    
    float s_ranges[] = { 0, 256 };
    
    const float* ranges[] = { s_ranges };
    
    Mat lbp_hist, lbp1_hist;
    
    calcHist( &lbpMAT, 1, channels, Mat(), lbp_hist, 1, histSize, ranges, true, false );
    normalize( lbp1_hist, lbp1_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    
    
    imshow("lbp hist", getHistImg(lbp_hist));
    
    
    waitKey();
    

    cvDestroyWindow("Src Image");

    waitKey();
    return 0;
    
}