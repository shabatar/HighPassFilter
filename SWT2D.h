#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#include <iostream>
using namespace cv;
using namespace std;

#ifndef SWT2D_H
#define SWT2D_H

enum MotherWavelet 
{
  Haar,
  
  Dmey,
  
  Symm
};

enum ConvolutionType {
  CONVOLUTION_FULL,
  CONVOLUTION_SAME,
  CONVOLUTION_VALID
};



void ExtendPeriod(const Mat &B, Mat &C, int level);
void conv2(const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest, int flipcode) ;
void ExtendPeriod(const Mat &B, Mat &C, int level);
void FilterBank(Mat &Kernel_High, Mat &Kernel_Low, MotherWavelet Type);
void KeepLoc(Mat &src, int Extension, int OriginalSize);
void DyadicUpsample(Mat &kernel);
void SWT(const Mat &src_Original, Mat &ca, Mat &ch, Mat &cd, Mat &cv, int Level, MotherWavelet Type);

#endif
