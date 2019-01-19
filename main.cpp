#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <exception>

using namespace cv;

int WINDOW = 60;


Mat abs(Mat m) { // abs from 2 channel image
    if (m.channels() != 2) {
        throw std::runtime_error("Incorrect input mat");
    }

    Mat result(m.rows, m.cols, CV_32F);
    std::vector<Mat> channels(2);
    split(m, channels);
    magnitude(channels[0], channels[1], result);

    return result;
}


Mat fft2(Mat img) {
    Mat padded;
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);

    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI); // Add to the expanded another plane with zeros

    dft(complexI, complexI);
    return complexI;
}

Mat ifft2(Mat& complexI) {
    Mat inverseTransform;
    cv::dft(complexI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    return inverseTransform;
}

void fftshift(Mat& f) {
    int center_cols = (f.cols) / 2;
    int center_rows = (f.rows) / 2;

    Mat topLeft = f(Rect(0, 0, center_cols, center_rows));
    Mat topRight = f(Rect(center_cols, 0, center_cols, center_rows));
    Mat bottomLeft = f(Rect(0, center_rows, center_cols, center_rows));
    Mat bottomRight = f(Rect(center_cols, center_rows, center_cols, center_rows));

    // swap quadrants (Top-Left with Bottom-Right)
    Mat tmp;
    topLeft.copyTo(tmp);
    bottomRight.copyTo(topLeft);
    tmp.copyTo(bottomRight);

    // swap quadrant (Top-Right with Bottom-Left)
    topRight.copyTo(tmp);
    bottomLeft.copyTo(topRight);
    tmp.copyTo(bottomLeft);
}

void ifftshift(Mat& f) {
    fftshift(f);
}

Mat log(const Mat& m) {
    Mat tmp;
    log(m, tmp);
    return tmp;
}

void processChannel(Mat f) {
    f = f(Rect(0, 0, f.cols & -2, f.rows & -2));
    int center_cols = (f.cols) / 2;
    int center_rows = (f.rows) / 2;
    fftshift(f);
    Mat ROI = f(cv::Rect(center_cols-WINDOW/2, center_rows-WINDOW/2, WINDOW, WINDOW));
    ROI.setTo(Scalar(0,0,0));
    ifftshift(f);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Specify the image path as command line argument";
        return -1;
    }
    Mat inputImage = imread(argv[1]);
    cvtColor(inputImage, inputImage, CV_BGR2GRAY);

    if (inputImage.empty())
        return -1;

    Mat complexI = fft2(inputImage);

    Mat channels[2];
    split(complexI, channels);
    processChannel(channels[0]);
    processChannel(channels[1]);

    Mat res;
    merge(channels, 2, res);
    res = ifft2(res);
    //abs(res1);
    normalize(res, res, 0, 1, CV_MINMAX);

    // --- if you want threshold
    res = res * 255;
    Mat dst;
    threshold( res, dst, 125, 250, CV_THRESH_BINARY );
    // ---

    imwrite(argv[1], dst);
}
