#include <iostream>
#include <cmath>

using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

// From ID Software
inline float fast_inv_sqrt(float n) {
    const float threehalfs = 1.5F;
    float y = n;

    long i = * ( long * ) &y;

    i = 0x5f3759df - ( i >> 1 );
    y = * ( float * ) &i;

    y = y * ( threehalfs - ( (n * 0.5F) * y * y ) );

    return y;
}

Mat RGB_2_XYZ(const Mat &);
Mat XYZ_2_LMS(const Mat &);
Mat RGB_2_LMS(const Mat &);
Mat RGB_2_logLMS(const Mat &);
Mat LMS_2_LAlBe(const Mat &);
Mat LAlBe_2_logLMS(const Mat &);
Mat logLMS_2_RGB(const Mat &);

int main() {
    const string IMG_PATH = "./res/",
                 IMG_NAME = "test1",
                 IMG_EXT = ".jpg",
                 IMG_FILENAME = IMG_PATH + IMG_NAME + IMG_EXT;

    Mat img = imread(IMG_FILENAME);
//    imshow("Imagen " + IMG_NAME, img);

    Mat rgb_xyz     = RGB_2_XYZ(img),
        xyz_lms     = XYZ_2_LMS(img),
        rgb_lms     = RGB_2_LMS(img),
        rgb_log_lms = RGB_2_logLMS(img),
        lms_lalbe   = LMS_2_LAlBe(img),
        lalbe_lms   = LAlBe_2_logLMS(rgb_log_lms),
        lms_rgb_1   = logLMS_2_RGB(rgb_log_lms),
        lms_rgb_2   = logLMS_2_RGB(rgb_log_lms),
        lms_rgb_3   = logLMS_2_RGB(rgb_log_lms);

   cout <<     xyz_lms.channels() << endl
        <<     rgb_lms.channels() << endl
        << rgb_log_lms.channels() << endl
        <<   lms_lalbe.channels() << endl
        <<   lalbe_lms.channels() << endl
        <<   lms_rgb_1.channels() << endl
        <<   lms_rgb_2.channels() << endl
        <<   lms_rgb_3.channels() << endl;

    cout <<     xyz_lms.at<Vec3f>(0) << endl
         <<     rgb_lms.at<Vec3f>(0) << endl
         << rgb_log_lms.at<Vec3f>(0) << endl
         <<   lms_lalbe.at<Vec3f>(0) << endl
         <<   lalbe_lms.at<Vec3f>(0) << endl
         <<   lms_rgb_1.at<Vec3f>(0) << endl
         <<   lms_rgb_2.at<Vec3f>(0) << endl
         <<   lms_rgb_3.at<Vec3f>(0) << endl;

    imshow("lms_rgb_1", lms_rgb_1);
    imshow("lms_rgb_2", lms_rgb_2);
    imshow("lms_rgb_3", lms_rgb_3);

    waitKey();
    return 0;
}

Mat RGB_2_XYZ(const Mat &src) {
   Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        0.5141, 0.3239, 0.1604,
        0.2651, 0.6702, 0.0641,
        0.0241, 0.1228, 0.8444
    );

    Mat srcCopy;
    src.convertTo(srcCopy, CV_32FC3);
    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < srcCopy.rows; i++) {
        Vec3f *row = (Vec3f *) srcCopy.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < srcCopy.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat XYZ_2_LMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
         0.3897, 0.6890, -0.0787,
        -0.2298, 1.1834,  0.0464,
         0.0000, 0.0000,  1.0000
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat RGB_2_LMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        0.3811, 0.5783, 0.0402,
        0.1967, 0.7244, 0.0782,
        0.0241, 0.1288, 0.8444
    );

    Mat srcCopy;
    src.convertTo(srcCopy, CV_32FC3);
    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < srcCopy.rows; i++) {
        Vec3f *row = (Vec3f *) srcCopy.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < srcCopy.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat RGB_2_logLMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        0.3811, 0.5783, 0.0402,
        0.1967, 0.7244, 0.0782,
        0.0241, 0.1288, 0.8444
    );

    Mat srcCopy;
    src.convertTo(srcCopy, CV_32FC3);
    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < srcCopy.rows; i++) {
        Vec3f *row = (Vec3f *) srcCopy.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < srcCopy.cols; j++) {
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
            for(int k = 0; k < 3; k++)
                out[j][k] = log10f(out[j][k]);
        }
    }

    return output;
}

Mat LMS_2_LAlBe(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        fast_inv_sqrt(3), 0,                0,
        0,                fast_inv_sqrt(6), 0,
        0,                0,                fast_inv_sqrt(2)
    ),
    M_CONV_2 = (Mat_<float>(3, 3) <<
        1,  1,  1,
        1,  1, -2,
        1, -1,  0
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *)    src.ptr<Vec3f>(i),
              *out = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV * M_CONV_2, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat LAlBe_2_logLMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        1,  1,  1,
        1,  1, -1,
        1, -2,  0
    ),
    M_CONV_2 = (Mat_<float>(3, 3) <<
        sqrt(3)/3, 0,         0,
        0,         sqrt(6)/6, 0,
        0,         0,         sqrt(2)/2
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV * M_CONV_2, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat logLMS_2_RGB(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
         4.4679, -3.5873,  0.1193,
        -1.2186,  2.3809, -0.1624,
         0.0497, -0.2439,  1.2045
    );

    Mat lms_normal;
    pow(src, 10, lms_normal);
    cout << "Before power 10: " << src.at<Vec3b>(0) << endl
         << "After power 10: " << lms_normal.at<Vec3b>(0) << endl;
    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}
