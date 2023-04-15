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

    Mat rgb_xyz = RGB_2_XYZ(img);
    cout << "Original RGB Pixel: " <<     img.at<Vec3b>(0, 0) << endl
         << "RGB Pixel to XYZ: "   << rgb_xyz.at<Vec3f>(0, 0) << endl << endl;

    // Now is XYZ_2_logLMS
    Mat xyz_lms = XYZ_2_LMS(rgb_xyz);
    cout << "Original XYZ Pixel: " << rgb_xyz.at<Vec3f>(0, 0) << endl
         << "XYZ Pixel to LMS: "   << xyz_lms.at<Vec3f>(0, 0) << endl << endl;

    // Mat rgb_lms = RGB_2_LMS(img);
    // cout << "Original RGB Pixel: " <<     img.at<Vec3b>(0, 0) << endl
    //      << "RGB Pixel to LMS: "   << rgb_lms.at<Vec3f>(0, 0) << endl << endl;

    // Mat rgb_logLms = RGB_2_logLMS(img);
    // cout << "Original RGB Pixel: "  <<        img.at<Vec3b>(0, 0) << endl
    //      << "RGB Pixel to logLMS: " << rgb_logLms.at<Vec3f>(0, 0) << endl << endl;

    Mat logLms_lalbe = LMS_2_LAlBe(xyz_lms);
    cout << "Original logLMS Pixel: " <<      xyz_lms.at<Vec3f>(0, 0) << endl
         << "logLMS Pixel to LAlBe: " << logLms_lalbe.at<Vec3f>(0, 0) << endl << endl;

    Mat lalbe_logLms = LAlBe_2_logLMS(logLms_lalbe);
    cout << "LAlBe Pixel: "                << logLms_lalbe.at<Vec3f>(0, 0) << endl
         << "LAlBe Pixel back to logLMS: " << lalbe_logLms.at<Vec3f>(0, 0) << endl << endl;

    Mat lms_rgb = logLMS_2_RGB(xyz_lms);
    cout << "logLMS pixel: "          << xyz_lms.at<Vec3f>(0, 0) << endl
         << "logLMS^10 back to RGB: " << lms_rgb.at<Vec3b>(0, 0) << endl << endl;
    imshow("Hola", lms_rgb);

    Mat cv_rgb_xyz;
    cvtColor(img, cv_rgb_xyz, COLOR_RGB2XYZ);
    cv_rgb_xyz.convertTo(cv_rgb_xyz, CV_32FC3);
    cout << cv_rgb_xyz.type() << endl;
    cout << "Original RGB Pixel: " <<        img.at<Vec3b>(0, 0) << endl
         << "RGB Pixel to XYZ: "   << cv_rgb_xyz.at<Vec3f>(0, 0) << endl << endl;
         
    Mat cv_rgb_lab;
    cvtColor(img, cv_rgb_lab, COLOR_RGB2Lab);
    cv_rgb_lab.convertTo(cv_rgb_lab, CV_32FC3);
    cout << cv_rgb_lab.type() << endl;
    cout << "Original RGB Pixel: "  <<        img.at<Vec3b>(0, 0) << endl
         << "RGB Pixel to CIELab: " << cv_rgb_lab.at<Vec3f>(0, 0) << endl << endl;

    // Mat lms_rgb = logLMS_2_RGB(rgb_lms);
    // cout << "LMS pixel: "       << rgb_lms.at<Vec3f>(0, 0) << endl
    //      << "LMS back to RGB: " << lms_rgb.at<Vec3b>(0, 0) << endl << endl;
    // imshow("Hola", lms_rgb);

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
        0.4124f, 0.3576f, 0.1805f,
        0.2126f, 0.7152f, 0.0722f,
        0.0193f, 0.1192f, 0.9505f
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
         0.3897f, 0.6890f, -0.0787f,
        -0.2298f, 1.1834f,  0.0432f,
         0.0000f, 0.0000f,  1.0000f
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);
    
    // float x = 0, y = 0, z = 0,
    //       l = 0, m = 0, s = 0;
    // for(int i = 0; i < src.rows; i++) {
    //     for(int j = 0; j < src.cols; j++) {
    //         x = src.at<Vec3f>(i, j)[0];
    //         y = src.at<Vec3f>(i, j)[1];
    //         z = src.at<Vec3f>(i, j)[2];

    //         l = ( 0.3897 * x)  + (0.6890 * y) + (-0.0787 * z);
    //         m = (-0.2298 * x)  + (1.1834 * y) + ( 0.0432 * z);
    //         s = ( 0.0000 * x)  + (0.0000 * y) + ( 1.0000 * z);

    //         l = (l == 0.0000) ? 1.0000 : log10(l);
    //         m = (m == 0.0000) ? 1.0000 : log10(m);
    //         s = (s == 0.0000) ? 1.0000 : log10(s);

    //         output.at<Vec3f>(i, j)[0] = l;
    //         output.at<Vec3f>(i, j)[1] = m;
    //         output.at<Vec3f>(i, j)[2] = s;
    //     }
    // }
    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *)    src.ptr<Vec3f>(i),
              *out = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++) {
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
            for(int k = 0; k < 3; k++)
                out[j][k] = log10((out[j][k] == 0.0000 ? 1.0000 : 0.0) + out[j][k]);
        }
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
        0.3811f, 0.5783f, 0.0402f,
        0.1967f, 0.7244f, 0.0782f,
        0.0241f, 0.1288f, 0.8444f
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
        0.3811f, 0.5783f, 0.0402f,
        0.1967f, 0.7244f, 0.0782f,
        0.0241f, 0.1288f, 0.8444f
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
                out[j][k] = log10((out[j][k] == 0.0000 ? 1.0000 : 0.0) + out[j][k]);
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

    const float SQRT_3   = fast_inv_sqrt(3),
                SQRT_6   = fast_inv_sqrt(6),
                SQRT_2   = fast_inv_sqrt(2),
                SQRT_2_3 = -1 * sqrt(2.0/3.0);

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        SQRT_3,  SQRT_3,   SQRT_3,
        SQRT_6,  SQRT_6, SQRT_2_3, 
        SQRT_2, -SQRT_2,   0.0000
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);
    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *)    src.ptr<Vec3f>(i),
              *out = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat LAlBe_2_logLMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const float SQRT_3   = fast_inv_sqrt(3),
                SQRT_6   = fast_inv_sqrt(6),
                SQRT_2   = fast_inv_sqrt(2),
                SQRT_2_3 = -1 * sqrt(2.0/3.0);

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        SQRT_3,   SQRT_6,  SQRT_2,
        SQRT_3,   SQRT_6, -SQRT_2,
        SQRT_3, SQRT_2_3,  0.0000
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

Mat logLMS_2_RGB(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
         4.4679f, -3.5873f,  0.1193f,
        -1.2186f,  2.3809f, -0.1624f,
         0.0497f, -0.2439f,  1.2045f
    );

    // Raising values to power ten to go back to linear space
    Mat log_pow10;
    pow(src, 10, log_pow10);

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < log_pow10.rows; i++) {
        Vec3f *row = (Vec3f *) log_pow10.ptr<Vec3f>(i),
              *out = (Vec3f *)    output.ptr<Vec3f>(i);
        for(int j = 0; j < log_pow10.cols; j++) {
            for(int k = 0; k < output.channels(); k++)
                row[j][k] = (row[j][k] == 1.0000 ? 0 : row[j][k]);
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
        }
    }

    // cout << "\tlogLMS pixel: "                  <<       src.at<Vec3f>(0, 0) << endl
    //      << "\tlogLMS^10 pixel: "               << log_pow10.at<Vec3f>(0, 0) << endl
    //      << "\tlogLMS^10 back to RGB (float): " <<    output.at<Vec3f>(0, 0) << endl;

    output.convertTo(output, CV_8UC3);

    // cout << "\tRGB float pixel back to uchar: " << output.at<Vec3b>(0, 0) << endl;

    return output;
}
