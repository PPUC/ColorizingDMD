#include "image_processing.h"

#include <algorithm>

void ApplyBrightnessContrastAndBlur(cv::Mat& mat, int brightness, int contrast, int blur)
{
    cv::Mat floatImage;
    mat.convertTo(floatImage, CV_32F);

    double alpha = 1 + 0.05 * contrast;
    if (alpha == 1.0) {
        alpha = 1.001;
    }
    double beta = 2.5 * brightness;
    floatImage = alpha * floatImage + beta;

    if (blur > 0) {
        cv::blur(floatImage, floatImage, cv::Size(blur, blur));
    }

    floatImage.convertTo(mat, mat.type());
}

cv::Mat PrepareImageMat(const cv::Mat& src, int brightness, int contrast, int blur)
{
    if (src.empty()) {
        return cv::Mat();
    }

    cv::Mat mat = src.clone();
    ApplyBrightnessContrastAndBlur(mat, brightness, contrast, blur);

    if (mat.cols % 4 != 0) {
        float ratio = static_cast<float>(mat.cols) / static_cast<float>(mat.rows);
        int cols = mat.cols - (mat.cols % 4) + 4;
        int rows = static_cast<int>(static_cast<float>(cols) / ratio);
        cv::Mat tmat;
        cv::resize(mat, tmat, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
        mat = tmat;
    }

    cv::Mat bgr;
    switch (mat.channels()) {
    case 3:
        bgr = mat;
        break;
    case 4:
        cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
        break;
    default:
        return cv::Mat();
    }

    if (bgr.depth() != CV_8U) {
        return cv::Mat();
    }

    return bgr;
}

cv::Mat LoadAndPrepareImage(const char* filename, int brightness, int contrast, int blur)
{
    cv::Mat mat = cv::imread(filename);
    return PrepareImageMat(mat, brightness, contrast, blur);
}

cv::Mat GetFrameAtTime(cv::VideoCapture& cap, long hour, long minute, long second, long frame, long* out_frame_rate)
{
    double fps = cap.get(cv::CAP_PROP_FPS);
    long frame_rate = static_cast<long>(fps + 0.5);
    if (frame_rate <= 0) {
        frame_rate = 1;
    }
    if (out_frame_rate) {
        *out_frame_rate = frame_rate;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double time_in_seconds = hour * 3600 + minute * 60 + second + (static_cast<double>(frame) / frame_rate);
    int position_in_frames = static_cast<int>(time_in_seconds * fps);
    if (total_frames > 0) {
        position_in_frames = std::max(0, std::min(total_frames - 1, position_in_frames));
    }
    cap.set(cv::CAP_PROP_POS_FRAMES, position_in_frames);

    cv::Mat frame_mat;
    if (!cap.read(frame_mat)) {
        return cv::Mat();
    }
    return frame_mat;
}

cv::Mat LoadAndPrepareVideoFrame(cv::VideoCapture& cap, long hour, long minute, long second, long frame, int brightness, int contrast, int blur, long* out_frame_rate)
{
    cv::Mat mat = GetFrameAtTime(cap, hour, minute, second, frame, out_frame_rate);
    return PrepareImageMat(mat, brightness, contrast, blur);
}

bool IsImageFile(const char* filename)
{
    cv::Mat mat = cv::imread(filename);
    return !mat.empty();
}

bool CanOpenVideoFile(const char* filename)
{
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        return false;
    }
    cv::Mat frame;
    cap >> frame;
    return !frame.empty();
}
