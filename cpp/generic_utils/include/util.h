//
// Created by Maksim Levental on 11/18/20.
//

#ifndef PYTORCH_ABSTRACTION_UTIL_H
#define PYTORCH_ABSTRACTION_UTIL_H

#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

template <typename... Args> std::string string_format(const std::string &format, Args... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

cv::Mat cv2_image(const std::string &fp) {
    cv::Mat image = imread(fp, cv::IMREAD_COLOR);
    return image;
}

std::string get_file_name(const std::string &s) {

    char sep = '/';

    size_t i = s.rfind(sep, s.length());
    if (i != std::string::npos) {
        return (s.substr(i + 1, s.length() - i));
    }

    return "";
}

#endif // PYTORCH_ABSTRACTION_UTIL_H
