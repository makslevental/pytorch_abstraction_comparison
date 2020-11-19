//
// Created by Maksim Levental on 11/18/20.
//

#ifndef PYTORCH_ABSTRACTION_PASCAL_GENERIC_H
#define PYTORCH_ABSTRACTION_PASCAL_GENERIC_H

#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum Split { train_, trainval, val };

struct PASCALGeneric {
    const int channels_ = 3;
    int height_;
    int width_;
    PASCALGeneric(const std::string &root_dir, Split split);
    std::tuple<cv::Mat, int> operator[](unsigned int idx);
    int len();

protected:
    Split split_;
    std::string image_dir_;
    std::string annotation_dir_;
    std::map<std::string, int> object_categories = {
        {"aeroplane", 0},    {"bicycle", 1}, {"bird", 2},   {"boat", 3},       {"bottle", 4},
        {"bus", 5},          {"car", 6},     {"cat", 7},    {"chair", 8},      {"cow", 9},
        {"diningtable", 10}, {"dog", 11},    {"horse", 12}, {"motorbike", 13}, {"person", 14},
        {"pottedplant", 15}, {"sheep", 16},  {"sofa", 17},  {"train", 18},     {"tvmonitor", 19},

    };
    std::vector<std::tuple<std::string, int>> sample_fps_labels{};
};

#endif // PYTORCH_ABSTRACTION_PASCAL_GENERIC_H
