//
// Created by Maksim Levental on 11/18/20.
//

#include <cuda_helper.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pascal_generic.h>
#include <sstream>
#include <tinyxml2.h>
#include <util.h>
#include <vector>
#include "progress_bar.h"

PASCALGeneric::PASCALGeneric(const std::string &root_dir, Split split)
    : image_dir_(root_dir + "/" + "JPEGImages"), annotation_dir_(root_dir + "/" + "Annotations"),
      split_(split) {
    height_ = 200;
    width_ = 200;

    std::vector<std::string> all_file_names{};
    for (auto &p : std::filesystem::directory_iterator(image_dir_)) {
        auto img_fn = get_file_name(p.path());
        all_file_names.emplace_back(img_fn.substr(0, img_fn.length() - 4));
    }

    std::vector<std::string> relevant_file_names;
    if (split == Split::train_) {
        std::vector<std::string> temp(
            all_file_names.begin(), all_file_names.begin() + int(all_file_names.size() * 0.8));
        relevant_file_names = temp;
    } else {
        std::vector<std::string> temp(
            all_file_names.begin() + int(all_file_names.size() * 0.8), all_file_names.end());
        relevant_file_names = temp;
    }

    progresscpp::ProgressBar progressBar(relevant_file_names.size(), 70);
    tinyxml2::XMLDocument doc;
    for (auto &img_fn : relevant_file_names) {
        // x["annotation"]["object"][0]["name"]
        auto xml_fp = string_format("%s/%s.xml", annotation_dir_.c_str(), img_fn.c_str());
        doc.LoadFile(xml_fp.c_str());
        const char *category = doc.FirstChildElement("annotation")
                                   ->FirstChildElement("object")
                                   ->FirstChildElement("name")
                                   ->ToElement()
                                   ->GetText();
        auto img_fp = string_format("%s/%s.jpg", image_dir_.c_str(), img_fn.c_str());
        sample_fps_labels.emplace_back(std::string(img_fp), object_categories[category]);
        ++progressBar;
        progressBar.display();
    }
    progressBar.done();
}

std::tuple<cv::Mat, int> PASCALGeneric::operator[](unsigned int idx) {
    auto [sample_fp, label] = sample_fps_labels[idx];
//    printf("%s, %d\n", sample_fp.c_str(), label);
    auto img = cv2_image(sample_fp);
    if (img.empty()) {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Size_<int> new_size{height_, width_};
    cv::resize(img, img, new_size);
    return std::make_tuple(img, label);
}
int PASCALGeneric::len() { return sample_fps_labels.size(); }
