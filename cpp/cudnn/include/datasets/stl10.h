//
// Created by Maksim Levental on 11/10/20.
//

#ifndef PROJECTNAME_STL10_H
#define PROJECTNAME_STL10_H

#define NUMBER_STL10_CLASSES 10

#include "dataset.h"

template <typename dtype>
class STL10 : public Dataset<dtype> {
public:
    STL10(
        const string &dataset_fp,
        const string &label_fp,
        bool shuffle,
        int batch_size,
        int num_classes);
    [[nodiscard]] int get_num_batches() const override;

protected:
    void load_data() override;
    void load_target() override;
    void normalize_data() override;
};

#endif // PROJECTNAME_STL10_H
