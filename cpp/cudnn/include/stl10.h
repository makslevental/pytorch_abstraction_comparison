//
// Created by Maksim Levental on 11/10/20.
//

#ifndef PROJECTNAME_STL10_H
#define PROJECTNAME_STL10_H

#include "dataset.h"

class STL10 : public Dataset {
public:
    STL10(
        const string &dataset_fp,
        const string &label_fp,
        bool shuffle,
        int batch_size,
        int channels,
        int height,
        int width,
        int num_classes);
    [[nodiscard]] int get_num_batches() const override;

protected:
    void load_data() override;
    void load_target() override;
    void normalize_data() override;
};

#endif // PROJECTNAME_STL10_H
