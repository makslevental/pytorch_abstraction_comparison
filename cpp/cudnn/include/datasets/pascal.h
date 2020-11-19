//
// Created by Maksim Levental on 11/10/20.
//

#ifndef PROJECTNAME_PASCAL_H
#define PROJECTNAME_PASCAL_H

#define NUMBER_PASCAL_CLASSES 20

#include "dataset.h"
#include "pascal_generic.h"

enum Mode { kTrain, kVal };

template <typename dtype> class PASCAL : public Dataset<dtype> {
public:
    PASCAL(const string &root_dir, Mode mode, bool shuffle, int batch_size, int num_classes);
    [[nodiscard]] int get_num_batches() const override;

protected:
    void load_data() override;
    void load_target() override;
    void normalize_data() override;
    PASCALGeneric pascal_generic_;
};


#endif // PROJECTNAME_PASCAL_H
