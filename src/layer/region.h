//
// Created by sooda on 2018/1/4.
//

#ifndef NCNN_REGION_H
#define NCNN_REGION_H
#include "layer.h"

namespace ncnn {

    class Region : public Layer
    {
    public:
        Region();

        virtual int load_param(const ParamDict& pd);

        virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

    public:
        Mat anchors_;
        int class_num_;
        float nms_thresh_;
        float conf_thresh_;
    };

} // namespace ncnn

#endif //NCNN_REGION_H
