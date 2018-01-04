//
// Created by sooda on 2018/1/3.
//

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "net.h"
#include "box.h"
#include "region_layer.h"

struct Object{
    cv::Rect rec;
    int class_id;
    float prob;
};

const char* class_names[] = {"background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};


static int detect_yolo(cv::Mat& raw_img, float show_threshold)
{
    ncnn::Net yolo;
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
    yolo.load_param("yolo.proto");
    yolo.load_model("yolo.bin");
    int input_size = 416;
    int width = raw_img.cols;
    int height = raw_img.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0/256,1.0/256,1.0/256};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;

    ncnn::Extractor ex=yolo.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, in);
    ex.extract("layer27-conv", out);
    int outsize=13;
    int cell_dim = 35;
    int class_num = 2;
    ncnn::Mat out1(out.reshape(outsize*outsize*cell_dim));

    int iw=width,ih=height,nw=input_size,nh=input_size,w=outsize,h=outsize,c=cell_dim,num=5,classes=class_num;

    //float biases[10]={0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741};
    std::vector<float> detectOut = region_forward(out1.data,iw,ih,nw,nh,w,h,c,classes,4,num,0.45, 0.5);

    int object_offset = 6;
    int obj_num = detectOut.size() / object_offset;
    for (int i = 0; i < obj_num; i++) {
        int offset = i * object_offset;
        int id = detectOut[offset];
        float conf = detectOut[offset+1];
        int left = detectOut[offset+2] * width;
        int top = detectOut[offset+3] * height;
        int right = detectOut[offset+4] * width;
        int bottom = detectOut[offset+5] * height;
        cv::rectangle(raw_img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0));
    }


    std::cout << "done" << std::endl;

    cv::imshow("result",raw_img);
    cv::waitKey();

    return 0;
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    detect_yolo(m,0.45);

    return 0;
}
