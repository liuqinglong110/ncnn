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
#include "image.h"
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

void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}

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
    ncnn::Mat in = ncnn::Mat::from_pixels(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0/256,1.0/256,1.0/256};
    in.substract_mean_normalize(mean_vals, norm_vals);

    image im;

    im.data = in.data;
    im.w = width;
    im.h = height;
    im.c=3;
    image sized = letterbox_image(im, input_size, input_size);


    ncnn::Mat inresize(input_size,input_size,3,sized.data);
    //
    ncnn::Mat out;

    ncnn::Extractor ex=yolo.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, inresize);
    ex.extract("layer27-conv", out);
    int outsize=13;
    int cell_dim = 35;
    int class_num = 2;
    ncnn::Mat out1(out.reshape(outsize*outsize*cell_dim));

    int iw=width,ih=height,nw=input_size,nh=input_size,w=outsize,h=outsize,c=cell_dim,num=5,classes=class_num,strid=35;
    box *boxes = (box*)malloc(sizeof(box)*w*h*num);//calloc(w*h*num, sizeof(box));
    float **probs = (float**)malloc(sizeof(float*)*w*h*num);//calloc(w*h*num, sizeof(float *));
    for(int j = 0; j < w*h*num; ++j) probs[j] = (float*)malloc(sizeof(float)*(classes+1));//calloc(classes + 1, sizeof(float *));
    float biases[10]={0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
    region_forward(out1.data,iw,ih,nw,nh,w,h,c,classes,4,num,0.1,boxes,probs,biases);

    for(int i = 0;i<w*h*num;i++)
    {
        float max_prob = 0;
        int class_index = 0;
        for (int j = 0; j <= class_num; j++) {
            if (probs[i][j] > max_prob) {
                max_prob = probs[i][j];
                class_index = j;
                std::cout << "max_prob: " << max_prob << std::endl;
            }
        }
        if (max_prob < show_threshold) {
            continue;
        }

        cv::Rect rect;
        box b = boxes[i];
        rect.x = (b.x - b.w/2) * width;
        rect.y = (b.y - b.y/2) * height;
        rect.width = b.w * width;
        rect.height = b.h * height;
        cv::rectangle(raw_img, rect, cv::Scalar(255, 255, 0));
    }

    std::cout << "done" << std::endl;

    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, w*h*num);

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
