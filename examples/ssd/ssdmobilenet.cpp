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
#include <fstream>

#include "net.h"

#define mxnet_ssd true

struct Object{
    cv::Rect rec;
    int class_id;
    float prob;
};

std::string getFileString(const std::string& filepath) {
    std::ifstream is(filepath);
    std::string filebuffer="";
    if (is.is_open()) {
        // get length of file:
        is.seekg (0, is.end);
        long long length = is.tellg();
        is.seekg (0, is.beg);
        char * buffer = new char [length];
        std::cout << "Reading " << filepath << " " << length << " characters... ";
        // read data as a block:
        is.read (buffer,length);
        if (is)
            std::cout << "all characters read successfully." << std::endl;
        else
            std::cout << "error: only " << is.gcount() << " could be read";
        is.close();
        // ...buffer contains the entire file...
        filebuffer = std::string(buffer,length);
        delete[] buffer;
    } else {
        std::cout << filepath << "open faild in getFileString" << std::endl;
    }
    return filebuffer;
}

const char* class_names[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

static int detect_mobilenet(cv::Mat& raw_img, float show_threshold)
{
    ncnn::Net mobilenet;
    /*
     * model is  converted from https://github.com/chuanqi305/MobileNet-SSD
     * and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
     */
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
    std::string proto_str;
    std::string bin_str;

#if mxnet_ssd
//        mobilenet.load_param("ncnn.proto");
//        mobilenet.load_model("ncnn.bin");
    proto_str = getFileString("ncnn.proto.bin");
    bin_str = getFileString("ncnn.bin");
    mobilenet.load_param((const unsigned char*) proto_str.data());
    mobilenet.load_model((const unsigned char*) bin_str.data());
//    mobilenet.load_param("vgg_textboxes.proto");
//    mobilenet.load_model("vgg_textboxes.bin");
#else
    mobilenet.load_param("TextBoxes.proto");

    mobilenet.load_model("TextBoxes.bin");
//        mobilenet.load_param("mobilenet_ssd_voc_ncnn.param");
//        mobilenet.load_model("mobilenet_ssd_voc_ncnn.bin");
#endif

//    mobilenet.load_param("ncnn.proto");
//    mobilenet.load_model("ncnn.bin");

    int input_size = 300;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);
    //ncnn::Mat in = ncnn::Mat::from_pixels(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows);
#if mxnet_ssd
    const float mean_vals[3] = {123, 117, 104};
    const float norm_vals[3] = {1.0,1.0,1.0};
#else
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
#endif
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;


    clock_t a = clock();
    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, in);
    ex.extract(161, out);

    std::cout << "time: " << (clock() - a) * 1000.0 / CLOCKS_PER_SEC << std::endl;


    printf("%d %d %d\n", out.w, out.h, out.c);
    std::vector<Object> objects;
    for (int iw=0;iw<out.h;iw++)
    {
        Object object;
        const float *values = out.row(iw);
        object.class_id = values[0];
        object.prob = values[1];
        object.rec.x = values[2] * img_w;
        object.rec.y = values[3] * img_h;
        object.rec.width = values[4] * img_w - object.rec.x;
        object.rec.height = values[5] * img_h - object.rec.y;
        objects.push_back(object);
    }

    for(int i = 0;i<objects.size();++i)
    {
        Object object = objects.at(i);
        if(object.prob > show_threshold)
        {
            std::cout << i << " " << object.prob << std::endl;
            cv::rectangle(raw_img, object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str<<object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(raw_img, cv::Rect(cv::Point(object.rec.x, object.rec.y- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);
            cv::putText(raw_img, label, cv::Point(object.rec.x, object.rec.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
#if mxnet_ssd
    cv::cvtColor(raw_img, raw_img, cv::COLOR_RGB2BGR);
#endif
    cv::imwrite("result.jpg", raw_img);
    cv::imshow("result",raw_img);
    cv::waitKey();

    return 0;
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
#if mxnet_ssd
    cv::cvtColor(m, m, cv::COLOR_BGR2RGB);
#endif
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    detect_mobilenet(m,0.2);

    return 0;
}
