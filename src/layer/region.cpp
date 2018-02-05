//
// Created by sooda on 2018/1/4.
//

#include "region.h"
#include <math.h>

namespace ncnn {

    DEFINE_LAYER_CREATOR(Region)

    static inline float logistic_activate(float x){return 1./(1. + exp(-x));}

    typedef struct{
        float x, y, w, h;
    } box;
    typedef struct{
        float dx, dy, dw, dh;
    } dbox;
    typedef struct{
        int index;
        int classc;
        float **probs;
    } sortable_bbox;


    dbox derivative(box a, box b)
    {
        dbox d;
        d.dx = 0;
        d.dw = 0;
        float l1 = a.x - a.w/2;
        float l2 = b.x - b.w/2;
        if (l1 > l2){
            d.dx -= 1;
            d.dw += .5;
        }
        float r1 = a.x + a.w/2;
        float r2 = b.x + b.w/2;
        if(r1 < r2){
            d.dx += 1;
            d.dw += .5;
        }
        if (l1 > r2) {
            d.dx = -1;
            d.dw = 0;
        }
        if (r1 < l2){
            d.dx = 1;
            d.dw = 0;
        }

        d.dy = 0;
        d.dh = 0;
        float t1 = a.y - a.h/2;
        float t2 = b.y - b.h/2;
        if (t1 > t2){
            d.dy -= 1;
            d.dh += .5;
        }
        float b1 = a.y + a.h/2;
        float b2 = b.y + b.h/2;
        if(b1 < b2){
            d.dy += 1;
            d.dh += .5;
        }
        if (t1 > b2) {
            d.dy = -1;
            d.dh = 0;
        }
        if (b1 < t2){
            d.dy = 1;
            d.dh = 0;
        }
        return d;
    }

    float overlap(float x1, float w1, float x2, float w2)
    {
        float l1 = x1 - w1/2;
        float l2 = x2 - w2/2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1/2;
        float r2 = x2 + w2/2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    float box_intersection(box a, box b)
    {
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        if(w < 0 || h < 0) return 0;
        float area = w*h;
        return area;
    }

    float box_union(box a, box b)
    {
        float i = box_intersection(a, b);
        float u = a.w*a.h + b.w*b.h - i;
        return u;
    }

    float box_iou(box a, box b)
    {
        return box_intersection(a, b)/box_union(a, b);
    }

    dbox dintersect(box a, box b)
    {
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        dbox dover = derivative(a, b);
        dbox di;

        di.dw = dover.dw*h;
        di.dx = dover.dx*h;
        di.dh = dover.dh*w;
        di.dy = dover.dy*w;

        return di;
    }

    dbox dunion(box a, box b)
    {
        dbox du;

        dbox di = dintersect(a, b);
        du.dw = a.h - di.dw;
        du.dh = a.w - di.dh;
        du.dx = -di.dx;
        du.dy = -di.dy;

        return du;
    }

    int nms_comparator(const void *pa, const void *pb)
    {
        sortable_bbox a = *(sortable_bbox *)pa;
        sortable_bbox b = *(sortable_bbox *)pb;
        float diff = a.probs[a.index][b.classc] - b.probs[b.index][b.classc];
        if(diff < 0) return 1;
        else if(diff > 0) return -1;
        return 0;
    }

    void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh)
    {
        int i, j, k;
        sortable_bbox *s = (sortable_bbox*)malloc(sizeof(sortable_bbox)*total);//calloc(total, sizeof(sortable_bbox));

        for(i = 0; i < total; ++i){
            s[i].index = i;
            s[i].classc = classes;
            s[i].probs = probs;
        }

        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][classes] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    for(k = 0; k < classes+1; ++k){
                        probs[s[j].index][k] = 0;
                    }
                }
            }
        }
        free(s);
    }

    void activate_array(float *x, const int n)
    {
        int i;
        for(i = 0; i < n; ++i){
            x[i] = logistic_activate(x[i]);
        }
    }

    int entry_index(int location, int entry,int w,int h,int coords,int classes)
    {
        int n =   location / (w*h);
        int loc = location % (w*h);
        return n*w*h*(coords+classes+1) + entry*w*h + loc;
    }

    void softmax(float *input, int n, float temp, int stride, float *output)
    {
        int i;
        float sum = 0;
        float largest = -__FLT_MAX__;
        for(i = 0; i < n; ++i){
            if(input[i*stride] > largest) largest = input[i*stride];
        }
        for(i = 0; i < n; ++i){
            float e = exp(input[i*stride]/temp - largest/temp);
            sum += e;
            output[i*stride] = e;
        }
        for(i = 0; i < n; ++i){
            output[i*stride] /= sum;
        }
    }


    void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
    {
        int g, b;
        for(b = 0; b < batch; ++b){
            for(g = 0; g < groups; ++g){
                softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
            }
        }
    }

    void forward_act(float *input,float *output,int w,int h,int c,int classes,int coords,int num){
        for (int n=0;n<num;n++){
            int index = entry_index(n*w*h,0,w,h,coords,classes);
            activate_array(output+index,2*w*h);
            index = entry_index(n*w*h,coords,w,h,coords,classes);
            activate_array(output+index,w*h);
        }
        int index = entry_index(0,coords+1,w,h,coords,classes);
        softmax_cpu(input + index, classes, num, w*h*c/num, w*h, 1, w*h, 1, output + index);
    }

    box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
    {
        box b;
        b.x = (i + x[index + 0*stride]) / w;
        b.y = (j + x[index + 1*stride]) / h;
        b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
        b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;

        return b;
    }
    void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
    {
        int i;
        int new_w=0;
        int new_h=0;
        if (((float)netw/w) < ((float)neth/h)) {
            new_w = netw;
            new_h = (h * netw)/w;
        } else {
            new_h = neth;
            new_w = (w * neth)/h;
        }
        for (i = 0; i < n; ++i){
            box b = boxes[i];
            b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
            b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
            b.w *= (float)netw/new_w;
            b.h *= (float)neth/new_h;

            boxes[i] = b;
        }
    }
    void get_region_boxes(float thresh, float **probs, box *boxes,int relative,int w,int h,int c,int classes,int coords,int num,float *biases,float *output)
    {

        int i,j,n;
        float *predictions = output;

        for (i = 0; i < w*h; ++i){
            int row = i / w;
            int col = i % w;
            for(n = 0; n < num; ++n){
                int index = n*w*h + i;
                for(j = 0; j < classes; ++j){
                    probs[index][j] = 0;
                }
                int obj_index = entry_index(n*w*h + i, coords,w,h,coords,classes);
                int box_index = entry_index(n*w*h + i, 0,w,h,coords,classes);
                float scale = predictions[obj_index];
                boxes[index] = get_region_box(predictions, biases, n, box_index, col, row, w, h, w*h);

                float max = 0;
                for(j = 0; j < classes; ++j){
                    int class_index = entry_index(n*w*h + i, coords + 1 + j,w,h,coords,classes);
                    float prob = scale*predictions[class_index];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if(prob > max) max = prob;
                }
                probs[index][classes] = max;
            }
        }
    }

    void free_ptrs(void **ptrs, int n)
    {
        int i;
        for(i = 0; i < n; ++i) free(ptrs[i]);
        free(ptrs);
    }

    std::vector<float> region_forward(float * input,int w,int h,int c,int classes,int coords,int num, float* biases, float nms_thresh, float conf_thresh){
        float *output;
        output = (float*)calloc(w*h*c,sizeof(float));
        memcpy(output,input,w*h*c*sizeof(float));
        forward_act(input,output,w,h,c,classes,coords,num);
        box *boxes = (box*)malloc(sizeof(box)*w*h*num);
        float **probs = (float**)malloc(sizeof(float*)*w*h*num);
        for(int j = 0; j < w*h*num; ++j) probs[j] = (float*)malloc(sizeof(float)*(classes+1));
        get_region_boxes(conf_thresh,probs,boxes,0,w,h,c,classes,coords,num,biases,output);
        //if you want to correct letter box, do this outside
        //correct_region_boxes(boxes, w*h*num, iw, ih, nw, nh, 1);
        if (nms_thresh)
            do_nms_obj(boxes, probs, w*h*num,classes, nms_thresh);

        std::vector<float> detectionOut;

        for(int i = 0;i<w*h*num;i++)
        {
            float max_prob = 0;
            int class_index = 0;
            for (int j = 0; j < classes; j++) {
                if (probs[i][j] > max_prob) {
                    max_prob = probs[i][j];
                    class_index = j;
                }
            }
            if (max_prob < conf_thresh) {
                continue;
            }
            box b = boxes[i];
            detectionOut.push_back(class_index);
            detectionOut.push_back(max_prob);
            detectionOut.push_back(b.x - b.w/2);
            detectionOut.push_back(b.y - b.h/2);
            detectionOut.push_back(b.x + b.w/2);
            detectionOut.push_back(b.y + b.h/2);
        }

        free(boxes);
        free_ptrs((void **)probs, w*h*num);
        free(output);
        return detectionOut;
    }

    Region::Region()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    int Region::load_param(const ParamDict& pd)
    {
        anchors_ = pd.get(0, Mat());
        class_num_ = pd.get(1, 20);
        nms_thresh_ = pd.get(1, 0.45f);
        conf_thresh_ = pd.get(2, 0.2f);
        return 0;
    }


    int Region::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
    {
        const Mat& rawdata = bottom_blobs[0];
        float * biases = (float*)anchors_.data;
        int anchor_num = anchors_.w / 2;
        int w = rawdata.w;
        int h = rawdata.h;
        int c = rawdata.c;
        Mat out = rawdata.reshape(rawdata.w * rawdata.h * rawdata.c);
        std::vector<float> detectOut = region_forward((float*)out.data,w,h,c,class_num_,4, anchor_num, biases, nms_thresh_, conf_thresh_);
        int object_offset = 6;
        int num_detected = detectOut.size() / object_offset;

        Mat& top_blob = top_blobs[0];
        top_blob.create(object_offset, num_detected);

        for (int i = 0; i < num_detected; i++) {
            float* outptr = top_blob.row(i);
            int offset = i * object_offset;
            int id = detectOut[offset];
            float conf = detectOut[offset+1];
            outptr[0] = id;
            outptr[1] = conf;
            outptr[2] = detectOut[offset+2];
            outptr[3] = detectOut[offset+3];
            outptr[4] = detectOut[offset+4];
            outptr[5] = detectOut[offset+5];
        }
        return 0;
    }
}