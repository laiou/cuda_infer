int gpu_index = 0;
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <assert.h>  
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <device_launch_parameters.h>
#include "cuda_head.h"


void cuda_set_device(int n)
{
    gpu_index = n;
    //没有使用则默认使用device 0作为默认设备
    //用来设置代码在哪个设备上运行
    cudaError_t status = cudaSetDevice(n);
    //判断是否设置成功
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}




//在gpu上分配内存
float *cuda_make_array(float *x, size_t n)
{
    
    float *x_gpu;
    
    size_t size = sizeof(float)*n;
    
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
  
    check_error(status);
 
    if(x){
 
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}
//从设备上拷贝数据到主机
void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}


//释放设备内存
void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_free_int(int *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

//load image 

void fill_cpu_1(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void free_detections_1(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob); 
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}


static float get_pixel_1(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static float get_pixel_extend_1(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if(c < 0 || c >= m.c) return 0;
    return get_pixel_1(m, x, y, c);
}

static void set_pixel_1(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

static void add_pixel_1(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image make_empty_image_1(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void free_image_1(image m)
{
    if(m.data){
        free(m.data);
    }
}

image make_image_1(int w, int h, int c)
{
    image out = make_empty_image_1(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}


image load_image_stb_1(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image_1(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}


image resize_image_1(image im, int w, int h)
{
    image resized = make_image_1(w, h, im.c);   
    image part = make_image_1(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel_1(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel_1(im, ix, r, k) + dx * get_pixel_1(im, ix+1, r, k);
                }
                set_pixel_1(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel_1(part, c, iy, k);
                set_pixel_1(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel_1(part, c, iy+1, k);
                add_pixel_1(resized, c, r, k, val);
            }
        }
    }

    free_image_1(part);
    return resized;
}


image load_image_1(char *filename, int w, int h, int c)
{

    image out = load_image_stb_1(filename, c);

    if((h && w) && (h != out.h || w != out.w)){
        image resized = resize_image_1(out, w, h);
        free_image_1(out);
        out = resized;
    }
    return out;
}

image load_image_color_1(char *filename, int w, int h)
{
    return load_image_1(filename, w, h, 3);
}

image **load_alphabet_1()
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color_1(buff, 0, 0);
        }
    }
    return alphabets;
}

void fill_image_1(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void embed_image_1(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel_1(source, x,y,k);
                set_pixel_1(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

image letterbox_image_1(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image_1(im, new_w, new_h);
    image boxed = make_image_1(w, h, im.c);
    fill_image_1(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image_1(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    free_image_1(resized);
 // printf("%d,%d,%d\n",boxed.w,boxed.h,boxed.c);
    return boxed;
}

//nms
int nms_comparator_1(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float overlap_1(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection_1(box a, box b)
{
    float w = overlap_1(a.x, a.w, b.x, b.w);
    float h = overlap_1(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union_1(box a, box b)
{
    float i = box_intersection_1(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou_1(box a, box b)
{
    return box_intersection_1(a, b)/box_union_1(a, b);
}

void do_nms_sort_1(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_1);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou_1(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

float colors_1[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color_1(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors_1[i][c] + ratio*colors_1[j][c];
    //printf("%f\n", r);
    return r;
}

void draw_box_1(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for(i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void draw_box_width_1(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i){
        draw_box_1(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

void draw_bbox_1(image a, box bbox, int w, float r, float g, float b)
{
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    int i;
    for(i = 0; i < w; ++i){
        draw_box_1(a, left+i, top+i, right-i, bot-i, r, g, b);
    }
}

image copy_image_1(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void composite_image_1(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel_1(source, x, y, k);
                float val2 = get_pixel_extend_1(dest, dx+x, dy+y, k);
                set_pixel_1(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}

image tile_images_1(image a, image b, int dx)
{
    if(a.w == 0) return copy_image_1(b);
printf("11\n");
    image c = make_image_1(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
printf("22\n");
printf("image c size %d  %d  %d\n",c.w,c.h,c.c);
    fill_cpu_1(c.w*c.h*c.c, 1, c.data, 1);
printf("33\n");
    embed_image_1(a, c, 0, 0); 
printf("44\n");
    composite_image_1(b, c, a.w + dx, 0);
printf("55\n");
    return c;
}

image border_image_1(image a, int border)
{
    image b = make_image_1(a.w + 2*border, a.h + 2*border, a.c);
    int x,y,k;
    for(k = 0; k < b.c; ++k){
        for(y = 0; y < b.h; ++y){
            for(x = 0; x < b.w; ++x){
                float val = get_pixel_extend_1(a, x - border, y - border, k);
                if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel_1(b, x, y, k, val);
            }
        }
    }
    return b;
}

image get_label_1(image **characters, char *string, int size)
{
    size = size/10;
    if(size > 7) size = 7;
    image label = make_empty_image_1(0,0,0);
//'dog'
    int i = 0;
    while(*string){
 
        image l = characters[size][(int)*string];

        image n = tile_images_1(label, l, -size - 1 + (size+1)/2);

        free_image_1(label);
        label = n;
        ++string;
        i = i+1;

    }
    image b = border_image_1(label, label.h*.25);
    free_image_1(label);
    return b;
}

void draw_label_1(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel_1(label, i, j, k);
                set_pixel_1(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
}

image float_to_image_1(int w, int h, int c, float *data)
{
    image out = make_empty_image_1(w,h,c);
    out.data = data;
    return out;
}


image threshold_image_1(image im, float thresh)
{
    int i;
    image t = make_image_1(im.w, im.h, im.c);
    for(i = 0; i < im.w*im.h*im.c; ++i){
        t.data[i] = im.data[i]>thresh ? 1 : 0;
    }
    return t;
}

//void draw_detections_1(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
void draw_detections_1(image im, detection *dets, int num, float thresh, char **names, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
       printf("classes %d\n",class);
printf("num %d\n",i);
        if(class >= 0){
            
            int width = im.h * .006;

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            int offset = class*123457 % classes;
            float red = get_color_1(2,offset,classes);
            float green = get_color_1(1,offset,classes);
            float blue = get_color_1(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width_1(im, left, top, right, bot, width, red, green, blue);
		/*
            if (alphabet) {
		printf("%s\n",labelstr);
                image label = get_label_1(alphabet, labelstr, (im.h*.03));

                draw_label_1(im, top + width, left, label, rgb);
                free_image_1(label);
            }*/
            if (dets[i].mask){
                image mask = float_to_image_1(14, 14, 1, dets[i].mask);
                image resized_mask = resize_image_1(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image_1(resized_mask, .5);
                embed_image_1(tmask, im, left, top);
                free_image_1(mask);
                free_image_1(resized_mask);
                free_image_1(tmask);
            }
        }
    }
}

void save_image_options_1(image im, const char *name, IMTYPE f, int quality)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    if(f == PNG)       sprintf(buff, "%s.png", name);
    else if (f == BMP) sprintf(buff, "%s.bmp", name);
    else if (f == TGA) sprintf(buff, "%s.tga", name);
    else if (f == JPG) sprintf(buff, "%s.jpg", name);
    else               sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = 0;
    if(f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
    else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
    else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}


void save_image_1(image im, const char *name)
{
    save_image_options_1(im, name, JPG, 80);
}





int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}




//卷积，bn,激活层的数据处理

float im2col_get_pixel_1(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}


cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        //cublasCreate用来初始化cuBLAS库的上下文句柄，初始化的句柄会传递给后续库函数使用
        //使用完毕调用cublasDestroy()销毁句柄
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

void gemm_1_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{

    cublasHandle_t handle = blas_handle();
  
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}



void batchnorm_layer_1_gpu(float *layer_output,int batch,float *biases, int output_w,int output_h,int output_c,float *mean,float *variance,float *scales){
	
    normalize_1_gpu(layer_output, mean, variance, batch, output_c, output_h*output_w);
    scale_bias_1_gpu(layer_output, scales, batch, output_c, output_h*output_w);
    add_bias_1_gpu(layer_output, biases, batch, output_c, output_h*output_w);

}

//conv+bn+activation层的操作,也就是cfg中的一个[convolutional]单元，返回下一层的输入尺寸
//output_size里面对应的是下一层输入的w,h,c
void conv_bn_activation_layer_gpu(float *layer_input,int batch, int input_w, int input_h, int input_c,int activation, int kernel, int size, int stride, int padding,float *mean,float *variance,float *scales, float *weights, float *biases, float *workspace,float *layer_output){

	int i, j;
	int output_size[3];
	int outputs,output_w,output_h,output_c;
	output_w = (input_w + 2*padding - size) / stride + 1;
	output_h = (input_h + 2*padding - size) / stride + 1;
    output_c = kernel;
	output_size[0] = output_w;
	output_size[1] = output_h;
	output_size[2] = output_c;
//printf("%d,%d,%d\n",output_size[0],output_size[1],output_size[2]);
	//kernel is output_c,also is kernel number
	outputs = output_w*output_h*kernel;
    //fill_cpu_1(outputs*batch, 0, layer_output, 1);
    fill_gpu(outputs*batch, 0, layer_output, 1);
    
    int m = kernel;
    int k = size*size*input_c;
    int n = output_w*output_h;

    for(i = 0; i < batch; ++i){
        for(j = 0; j < 1; ++j){
            float *a = weights + j*(input_c*kernel*size*size);
            float *b = workspace;
            float *c = layer_output + (i + j)*n*m;
            float *im =  layer_input + (i + j)*input_c*input_h*input_w;

            if (size == 1) {
                b = im;
            } else {
                im2col_gpu_1(im, input_c, input_h, input_w, size, stride, padding, b);
            }

            gemm_1_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    //bn层的操作,偏置在这里面写了
    batchnorm_layer_1_gpu(layer_output,batch,biases,output_w,output_h,output_c,mean,variance,scales);
   
    //激活层的操作
    activate_layer_1_gpu(layer_output, outputs*batch, activation);
    //return output_size;

}



void conv_activation_layer_gpu(float *layer_input,int batch, int input_w, int input_h, int input_c,int activation, int kernel, int size, int stride, int padding,float *mean,float *variance,float *scales, float *weights, float *biases, float *workspace,float *layer_output){

	int i, j;
	int output_size[3];
	int outputs,output_w,output_h,output_c;
	output_w = (input_w + 2*padding - size) / stride + 1;
	output_h = (input_h + 2*padding - size) / stride + 1;
	output_c = kernel;
	output_size[0] = output_w;
	output_size[1] = output_h;
	output_size[2] = output_c;
	outputs = output_w*output_h*kernel;
    fill_gpu(outputs*batch, 0, layer_output, 1);
    
    int m = kernel;
    int k = size*size*input_c;
    int n = output_w*output_h;

    for(i = 0; i < batch; ++i){
        for(j = 0; j < 1; ++j){
            float *a = weights + j*(input_c*kernel*size*size);
            float *b = workspace;
            float *c = layer_output + (i + j)*n*m;
            float *im =  layer_input + (i + j)*input_c*input_h*input_w;

            if (size == 1) {
                b = im;
            } else {
                im2col_gpu_1(im, input_c, input_h, input_w, size, stride, padding, b);
            }

            gemm_1_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    //bn层的操作,偏置在这里面写了
     add_bias_1_gpu(layer_output, biases, batch, output_c, output_h*output_w);
    //激活层的操作
    activate_layer_1_gpu(layer_output, outputs*batch, activation);
    //return output_size;

}

 

void copy_1_gpu(int N, float * X, int INCX, float * Y, int INCY)
{

    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}




void route_layer_2_gpu(float *layer_input,int input_w,int input_h,int input_c,float *layer_input_1,int input_w1,int input_h1,int input_c1,int batch,float *layer_output){
    int i, j;
    int offset = 0;
    int output_size[3];
    //这里的l.n表示有几个层参与
    //float *layers[2];
    //layers[0] = layer_input;
    //layers[1] = layer_input_1;
    int sizes[2];
    int inputsize_0 = input_w*input_h*input_c;
    int inputsize_1 = input_w1*input_h1*input_c1;
    sizes[0] = inputsize_0;
    sizes[1] = inputsize_1;
    int n = 2;
    int outputs = sizes[0]+sizes[1];
    copy_1_gpu(sizes[0],layer_input,1,layer_output+offset,1);
    offset += sizes[0];
    copy_1_gpu(sizes[1],layer_input_1,1,layer_output+offset,1);
    offset += sizes[1];
    /*
    for(i = 0; i < n; ++i){
        float *input = &layers[i];
        int input_size = sizes[i];
        for(j = 0; j < batch; ++j){
            copy(input_size, input + j*input_size, 1, layer_output + offset + j*outputs, 1);
        }
        offset += input_size;
    }
	*/
    //计算输出维度：通道是参与的所有层次的通道的相加，w,h和第一的层次的一致
    int output_w = input_w;
    int output_h = input_h;
    int output_c = input_c + input_c1;
    output_size[0] = output_w;
    output_size[1] = output_h;
    output_size[2] = output_c;
    //return output_size;
}


//yolo层的实现
void yolo_layer_1_gpu(float *layer_input, float *layer_output,int batch,int input_w, int input_h,int n,int classes)
{
    int output_w,output_h,output_c;
    int output_size[3];
    output_w = input_w;
    output_h = input_h;
    output_c = n*(classes+4+1);
    output_size[0] = input_w;
    output_size[1] = input_h;
    output_size[2] = output_c;
    int outputs = input_w*input_h*n*(classes+4+1);
  

    copy_1_gpu(batch*outputs, layer_input, 1, layer_output, 1);

    for (int b = 0; b < 1; ++b){
        for(int i =  0; i < 3; ++i){
            //printf("b  is %d\n",b);
           // printf("i is %d\n",i);
            int index = entry_index_1(b, i*13*13, 0);
            activate_layer_1_gpu(layer_output + index, 2*13*13, 3);
            index = entry_index_1(b, i*13*13, 4);
            activate_layer_1_gpu(layer_output + index, (1+80)*13*13, 3);
        }
    }


printf("yolo1_output_number  %d  \n",outputs);
    return output_size;

 }


void yolo_layer_2_gpu(float *layer_input, float *layer_output,int batch,int input_w, int input_h,int n,int classes)
{
    int output_w,output_h,output_c;
    int output_size[3];
    output_w = input_w;
    output_h = input_h;
    output_c = n*(classes+4+1);
    output_size[0] = input_w;
    output_size[1] = input_h;
    output_size[2] = output_c;
    int outputs = input_w*input_h*n*(classes+4+1);
   
    //memcpy(layer_output, layer_input, outputs*batch*sizeof(float));
    copy_1_gpu(batch*outputs, layer_input, 1, layer_output, 1);
    for (int b = 0; b < 1; ++b){
        for(int i =  0; i < 3; ++i){}
           
            int index = entry_index_2(b, i*26*26, 0);
//printf("yolo2_index_get %d\n",index);
            activate_layer_1_gpu(layer_output + index, 2*26*26, 3);
            index = entry_index_2(b, i*26*26, 4);
//printf("yolo2_index_get_1 %d\n",index);
            activate_layer_1_gpu(layer_output + index, (1+80)*26*26, 3);
        }
    }

    printf("yolo2_output_number  %d  \n",outputs);
    return output_size;

 }


int *upsample_layer_gpu(float *layer_input,float *layer_output,int batch,int stride,int input_w,int input_h,int input_c){
    int output_w,output_h,output_c;
    int scale = 1;
    int output_size[3];
    output_w = input_w*stride;
    output_h = input_h*stride;
    output_c = input_c;
    output_size[0] = input_w;
    output_size[1] = input_h;
    output_size[2] = output_c;
    int outputs = output_h*output_w*output_c;
    fill_gpu(outputs*batch, 0, layer_output, 1);
  
    upsample_gpu_1(layer_input, input_w, input_h, input_c, batch, stride, 1, scale, layer_output);

    return output_size;
    

}

int entry_index_1( int batch, int location, int entry)
{
    int n =   location / (13*13);
    int loc = location % (13*13);
    int outputs = 13*13*3*85;
    int index = 0;
    index = batch*outputs + n*13*13*(4+80+1) + entry*13*13 + loc;
//printf("index result %d\n",batch*outputs + n*13*13*(4+80+1) + entry*13*13 + loc);
return index;
    //printf("index result %d\n",batch*outputs + n*13*13*(4+80+1) + entry*13*13 + loc);
    //return batch*outputs + n*13*13*(4+80+1) + entry*13*13 + loc;
}

int entry_index_2( int batch, int location, int entry)
{
    int n =   location / (26*26);
    int loc = location % (26*26);
    int outputs = 26*26*3*85; 
    int index = 0;
    index = batch*outputs + n*26*26*(4+80+1) + entry*26*26 + loc;
    return index;
}



int yolo_num_detections_yolo1(float *yolo_output, float thresh)
{
    int i, n;
    int count_1 = 0;
    for (i = 0; i < 13*13; ++i){
        for(n = 0; n < 3; ++n){
            int obj_index  =  entry_index_1(0, n*13*13 + i, 4);
            //printf("yolo_1_objindex %d\n",obj_index);
            if(yolo_output[obj_index] > thresh){
                ++count_1;
            }
        }
    }
    return count_1;
}

int yolo_num_detections_yolo2(float *yolo_output, float thresh)
{
    int i, n;
    int count_2 = 0;
    for (i = 0; i < 26*26; ++i){
        for(n = 0; n < 3; ++n){
            int obj_index  = entry_index_2(0, n*26*26 + i, 4);
            if(yolo_output[obj_index] > thresh){
                ++count_2;
            }
        }
    }
    return count_2;
}


int num_detections_1(float *yolo_1_output,float *yolo_2_output, float thresh)
{
    int i;
    int s = 0;
    int s1 = 0;
    int s2 = 0;
    s1 = yolo_num_detections_yolo1(yolo_1_output, thresh);
 
    s2 = yolo_num_detections_yolo2(yolo_2_output, thresh);
    printf("s1 %d\n",s1);
    printf("s2 %d\n",s2);
    /*
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            printf("yolo layer index %d\n",i);
            s += yolo_num_detections(l, thresh);
        }
    }*/
    s = s1+s2;
    return s;
}



box get_yolo_box_1(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    printf("box %.3f %.3f %.3f %.3f \n",b.x,b.y,b.w,b.h);
    return b;
}

void correct_yolo_boxes_1(detection *dets, int n, int w, int h, int netw, int neth, int relative)
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
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


int get_yolo_detections_1(float *yolo_1_output, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = yolo_1_output;
    //printf("yolo_1_output %.2f \n",yolo_1_output[0]);
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    float biases [12] = {10,14,23,27,37,58,81,82,135,169,344,319};
    int mask0 [] = {3,4,5};
    for (i = 0; i < 13*13; ++i){
        int row = i / 13;
        int col = i % 13;
        for(n = 0; n < 3; ++n){
            int obj_index  = entry_index_1( 0, n*13*13 + i, 4);
            float objectness = predictions[obj_index];
            //printf(" porb obj %.3f\n",objectness);
            if(objectness <= thresh) continue;
            //printf(" porb obj %.3f\n",objectness);
            int box_index  = entry_index_1(0, n*13*13 + i, 0);
            //printf("mask %d\n",mask0[n]);
            dets[count].bbox = get_yolo_box_1(predictions, biases, mask0[n], box_index, col, row, 13, 13, netw, neth, 13*13);
            dets[count].objectness = objectness;
            dets[count].classes = 80;
            for(j = 0; j < 80; ++j){
                int class_index = entry_index_1( 0, n*13*13 + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
	//printf("runheh\n");
    correct_yolo_boxes_1(dets, count, w, h, netw, neth, relative);
    return count;
}





int get_yolo_detections_2(float *yolo_2_output, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = yolo_2_output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    float biases [12] = {10,14,23,27,37,58,81,82,135,169,344,319};
    int mask1 [] = {0,1,2};
    for (i = 0; i < 26*26; ++i){
        int row = i / 26;
        int col = i % 26;
        for(n = 0; n < 3; ++n){
            int obj_index  = entry_index_2( 0, n*26*26 + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index_2(0, n*26*26 + i, 0);
            dets[count].bbox = get_yolo_box_1(predictions, biases, mask1[n], box_index, col, row, 26, 26, netw, neth, 26*26);
            dets[count].objectness = objectness;
            dets[count].classes = 80;
            for(j = 0; j < 80; ++j){
                int class_index = entry_index_2( 0, n*26*26 + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes_1(dets, count, w, h, netw, neth, relative);
    return count;
}



void fill_network_boxes_1(float *yolo_1_output,float *yolo_2_output, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    int count = get_yolo_detections_1(yolo_1_output, w, h, 416, 416, thresh, map, relative, dets);
    dets += count;
    int count_1 = get_yolo_detections_2(yolo_2_output, w, h, 416, 416, thresh, map, relative, dets);
    dets += count_1;
/*
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
    }*/
}


detection *make_network_boxes_1(float *yolo_1_output,float *yolo_2_output, float thresh, int *num)
{
    //layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections_1(yolo_1_output,yolo_2_output, thresh);
    if(num) *num = nboxes;
    printf("nboxes test %d\n",nboxes);
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
    //dets[i].prob = calloc(l.classes, sizeof(float));
	dets[i].prob = calloc(80, sizeof(float));
        
    }
    return dets;
}


detection *get_network_boxes_1(float *yolo_1_output, float *yolo_2_output,int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes_1(yolo_1_output,yolo_2_output, thresh, num);
    fill_network_boxes_1(yolo_1_output,yolo_2_output, w, h, thresh, hier, map, relative, dets);
    return dets;
}




void test_detector(char *image_path,float thresh, float hier_thresh)
{

    //srand(2222222);

    float nms=.1;

    image im = load_image_color_1(image_path,0,0);
    image sized = letterbox_image_1(im, 416, 416);
    //layer l = net->layers[net->n-1];
    //设置cuda设备
    cuda_set_device(0);
    float *X = sized.data;
    //time=what_time_is_it_now();
    //给输入分配cuda内存并将图片数据传到gpu
    float *X_gpu = cuda_make_array(X,416*416*3);
    float *test_input = calloc(416*416*3,sizeof(float));
   cuda_pull_array(X_gpu,test_input,416*416*3);


    char *filename_1 = "../yolov3-tiny.weights";

    fprintf(stderr, "Loading weights from %s...\n", filename_1);

    fflush(stdout);
   
    FILE *fp = fopen(filename_1, "rb");

    // if(!fp) file_error(filename_1);
    int major;
    int minor;
    int revision;

    fread(&major, sizeof(int), 1, fp);
	 
    fread(&minor, sizeof(int), 1, fp);

    fread(&revision, sizeof(int), 1, fp);

    size_t iseen ;
    fread(&iseen, sizeof(size_t), 1, fp);
 
    //开始加载权重
    //conv_1,输入维度是416x416x3的图片
    //n是卷积核个数，size是卷积核尺寸，c是输入通道数
    //int num = l.c*l.n*l.size*l.size;
    int conv_num_1 = 3*16*3*3;
    float *conv_1_biases = calloc(16, sizeof(float));
    fread(conv_1_biases, sizeof(float), 16, fp);
    //float *cuda_make_array(float *x, size_t n)

    float *conv_1_biases_gpu = cuda_make_array(conv_1_biases,16);

    float *conv_1_scales = calloc(16,sizeof(float));
    fread(conv_1_scales, sizeof(float), 16, fp);

    float *conv_1_scales_gpu = cuda_make_array(conv_1_scales,16);

    float *conv_1_mean = calloc(16,sizeof(float));
    fread(conv_1_mean, sizeof(float), 16, fp);

    float *conv_1_mean_gpu = cuda_make_array(conv_1_mean,16);
    float *conv_1_variance = calloc(16,sizeof(float));
    fread(conv_1_variance, sizeof(float), 16, fp);

    float *conv_1_variance_gpu = cuda_make_array(conv_1_variance,16);
    float *conv_1_weights = calloc(conv_num_1,sizeof(float));
    fread(conv_1_weights, sizeof(float), conv_num_1, fp);

    float *conv_1_weights_gpu = cuda_make_array(conv_1_weights,conv_num_1);

    //conv_2,输入维度是 (l.w + 2*l.pad - l.size) / l.stride + 1
    //也就是(320+2*1-3)/1 +1 = 320
    //输入维度320x320x32
    //num = 当前层的输入通道x卷积核个数x卷积核尺寸x卷积核尺寸
    int conv_num_2 = 16*32*3*3;
    float *conv_2_biases = calloc(32, sizeof(float));
    fread(conv_2_biases, sizeof(float), 32, fp);

    float *conv_2_biases_gpu = cuda_make_array(conv_2_biases,32);
    float *conv_2_scales = calloc(32,sizeof(float));
    fread(conv_2_scales, sizeof(float), 32, fp);

    float *conv_2_scales_gpu = cuda_make_array(conv_2_scales,32);
    float *conv_2_mean = calloc(32,sizeof(float));
    fread(conv_2_mean, sizeof(float), 32, fp);

    float *conv_2_mean_gpu = cuda_make_array(conv_2_mean,32);
    float *conv_2_variance = calloc(32,sizeof(float));
    fread(conv_2_variance, sizeof(float), 32, fp);

    float *conv_2_variance_gpu = cuda_make_array(conv_2_variance,32);
    float *conv_2_weights = calloc(conv_num_2,sizeof(float));
    fread(conv_2_weights, sizeof(float), conv_num_2, fp);

    float *conv_2_weights_gpu = cuda_make_array(conv_2_weights,conv_num_2);

    //printf("conv_2_weights %.3f\n",conv_2_weights[conv_num_2-1]);
    //printf("conv_index %d\n",conv_num_2);
    //conv_3
    int conv_num_3 = 32*64*3*3;
    float *conv_3_biases = calloc(64, sizeof(float));
    fread(conv_3_biases, sizeof(float), 64, fp);

    float *conv_3_biases_gpu = cuda_make_array(conv_3_biases,64);
    float *conv_3_scales = calloc(64,sizeof(float));
    fread(conv_3_scales, sizeof(float), 64, fp);

    float *conv_3_scales_gpu = cuda_make_array(conv_3_scales,64);
    float *conv_3_mean = calloc(64,sizeof(float));
    fread(conv_3_mean, sizeof(float), 64, fp);

    float *conv_3_mean_gpu = cuda_make_array(conv_3_mean,64);
    float *conv_3_variance = calloc(64,sizeof(float));
    fread(conv_3_variance, sizeof(float), 64, fp);

    float *conv_3_variance_gpu = cuda_make_array(conv_3_variance,64);
    float *conv_3_weights = calloc(conv_num_3,sizeof(float));
    fread(conv_3_weights, sizeof(float), conv_num_3, fp);

    float *conv_3_weights_gpu = cuda_make_array(conv_3_weights,conv_num_3);

    //printf("conv_3_weights %.3f\n",conv_3_weights[conv_num_3-1]);
    //printf("conv_index %d\n",conv_num_3);
    //conv_4
    int conv_num_4 = 64*128*3*3;
    float *conv_4_biases = calloc(128, sizeof(float));
    fread(conv_4_biases, sizeof(float), 128, fp);

    float *conv_4_biases_gpu = cuda_make_array(conv_4_biases,128);
    float *conv_4_scales = calloc(128,sizeof(float));
    fread(conv_4_scales, sizeof(float), 128, fp);

    float *conv_4_scales_gpu = cuda_make_array(conv_4_scales,128);
    float *conv_4_mean = calloc(128,sizeof(float));
    fread(conv_4_mean, sizeof(float), 128, fp);

    float *conv_4_mean_gpu = cuda_make_array(conv_4_mean,128);
    float *conv_4_variance = calloc(128,sizeof(float));
    fread(conv_4_variance, sizeof(float), 128, fp);

    float *conv_4_variance_gpu = cuda_make_array(conv_4_variance,128);
    float *conv_4_weights = calloc(conv_num_4,sizeof(float));
    fread(conv_4_weights, sizeof(float), conv_num_4, fp);

    float *conv_4_weights_gpu = cuda_make_array(conv_4_weights,conv_num_4);

    //printf("conv_4_weights %.3f\n",conv_4_weights[conv_num_4-1]);
    //printf("conv_index %d\n",conv_num_4);
    //conv_5
    int conv_num_5 = 128*256*3*3;
    float *conv_5_biases = calloc(256, sizeof(float));
    fread(conv_5_biases, sizeof(float), 256, fp);

    float *conv_5_biases_gpu = cuda_make_array(conv_5_biases,256);
    float *conv_5_scales = calloc(256,sizeof(float));
    fread(conv_5_scales, sizeof(float), 256, fp);

    float *conv_5_scales_gpu = cuda_make_array(conv_5_scales,256);
    float *conv_5_mean = calloc(256,sizeof(float));
    fread(conv_5_mean, sizeof(float), 256, fp);

    float *conv_5_mean_gpu = cuda_make_array(conv_5_mean,256);
    float *conv_5_variance = calloc(256,sizeof(float));
    fread(conv_5_variance, sizeof(float), 256, fp);

    float *conv_5_variance_gpu = cuda_make_array(conv_5_variance,256);
    float *conv_5_weights = calloc(conv_num_5,sizeof(float));
    fread(conv_5_weights, sizeof(float), conv_num_5, fp);

    float *conv_5_weights_gpu = cuda_make_array(conv_5_weights,conv_num_5);

    //printf("conv_5_weights %.3f\n",conv_5_weights[conv_num_5-1]);
    //printf("conv_index %d\n",conv_num_5);
    //conv_6
    int conv_num_6 = 256*512*3*3;
    float *conv_6_biases = calloc(512, sizeof(float));
    fread(conv_6_biases, sizeof(float), 512, fp);

    float *conv_6_biases_gpu = cuda_make_array(conv_6_biases,512);
    float *conv_6_scales = calloc(512,sizeof(float));
    fread(conv_6_scales, sizeof(float), 512, fp);

    float *conv_6_scales_gpu = cuda_make_array(conv_6_scales,512);
    float *conv_6_mean = calloc(512,sizeof(float));
    fread(conv_6_mean, sizeof(float), 512, fp);

    float *conv_6_mean_gpu = cuda_make_array(conv_6_mean,512);
    float *conv_6_variance = calloc(512,sizeof(float));
    fread(conv_6_variance, sizeof(float), 512, fp);

    float *conv_6_variance_gpu = cuda_make_array(conv_6_variance,512);
    float *conv_6_weights = calloc(conv_num_6,sizeof(float));
    fread(conv_6_weights, sizeof(float), conv_num_6, fp);

    float *conv_6_weights_gpu = cuda_make_array(conv_6_weights,conv_num_6);

    //printf("conv_6_weights %.3f\n",conv_6_weights[conv_num_6-1]);
    //printf("conv_index %d\n",conv_num_6);
    //conv_7
    int conv_num_7 = 512*1024*3*3;
    float *conv_7_biases = calloc(1024, sizeof(float));
    fread(conv_7_biases, sizeof(float), 1024, fp);

    float *conv_7_biases_gpu = cuda_make_array(conv_7_biases,1024);
    float *conv_7_scales = calloc(1024,sizeof(float));
    fread(conv_7_scales, sizeof(float), 1024, fp);

    float *conv_7_scales_gpu = cuda_make_array(conv_7_scales,1024);
    float *conv_7_mean = calloc(1024,sizeof(float));
    fread(conv_7_mean, sizeof(float), 1024, fp);

    float *conv_7_mean_gpu = cuda_make_array(conv_7_mean,1024);
    float *conv_7_variance = calloc(1024,sizeof(float));
    fread(conv_7_variance, sizeof(float), 1024, fp);

    float *conv_7_variance_gpu = cuda_make_array(conv_7_variance,1024);
    float *conv_7_weights = calloc(conv_num_7,sizeof(float));
    fread(conv_7_weights, sizeof(float), conv_num_7, fp);

    float *conv_7_weights_gpu = cuda_make_array(conv_7_weights,conv_num_7);

    //printf("conv_7_weights %.3f\n",conv_7_weights[conv_num_7-1]);
    //printf("conv_index %d\n",conv_num_7);
    //conv_8
    int conv_num_8 = 1024*256*1*1;  
    float *conv_8_biases = calloc(256, sizeof(float));
    fread(conv_8_biases, sizeof(float), 256, fp);

    float *conv_8_biases_gpu = cuda_make_array(conv_8_biases,256);
    float *conv_8_scales = calloc(256,sizeof(float));
    fread(conv_8_scales, sizeof(float), 256, fp);

    float *conv_8_scales_gpu = cuda_make_array(conv_8_scales,256);
    float *conv_8_mean = calloc(256,sizeof(float));
    fread(conv_8_mean, sizeof(float), 256, fp);

    float *conv_8_mean_gpu = cuda_make_array(conv_8_mean,256);
    float *conv_8_variance = calloc(256,sizeof(float));
    fread(conv_8_variance, sizeof(float), 256, fp);

    float *conv_8_variance_gpu = cuda_make_array(conv_8_variance,256);
    float *conv_8_weights = calloc(conv_num_8,sizeof(float));
    fread(conv_8_weights, sizeof(float), conv_num_8, fp);

    float *conv_8_weights_gpu = cuda_make_array(conv_8_weights,conv_num_8);

    //printf("conv_8_weights %.3f\n",conv_8_weights[conv_num_8-1]);
    //printf("conv_index %d\n",conv_num_8);
    //conv_9
    int conv_num_9 = 256*512*3*3;
    float *conv_9_biases = calloc(512, sizeof(float));
    fread(conv_9_biases, sizeof(float), 512, fp);

    float *conv_9_biases_gpu = cuda_make_array(conv_9_biases,512);
    float *conv_9_scales = calloc(512,sizeof(float));
    fread(conv_9_scales, sizeof(float), 512, fp);

    float *conv_9_scales_gpu = cuda_make_array(conv_9_scales,512);
    float *conv_9_mean = calloc(512,sizeof(float));
    fread(conv_9_mean, sizeof(float), 512, fp);

    float *conv_9_mean_gpu = cuda_make_array(conv_9_mean,512);
    float *conv_9_variance = calloc(512,sizeof(float));
    fread(conv_9_variance, sizeof(float), 512, fp);

    float *conv_9_variance_gpu = cuda_make_array(conv_9_variance,512);
    float *conv_9_weights = calloc(conv_num_9,sizeof(float));
    fread(conv_9_weights, sizeof(float), conv_num_9, fp);

    float *conv_9_weights_gpu = cuda_make_array(conv_9_weights,conv_num_9);

    //printf("conv_9_weights %.3f\n",conv_9_weights[conv_num_9-1]);
    //printf("conv_index %d\n",conv_num_9);
    //conv_10
    //这里没有BN操作
    int conv_num_10 = 512*255*1*1;
    float *conv_10_biases = calloc(255, sizeof(float));
    fread(conv_10_biases, sizeof(float), 255, fp);

    float *conv_10_biases_gpu = cuda_make_array(conv_10_biases,255);
    float *conv_10_scales = calloc(255,sizeof(float));
   // fread(conv_10_scales, sizeof(float), 255, fp);

    float *conv_10_scales_gpu = cuda_make_array(conv_10_scales,255);
    float *conv_10_mean = calloc(255,sizeof(float));

    float *conv_10_mean_gpu = cuda_make_array(conv_10_mean,255);
    //fread(conv_10_mean, sizeof(float), 255, fp);
    float *conv_10_variance = calloc(255,sizeof(float));
    //fread(conv_10_variance, sizeof(float), 255, fp);
    float *conv_10_variance_gpu = cuda_make_array(conv_10_variance,255);
    float *conv_10_weights = calloc(conv_num_10,sizeof(float));
    fread(conv_10_weights, sizeof(float), conv_num_10, fp);

    float *conv_10_weights_gpu = cuda_make_array(conv_10_weights,conv_num_10);

    //printf("conv_10_weights %.3f\n",conv_10_weights[conv_num_10-1]);
    //printf("conv_index %d\n",conv_num_10);
    //conv_11,有一个route
    int conv_num_11 = 256*128*1*1;
    float *conv_11_biases = calloc(128, sizeof(float));
    fread(conv_11_biases, sizeof(float), 128, fp);

    float *conv_11_biases_gpu = cuda_make_array(conv_11_biases,128);
    float *conv_11_scales = calloc(128,sizeof(float));
    fread(conv_11_scales, sizeof(float), 128, fp);

    float *conv_11_scales_gpu = cuda_make_array(conv_11_scales,128);
    float *conv_11_mean = calloc(128,sizeof(float));
    fread(conv_11_mean, sizeof(float), 128, fp);

    float *conv_11_mean_gpu = cuda_make_array(conv_11_mean,128);
    float *conv_11_variance = calloc(128,sizeof(float));
    fread(conv_11_variance, sizeof(float), 128, fp);

    float *conv_11_variance_gpu  =  cuda_make_array(conv_11_variance,128);
    float *conv_11_weights = calloc(conv_num_11,sizeof(float));
    fread(conv_11_weights, sizeof(float), conv_num_11, fp);

    float *conv_11_weights_gpu = cuda_make_array(conv_11_weights,conv_num_11);

    //printf("conv_11_weights %.3f\n",conv_11_weights[conv_num_11-1]);
    //printf("conv_index %d\n",conv_num_11);
    //conv_12,有一个route
    int conv_num_12 = 384*256*3*3;
    float *conv_12_biases = calloc(256, sizeof(float));
    fread(conv_12_biases, sizeof(float), 256, fp);

    float *conv_12_biases_gpu = cuda_make_array(conv_12_biases,256);
    float *conv_12_scales = calloc(256,sizeof(float));
    fread(conv_12_scales, sizeof(float), 256, fp);

    float *conv_12_scales_gpu = cuda_make_array(conv_12_scales,256);
    float *conv_12_mean = calloc(256,sizeof(float));
    fread(conv_12_mean, sizeof(float), 256, fp);

    float *conv_12_mean_gpu = cuda_make_array(conv_12_mean,256);
    float *conv_12_variance = calloc(256,sizeof(float));
    fread(conv_12_variance, sizeof(float), 256, fp);

    float *conv_12_variance_gpu = cuda_make_array(conv_12_variance,256);
    float *conv_12_weights = calloc(conv_num_12,sizeof(float));
    fread(conv_12_weights, sizeof(float), conv_num_12, fp);

    float *conv_12_weights_gpu =cuda_make_array(conv_12_weights,conv_num_12);

    //printf("conv_12_weights %.3f\n",conv_12_weights[conv_num_12-1]);
    //printf("conv_index %d\n",conv_num_12);
    //conv_13
    //这里没有BN操作
    int conv_num_13 = 256*255*1*1;
    float *conv_13_biases = calloc(255, sizeof(float));
    fread(conv_13_biases, sizeof(float), 255, fp);

    float *conv_13_biases_gpu = cuda_make_array(conv_13_biases,255);
    float *conv_13_scales = calloc(255,sizeof(float));

    float *conv_13_scales_gpu = cuda_make_array(conv_13_scales,255);
    //fread(conv_13_scales, sizeof(float), 255, fp);
    float *conv_13_mean = calloc(255,sizeof(float));

    float *conv_13_mean_gpu = cuda_make_array(conv_13_mean,255);
    //fread(conv_13_mean, sizeof(float), 255, fp);
    float *conv_13_variance = calloc(255,sizeof(float));

    float *conv_13_variance_gpu = cuda_make_array(conv_13_variance,255);
    //fread(conv_13_variance, sizeof(float), 255, fp);
    float *conv_13_weights = calloc(conv_num_13,sizeof(float));
    fread(conv_13_weights, sizeof(float), conv_num_13, fp);

    float *conv_13_weights_gpu = cuda_make_array(conv_13_weights,conv_num_13);


    fclose(fp);

 	//create_network
    //输入416x416的图片，绝对够了
    size_t workspace_size_1 = (size_t)416*416*3*3*1024*sizeof(float);
    float *workspace0 = calloc(1,workspace_size_1);

    //分配gpu上工作空间
    float *workspace0_gpu = cuda_make_array(workspace0,416*416*3*3*1024);
    int *conv_1_out = calloc(3,sizeof(int));
    //actination:0,1,2
 
    float *conv_1_output = calloc(416*416*16,sizeof(float));
    float *conv_1_output_gpu = cuda_make_array(conv_1_output,416*416*16);

    conv_bn_activation_layer_gpu(X_gpu,1,416,416,3,0,16,3,1,1,conv_1_mean_gpu,conv_1_variance_gpu,conv_1_scales_gpu,conv_1_weights_gpu,conv_1_biases_gpu,workspace0_gpu,conv_1_output_gpu); 
    
	cudaThreadSynchronize();

    float *maxpool_1_output = calloc(208*208*16,sizeof(float));
    int *indexes_gpu_1 = cuda_make_int_array(0, 208*208*16);
    float *maxpool_1_output_gpu = cuda_make_array(maxpool_1_output,208*208*16);
    
    maxpool_layer_gpu(conv_1_output_gpu,1,416,416,16,2,2,1,maxpool_1_output_gpu,indexes_gpu_1);

    cudaThreadSynchronize();

    float *conv_2_output = calloc(208*208*32,sizeof(float));
 
    float *conv_2_output_gpu = cuda_make_array(conv_2_output,208*208*32);
 
    cudaThreadSynchronize();
 
    conv_bn_activation_layer_gpu(maxpool_1_output_gpu,1,208,208,16,0,32,3,1,1,conv_2_mean_gpu,conv_2_variance_gpu,conv_2_scales_gpu,conv_2_weights_gpu,conv_2_biases_gpu,workspace0_gpu,conv_2_output_gpu);

    cudaThreadSynchronize();
         
    float *maxpool_2_output = calloc(104*104*32,sizeof(float));
    float *maxpool_2_output_gpu = cuda_make_array(maxpool_2_output,104*104*32);
    int *indexes_gpu_2 = cuda_make_int_array(0, 104*104*32);
 
    maxpool_layer_gpu(conv_2_output_gpu,1,208,208,32,2,2,1,maxpool_2_output_gpu,indexes_gpu_2);

    cudaThreadSynchronize();
         
    float *conv_3_output = calloc(104*104*64,sizeof(float));
    float *conv_3_output_gpu = cuda_make_array(conv_3_output,104*104*64);
     
    conv_bn_activation_layer_gpu(maxpool_2_output_gpu,1,104,104,32,0,64,3,1,1,conv_3_mean_gpu,conv_3_variance_gpu,conv_3_scales_gpu,conv_3_weights_gpu,conv_3_biases_gpu,workspace0_gpu,conv_3_output_gpu);

    cudaThreadSynchronize();
     
    float *maxpool_3_output = calloc(52*52*64,sizeof(float));
    float *maxpool_3_output_gpu = cuda_make_array(maxpool_3_output,52*52*64);
    int *indexes_gpu_3 = cuda_make_int_array(0, 52*52*64);
    
    maxpool_layer_gpu(conv_3_output_gpu,1,104,104,64,2,2,1,maxpool_3_output_gpu,indexes_gpu_3);

    cudaThreadSynchronize();
    
    float *conv_4_output = calloc(52*52*128,sizeof(float));
    float *conv_4_output_gpu = cuda_make_array(conv_4_output,52*52*128);
     
    conv_bn_activation_layer_gpu(maxpool_3_output_gpu,1,52,52,64,0,128,3,1,1,conv_4_mean_gpu,conv_4_variance_gpu,conv_4_scales_gpu,conv_4_weights_gpu,conv_4_biases_gpu,workspace0_gpu,conv_4_output_gpu);

    cudaThreadSynchronize();
         
    float *maxpool_4_output = calloc(26*26*128,sizeof(float));
    float *maxpool_4_output_gpu = cuda_make_array(maxpool_4_output,26*26*128);
    int *indexes_gpu_4 = cuda_make_int_array(0, 26*26*128);
     
    maxpool_layer_gpu(conv_4_output_gpu,1,52,52,128,2,2,1,maxpool_4_output_gpu,indexes_gpu_4);

    cudaThreadSynchronize();
   
    float *conv_5_output = calloc(26*26*256,sizeof(float));
    float *conv_5_output_gpu = cuda_make_array(conv_5_output,26*26*256);
     
    conv_bn_activation_layer_gpu(maxpool_4_output_gpu,1,26,26,128,0,256,3,1,1,conv_5_mean_gpu,conv_5_variance_gpu,conv_5_scales_gpu,conv_5_weights_gpu,conv_5_biases_gpu,workspace0_gpu,conv_5_output_gpu);

    cudaThreadSynchronize();
     
    float *maxpool_5_output = calloc(13*13*256,sizeof(float));
    float *maxpool_5_output_gpu = cuda_make_array(maxpool_5_output,13*13*256);
    int *indexes_gpu_5 = cuda_make_int_array(0,13*13*256);
     
    maxpool_layer_gpu(conv_5_output_gpu,1,26,26,256,2,2,1,maxpool_5_output_gpu,indexes_gpu_5);
 
    float *conv_6_output = calloc(13*13*512,sizeof(float));
    float *conv_6_output_gpu = cuda_make_array(conv_6_output,13*13*512);
     
    conv_bn_activation_layer_gpu(maxpool_5_output_gpu,1,13,13,256,0,512,3,1,1,conv_6_mean_gpu,conv_6_variance_gpu,conv_6_scales_gpu,conv_6_weights_gpu,conv_6_biases_gpu,workspace0_gpu,conv_6_output_gpu);

    cudaThreadSynchronize();
     
    float *maxpool_6_output = calloc(13*13*512,sizeof(float));
    float *maxpool_6_output_gpu = cuda_make_array(maxpool_6_output,13*13*512);
    int *indexes_gpu_6 = cuda_make_int_array(0, 13*13*512);
   
    maxpool_layer_gpu(conv_6_output_gpu,1,13,13,512,2,1,1,maxpool_6_output_gpu,indexes_gpu_6);
    cudaThreadSynchronize();

   
    float *conv_7_output = calloc(13*13*1024,sizeof(float));
    float *conv_7_output_gpu = cuda_make_array(conv_7_output,13*13*1024);
     
    conv_bn_activation_layer_gpu(maxpool_6_output_gpu,1,13,13,512,0,1024,3,1,1,conv_7_mean_gpu,conv_7_variance_gpu,conv_7_scales_gpu,conv_7_weights_gpu,conv_7_biases_gpu,workspace0_gpu,conv_7_output_gpu);

    cudaThreadSynchronize();
     
    float *conv_8_output = calloc(13*13*256,sizeof(float));
    float *conv_8_output_gpu = cuda_make_array(conv_8_output,13*13*256);
     
    conv_bn_activation_layer_gpu(conv_7_output_gpu,1,13,13,1024,0,256,1,1,0,conv_8_mean_gpu,conv_8_variance_gpu,conv_8_scales_gpu,conv_8_weights_gpu,conv_8_biases_gpu,workspace0_gpu,conv_8_output_gpu);

    cudaThreadSynchronize();
         
    float *conv_9_output = calloc(13*13*512,sizeof(float));
    float *conv_9_output_gpu = cuda_make_array(conv_9_output,13*13*512);
    
    
    conv_bn_activation_layer_gpu(conv_8_output_gpu,1,13,13,256,0,512,3,1,1,conv_9_mean_gpu,conv_9_variance_gpu,conv_9_scales_gpu,conv_9_weights_gpu,conv_9_biases_gpu,workspace0_gpu,conv_9_output_gpu);
    cudaThreadSynchronize();

    
    float *conv_10_output = calloc(13*13*255,sizeof(float));
    float *conv_10_output_gpu = cuda_make_array(conv_10_output,13*13*255);
    
    conv_activation_layer_gpu(conv_9_output_gpu,1,13,13,512,2,255,1,1,0,conv_10_mean_gpu,conv_10_variance_gpu,conv_10_scales_gpu,conv_10_weights_gpu,conv_10_biases_gpu,workspace0_gpu,conv_10_output_gpu);

    cudaThreadSynchronize();
     
    float *yolo_1_output = calloc(13*13*3*(80+4+1),sizeof(float));
    float *yolo_1_output_gpu = cuda_make_array(yolo_1_output,13*13*3*85);
     
    yolo_layer_1_gpu(conv_10_output_gpu,yolo_1_output_gpu,1,13,13,3,80);

    cudaThreadSynchronize();
        
    float *conv_11_output = calloc(13*13*128,sizeof(float));
    float *conv_11_output_gpu = cuda_make_array(conv_11_output,13*13*128);
 
    conv_bn_activation_layer_gpu(conv_8_output_gpu,1,13,13,256,0,128,1,1,0,conv_11_mean_gpu,conv_11_variance_gpu,conv_11_scales_gpu,conv_11_weights_gpu,conv_11_biases_gpu,workspace0_gpu,conv_11_output_gpu);

    cudaThreadSynchronize();
     
    float *upsample_1_output = calloc(26*26*128,sizeof(float));
    float *upsample_1_output_gpu = cuda_make_array(upsample_1_output,26*26*128);
  
    upsample_layer_gpu(conv_11_output_gpu,upsample_1_output_gpu,1,2,13,13,128);

    cudaThreadSynchronize();
    
    float *route_2_output = calloc(26*26*384,sizeof(float));
    float *route_2_output_gpu = cuda_make_array(route_2_output,26*26*384);
     
    route_layer_2_gpu(upsample_1_output_gpu,26,26,128,conv_5_output_gpu,26,26,256,1,route_2_output_gpu);

    cudaThreadSynchronize();
 
    float *conv_12_output = calloc(26*26*256,sizeof(float));
    float *conv_12_output_gpu = cuda_make_array(conv_12_output,26*26*256);
     
    conv_bn_activation_layer_gpu(route_2_output_gpu,1,26,26,384,0,256,3,1,1,conv_12_mean_gpu,conv_12_variance_gpu,conv_12_scales_gpu,conv_12_weights_gpu,conv_12_biases_gpu,workspace0_gpu,conv_12_output_gpu);

    cudaThreadSynchronize();
    
    float *conv_13_output = calloc(26*26*255,sizeof(float));
    float *conv_13_output_gpu = cuda_make_array(conv_13_output,26*26*256);
     
    conv_activation_layer_gpu(conv_12_output_gpu,1,26,26,256,2,255,1,1,0,conv_13_mean_gpu,conv_13_variance_gpu,conv_13_scales_gpu,conv_13_weights_gpu,conv_13_biases_gpu,workspace0_gpu,conv_13_output_gpu);

    cudaThreadSynchronize();
     
    float *yolo_2_output = calloc(26*26*3*(80+4+1),sizeof(float));
    float *yolo_2_output_gpu = cuda_make_array(yolo_2_output,26*26*3*85);
     
    yolo_layer_2_gpu(conv_13_output_gpu,yolo_2_output_gpu,1,26,26,3,80);

    cudaThreadSynchronize();
      
    //将yolo层输出复制到主机
    cuda_pull_array(yolo_1_output_gpu, yolo_1_output, 13*13*3*85);
    cuda_pull_array(yolo_2_output_gpu,yolo_2_output,26*26*3*85);

 
    //post processor
    char *coco_names[] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",   "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};   
    int nboxes = 0;
      
	detection *dets = get_network_boxes_1(yolo_1_output,yolo_2_output, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    printf("nboxes %d\n", nboxes);
    if (nms) do_nms_sort_1(dets, nboxes, 80, nms);
        
	draw_detections_1(im, dets, nboxes, thresh, coco_names,80);


    free_detections_1(dets, nboxes);

    save_image_1(im, "predictions");

    //free memory
    free_image_1(im);
    free_image_1(sized);
    free(conv_1_mean);
    free(conv_1_variance);
    free(conv_1_scales);
    free(conv_1_weights);
    free(conv_1_biases);
    //free(conv_1_out);
    free(conv_1_output);
    cuda_free(conv_1_mean_gpu);
    cuda_free(conv_1_variance_gpu);
    cuda_free(conv_1_scales_gpu);
    cuda_free(conv_1_weights_gpu);
    cuda_free(conv_1_biases_gpu);
    cuda_free(conv_1_output_gpu);

    free(conv_2_mean);
    free(conv_2_variance);
    free(conv_2_scales);
    free(conv_2_weights);
    free(conv_2_biases);
    //free(conv_2_out);
    free(conv_2_output);
    cuda_free(conv_2_mean_gpu);
    cuda_free(conv_2_variance_gpu);
    cuda_free(conv_2_scales_gpu);
    cuda_free(conv_2_weights_gpu);
    cuda_free(conv_2_biases_gpu);
    cuda_free(conv_2_output_gpu);

    free(conv_3_mean);
    free(conv_3_variance);
    free(conv_3_scales);
    free(conv_3_weights);
    free(conv_3_biases);
    //free(conv_3_out);
    free(conv_3_output);
    cuda_free(conv_3_mean_gpu);
    cuda_free(conv_3_variance_gpu);
    cuda_free(conv_3_scales_gpu);
    cuda_free(conv_3_weights_gpu);
    cuda_free(conv_3_biases_gpu);
    cuda_free(conv_3_output_gpu);

    free(conv_4_mean);
    free(conv_4_variance);
    free(conv_4_scales);
    free(conv_4_weights);
    free(conv_4_biases);
    //free(conv_4_out);
    free(conv_4_output);
    cuda_free(conv_4_mean_gpu);
    cuda_free(conv_4_variance_gpu);
    cuda_free(conv_4_scales_gpu);
    cuda_free(conv_4_weights_gpu);
    cuda_free(conv_4_biases_gpu);
    cuda_free(conv_4_output_gpu);


    free(conv_5_mean);
    free(conv_5_variance);
    free(conv_5_scales);
    free(conv_5_weights);
    free(conv_5_biases);
    //free(conv_5_out);
    free(conv_5_output);
    cuda_free(conv_5_mean_gpu);
    cuda_free(conv_5_variance_gpu);
    cuda_free(conv_5_scales_gpu);
    cuda_free(conv_5_weights_gpu);
    cuda_free(conv_5_biases_gpu);
    cuda_free(conv_5_output_gpu);


    free(conv_6_mean);
    free(conv_6_variance);
    free(conv_6_scales);
    free(conv_6_weights);
    free(conv_6_biases);
    //free(conv_6_out);
    free(conv_6_output);
    cuda_free(conv_6_mean_gpu);
    cuda_free(conv_6_variance_gpu);
    cuda_free(conv_6_scales_gpu);
    cuda_free(conv_6_weights_gpu);
    cuda_free(conv_6_biases_gpu);
    cuda_free(conv_6_output_gpu);


    free(conv_7_mean);
    free(conv_7_variance);
    free(conv_7_scales);
    free(conv_7_weights);
    free(conv_7_biases);
    //free(conv_7_out);
    free(conv_7_output);
    cuda_free(conv_7_mean_gpu);
    cuda_free(conv_7_variance_gpu);
    cuda_free(conv_7_scales_gpu);
    cuda_free(conv_7_weights_gpu);
    cuda_free(conv_7_biases_gpu);
    cuda_free(conv_7_output_gpu);


    free(conv_8_mean);
    free(conv_8_variance);
    free(conv_8_scales);
    free(conv_8_weights);
    free(conv_8_biases);
    //free(conv_8_out);
    free(conv_8_output);
    cuda_free(conv_8_mean_gpu);
    cuda_free(conv_8_variance_gpu);
    cuda_free(conv_8_scales_gpu);
    cuda_free(conv_8_weights_gpu);
    cuda_free(conv_8_biases_gpu);
    cuda_free(conv_8_output_gpu);


    free(conv_9_mean);
    free(conv_9_variance);
    free(conv_9_scales);
    free(conv_9_weights);
    free(conv_9_biases);
    //free(conv_9_out);
    free(conv_9_output);
    cuda_free(conv_9_mean_gpu);
    cuda_free(conv_9_variance_gpu);
    cuda_free(conv_9_scales_gpu);
    cuda_free(conv_9_weights_gpu);
    cuda_free(conv_9_biases_gpu);
    cuda_free(conv_9_output_gpu);


    free(conv_10_mean);
    free(conv_10_variance);
    free(conv_10_scales);
    free(conv_10_weights);
    free(conv_10_biases);
    //free(conv_10_out);
    free(conv_10_output);
    cuda_free(conv_10_mean_gpu);
    cuda_free(conv_10_variance_gpu);
    cuda_free(conv_10_scales_gpu);
    cuda_free(conv_10_weights_gpu);
    cuda_free(conv_10_biases_gpu);
    cuda_free(conv_10_output_gpu);


    free(conv_11_mean);
    free(conv_11_variance);
    free(conv_11_scales);
    free(conv_11_weights);
    free(conv_11_biases);
    //free(conv_11_out);
    free(conv_11_output);
    cuda_free(conv_11_mean_gpu);
    cuda_free(conv_11_variance_gpu);
    cuda_free(conv_11_scales_gpu);
    cuda_free(conv_11_weights_gpu);
    cuda_free(conv_11_biases_gpu);
    cuda_free(conv_11_output_gpu);


    free(conv_12_mean);
    free(conv_12_variance);
    free(conv_12_scales);
    free(conv_12_weights);
    free(conv_12_biases);
    //free(conv_12_out);
    free(conv_12_output);
    cuda_free(conv_12_mean_gpu);
    cuda_free(conv_12_variance_gpu);
    cuda_free(conv_12_scales_gpu);
    cuda_free(conv_12_weights_gpu);
    cuda_free(conv_12_biases_gpu);
    cuda_free(conv_12_output_gpu);


    free(conv_13_mean);
    free(conv_13_variance);
    free(conv_13_scales);
    free(conv_13_weights);
    free(conv_13_biases);
    //free(conv_13_out);
    free(conv_13_output);
    cuda_free(conv_13_mean_gpu);
    cuda_free(conv_13_variance_gpu);
    cuda_free(conv_13_scales_gpu);
    cuda_free(conv_13_weights_gpu);
    cuda_free(conv_13_biases_gpu);
    cuda_free(conv_13_output_gpu);


    free(workspace0);
    cuda_free(workspace0_gpu);

    //free(maxpool_1_out);
    free(maxpool_1_output);
    cuda_free(maxpool_1_output_gpu);
    cuda_free_int(indexes_gpu_1);

    //free(maxpool_2_out);
    free(maxpool_2_output);
    cuda_free(maxpool_2_output_gpu);
    cuda_free_int(indexes_gpu_2);

    //free(maxpool_3_out);
    free(maxpool_3_output);
    cuda_free(maxpool_3_output_gpu);
    cuda_free_int(indexes_gpu_3);

    //free(maxpool_4_out);
    free(maxpool_4_output);
    cuda_free(maxpool_4_output_gpu);
    cuda_free_int(indexes_gpu_4);

    //free(maxpool_5_out);
    free(maxpool_5_output);
    cuda_free(maxpool_5_output_gpu);
    cuda_free_int(indexes_gpu_5);

    //free(maxpool_6_out);
    free(maxpool_6_output);
    cuda_free(maxpool_6_output_gpu);
    cuda_free_int(indexes_gpu_6);

    //free(upsample_1_out);
    free(upsample_1_output);
    cuda_free(upsample_1_output_gpu);

    //free(route_1_out);
    //free(route_1_output);

    //free(route_2_out);
    free(route_2_output);
    cuda_free(route_2_output_gpu);

    //free(yolo_1_out);
    free(yolo_1_output);
    cuda_free(yolo_1_output_gpu);

    //free(yolo_2_out);
    free(yolo_2_output);
    cuda_free(yolo_2_output_gpu);

}



int main(int argc, char **argv)
{
    
    if (0 == strcmp(argv[1], "-infer")){

	 float thresh = 0.15 ;
        test_detector(argv[2],thresh, .5);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}

