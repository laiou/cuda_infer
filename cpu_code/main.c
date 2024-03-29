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
        printf("/////\n");
        printf("%s\n",string);
        image l = characters[size][(int)*string];
	printf("**\n");
        image n = tile_images_1(label, l, -size - 1 + (size+1)/2);
printf("^^^\n");
        free_image_1(label);
        label = n;
        ++string;
        i = i+1;
printf("i = %d\n",i);
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



void activate_layer_1(float *x,const int n,int y){
	if (y==0){
		leaky_relu_1(x,n);
	}else if(y==1){
		relu_1(x,n);
	}else if (y==2){
		linear_1(x,n);
	}else if (y==3){
		logistic(x,n);
	}else{
		printf("activation set err!\n");
	}

}

void logistic(float *x,const int n){
        int i;
//printf("logistic start...\n");
        for(i = 0;i<n;i++){
	x[i] = 1./(1. + exp(-x[i]));

}
//printf("logistci end...\n");
}

void linear_1(float *x,const int n){
	return x;
}

void leaky_relu_1(float *x,const int n){
	int i;
	for(i = 0;i< n;i++){
		x[i] = ((x[i]>0) ? x[i] : .1*x[i]);
	}

}
//darknet中mish没实现，用的还是relu
void relu_1(float *x,const int n){
	int i;
	for(i=0;i<n;i++){
		x[i] = x[i]*(x[i]>0);
	}

}

//maxpooling的操作
int* maxpool_layer(float *layer_input,int batch, int input_h, int input_w, int input_c, int size, int stride, int padding, float *layer_output){
	int b,i,j,k,m,n;
    int w_offset = -padding/2;
    int h_offset = -padding/2;

    int h = (input_h + padding - size)/stride + 1;
    int w = (input_w + padding - size)/stride + 1;
    int c = input_c;

    int output_size[3];
    int output_w;
    int output_h;
    int output_c;
    output_w = (input_w + padding - size)/stride + 1;
    output_h = (input_h + padding - size)/stride + 1;
    output_c = input_c;
    output_size[0] = output_w;
	output_size[1] = output_h;
	output_size[2] = output_c;

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < size; ++n){
                        for(m = 0; m < size; ++m){
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + input_w*(cur_h + input_h*(k + b*input_c));
                            int valid = (cur_h >= 0 && cur_h < input_h &&
                                         cur_w >= 0 && cur_w < input_w);
                            float val = (valid != 0) ? layer_input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    layer_output[out_index] = max;
                    //l.indexes[out_index] = max_i;
                }
            }
        }
    }

    return output_size;
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

void im2col_cpu_1(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;

    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
		//printf("%d\n",col_index);
		//printf("%d\n",im_col + width*(im_row + height*c_im));
                data_col[col_index] = im2col_get_pixel_1(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void gemm_nn_1(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_1(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
 
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    gemm_nn_1(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);

}

void normalize_1(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void scale_bias_1(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void add_bias_1(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void batchnorm_layer_1(float *layer_output,int batch,float *biases, int output_w,int output_h,int output_c,float *mean,float *variance,float *scales){
	
    normalize_1(layer_output, mean, variance, batch, output_c, output_h*output_w);
    scale_bias_1(layer_output, scales, batch, output_c, output_h*output_w);
    add_bias_1(layer_output, biases, batch, output_c, output_h*output_w);

}

//conv+bn+activation层的操作,也就是cfg中的一个[convolutional]单元，返回下一层的输入尺寸
//output_size里面对应的是下一层输入的w,h,c
int* conv_bn_activation_layer(float *layer_input,int batch, int input_w, int input_h, int input_c,int activation, int kernel, int size, int stride, int padding,float *mean,float *variance,float *scales, float *weights, float *biases, float *workspace,float *layer_output){

	int i, j;
	int output_size[3];
	int outputs,output_w,output_h,output_c;
	output_w = (input_w + 2*padding - size) / stride + 1;
	output_h = (input_h + 2*padding - size) / stride + 1;
	output_c = kernel;
	output_size[0] = output_w;
	output_size[1] = output_h;
	output_size[2] = output_c;
printf("%d,%d,%d\n",output_size[0],output_size[1],output_size[2]);
	//kernel is output_c,also is kernel number
	outputs = output_w*output_h*kernel;
    fill_cpu_1(outputs*batch, 0, layer_output, 1);
    
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
                im2col_cpu_1(im, input_c, input_h, input_w, size, stride, padding, b);
            }

            gemm_1(m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    //bn层的操作,偏置在这里面写了
    batchnorm_layer_1(layer_output,batch,biases,output_w,output_h,output_c,mean,variance,scales);
   
    //激活层的操作
    activate_layer_1(layer_output, outputs*batch, activation);
    return output_size;

}



int* conv_activation_layer(float *layer_input,int batch, int input_w, int input_h, int input_c,int activation, int kernel, int size, int stride, int padding,float *mean,float *variance,float *scales, float *weights, float *biases, float *workspace,float *layer_output){

	int i, j;
	int output_size[3];
	int outputs,output_w,output_h,output_c;
	output_w = (input_w + 2*padding - size) / stride + 1;
	output_h = (input_h + 2*padding - size) / stride + 1;
	output_c = kernel;
	output_size[0] = output_w;
	output_size[1] = output_h;
	output_size[2] = output_c;
printf("%d,%d,%d\n",output_size[0],output_size[1],output_size[2]);
	//kernel is output_c,also is kernel number
	outputs = output_w*output_h*kernel;
    fill_cpu_1(outputs*batch, 0, layer_output, 1);
    
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
                im2col_cpu_1(im, input_c, input_h, input_w, size, stride, padding, b);
            }

            gemm_1(m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    //bn层的操作,偏置在这里面写了
    //batchnorm_layer_1(layer_output,batch,biases,output_w,output_h,output_c,mean,variance,scales);
     add_bias_1(layer_output, biases, batch, output_c, output_h*output_w);
    //激活层的操作
    activate_layer_1(layer_output, outputs*batch, activation);
    return output_size;

}





void copy_1(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void shortcut_1(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    //assert(stride == h1/h2);
    //assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}
//接收上一层的输出和其他某一层的输出,这里alpha和beta都是1
//layer_input是当前层次正常的输入，layer_input_1是其他需要进行拼接到当前层的输出 
//input_w等指的是当前层上一层输出的维度，input_w1指的是其他拼接过来的层的维度
int* shortcut_layer(float *layer_input,float *layer_input_1,float *layer_output,int batch,int input_w,int input_h,int input_c,int input_w1,int input_h1,int input_c1,int y){
	int output_size[3];
    int alpha = 1;
    int beta = 1;
    int outputs,output_w,output_h,output_c;
    output_w = input_w;
    output_h = input_h;
    output_c = input_c;
    output_size[0] = output_w;
    output_size[1] = output_h;
    output_size[2] = output_c;
    outputs = output_w*output_h*output_c;
    copy_1(outputs*batch, layer_input, 1, layer_output, 1);
    shortcut_1(batch, input_w1, input_h1,input_c1,layer_input_1, output_w, output_h, output_c, alpha, beta, layer_output);
    activate_layer_1(layer_output, outputs*batch, y);
    return output_size;
}

//route层的实现,需要根据具体的情况实现多个，这里以拼接两个层次为例
void  route_layer_1(float *layer_input,int input_w,int input_h,int input_c,int batch,float *layer_output)
{
    int i, j;
    int offset = 0;
int input_size = input_w*input_h*input_c;
 copy_1(input_size, layer_input, 1, layer_output, 1);


}


int* route_layer_2(float *layer_input,int input_w,int input_h,int input_c,float *layer_input_1,int input_w1,int input_h1,int input_c1,int batch,float *layer_output){
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
    copy_1(sizes[0],layer_input,1,layer_output+offset,1);
    offset += sizes[0];
    copy_1(sizes[1],layer_input_1,1,layer_output+offset,1);
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
    return output_size;
}


//yolo层的实现
int *yolo_layer_1(float *layer_input, float *layer_output,int batch,int input_w, int input_h,int n,int classes)
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
 
    memcpy(layer_output, layer_input, outputs*batch*sizeof(float));

    for (int b = 0; b < 1; ++b){
        for(int i =  0; i < 3; ++i){
            //printf("b  is %d\n",b);
           // printf("i is %d\n",i);
            int index = entry_index_1(b, i*13*13, 0);
            activate_layer_1(layer_output + index, 2*13*13, 3);
            index = entry_index_1(b, i*13*13, 4);
            activate_layer_1(layer_output + index, (1+80)*13*13, 3);
        }
    }


printf("  %d  \n",outputs);
    return output_size;

 }


int *yolo_layer_2(float *layer_input, float *layer_output,int batch,int input_w, int input_h,int n,int classes)
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
   
    memcpy(layer_output, layer_input, outputs*batch*sizeof(float));
    for (int b = 0; b < 1; ++b){
        for(int i =  0; i < 3; ++i){
           
            int index = entry_index_2(b, i*26*26, 0);
//printf("yolo2_index_get %d\n",index);
            activate_layer_1(layer_output + index, 2*26*26, 3);
            index = entry_index_2(b, i*26*26, 4);
//printf("yolo2_index_get_1 %d\n",index);
            activate_layer_1(layer_output + index, (1+80)*26*26, 3);
        }
    }


printf("  %d  \n",outputs);
    return output_size;

 }

void upsample_cpu_1(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

int *upsample_layer(float *layer_input,float *layer_output,int batch,int stride,int input_w,int input_h,int input_c){
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
    fill_cpu_1(outputs*batch, 0, layer_output, 1);
  
    upsample_cpu_1(layer_input, input_w, input_h, input_c, batch, stride, 1, scale, layer_output);
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


    float *X = sized.data;

    char *filename_1 = "./yolov3-tiny.weights";

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

    float *conv_1_scales = calloc(16,sizeof(float));
    fread(conv_1_scales, sizeof(float), 16, fp);
    float *conv_1_mean = calloc(16,sizeof(float));
    fread(conv_1_mean, sizeof(float), 16, fp);
    float *conv_1_variance = calloc(16,sizeof(float));
    fread(conv_1_variance, sizeof(float), 16, fp);
    float *conv_1_weights = calloc(conv_num_1,sizeof(float));
    fread(conv_1_weights, sizeof(float), conv_num_1, fp);

    //conv_2,输入维度是 (l.w + 2*l.pad - l.size) / l.stride + 1
    //也就是(320+2*1-3)/1 +1 = 320
    //输入维度320x320x32
    //num = 当前层的输入通道x卷积核个数x卷积核尺寸x卷积核尺寸
    int conv_num_2 = 16*32*3*3;
    float *conv_2_biases = calloc(32, sizeof(float));
    fread(conv_2_biases, sizeof(float), 32, fp);
    float *conv_2_scales = calloc(32,sizeof(float));
    fread(conv_2_scales, sizeof(float), 32, fp);
    float *conv_2_mean = calloc(32,sizeof(float));
    fread(conv_2_mean, sizeof(float), 32, fp);
    float *conv_2_variance = calloc(32,sizeof(float));
    fread(conv_2_variance, sizeof(float), 32, fp);
    float *conv_2_weights = calloc(conv_num_2,sizeof(float));
    fread(conv_2_weights, sizeof(float), conv_num_2, fp);

    int conv_num_3 = 32*64*3*3;
    float *conv_3_biases = calloc(64, sizeof(float));
    fread(conv_3_biases, sizeof(float), 64, fp);
    float *conv_3_scales = calloc(64,sizeof(float));
    fread(conv_3_scales, sizeof(float), 64, fp);
    float *conv_3_mean = calloc(64,sizeof(float));
    fread(conv_3_mean, sizeof(float), 64, fp);
    float *conv_3_variance = calloc(64,sizeof(float));
    fread(conv_3_variance, sizeof(float), 64, fp);
    float *conv_3_weights = calloc(conv_num_3,sizeof(float));
    fread(conv_3_weights, sizeof(float), conv_num_3, fp);

    int conv_num_4 = 64*128*3*3;
    float *conv_4_biases = calloc(128, sizeof(float));
    fread(conv_4_biases, sizeof(float), 128, fp);
    float *conv_4_scales = calloc(128,sizeof(float));
    fread(conv_4_scales, sizeof(float), 128, fp);
    float *conv_4_mean = calloc(128,sizeof(float));
    fread(conv_4_mean, sizeof(float), 128, fp);
    float *conv_4_variance = calloc(128,sizeof(float));
    fread(conv_4_variance, sizeof(float), 128, fp);
    float *conv_4_weights = calloc(conv_num_4,sizeof(float));
    fread(conv_4_weights, sizeof(float), conv_num_4, fp);

    int conv_num_5 = 128*256*3*3;
    float *conv_5_biases = calloc(256, sizeof(float));
    fread(conv_5_biases, sizeof(float), 256, fp);
    float *conv_5_scales = calloc(256,sizeof(float));
    fread(conv_5_scales, sizeof(float), 256, fp);
    float *conv_5_mean = calloc(256,sizeof(float));
    fread(conv_5_mean, sizeof(float), 256, fp);
    float *conv_5_variance = calloc(256,sizeof(float));
    fread(conv_5_variance, sizeof(float), 256, fp);
    float *conv_5_weights = calloc(conv_num_5,sizeof(float));
    fread(conv_5_weights, sizeof(float), conv_num_5, fp);

    int conv_num_6 = 256*512*3*3;
    float *conv_6_biases = calloc(512, sizeof(float));
    fread(conv_6_biases, sizeof(float), 512, fp);
    float *conv_6_scales = calloc(512,sizeof(float));
    fread(conv_6_scales, sizeof(float), 512, fp);
    float *conv_6_mean = calloc(512,sizeof(float));
    fread(conv_6_mean, sizeof(float), 512, fp);
    float *conv_6_variance = calloc(512,sizeof(float));
    fread(conv_6_variance, sizeof(float), 512, fp);
    float *conv_6_weights = calloc(conv_num_6,sizeof(float));
    fread(conv_6_weights, sizeof(float), conv_num_6, fp);

    int conv_num_7 = 512*1024*3*3;
    float *conv_7_biases = calloc(1024, sizeof(float));
    fread(conv_7_biases, sizeof(float), 1024, fp);
    float *conv_7_scales = calloc(1024,sizeof(float));
    fread(conv_7_scales, sizeof(float), 1024, fp);
    float *conv_7_mean = calloc(1024,sizeof(float));
    fread(conv_7_mean, sizeof(float), 1024, fp);
    float *conv_7_variance = calloc(1024,sizeof(float));
    fread(conv_7_variance, sizeof(float), 1024, fp);
    float *conv_7_weights = calloc(conv_num_7,sizeof(float));
    fread(conv_7_weights, sizeof(float), conv_num_7, fp);

    int conv_num_8 = 1024*256*1*1;  
    float *conv_8_biases = calloc(256, sizeof(float));
    fread(conv_8_biases, sizeof(float), 256, fp);
    float *conv_8_scales = calloc(256,sizeof(float));
    fread(conv_8_scales, sizeof(float), 256, fp);
    float *conv_8_mean = calloc(256,sizeof(float));
    fread(conv_8_mean, sizeof(float), 256, fp);
    float *conv_8_variance = calloc(256,sizeof(float));
    fread(conv_8_variance, sizeof(float), 256, fp);
    float *conv_8_weights = calloc(conv_num_8,sizeof(float));
    fread(conv_8_weights, sizeof(float), conv_num_8, fp);

    int conv_num_9 = 256*512*3*3;
    float *conv_9_biases = calloc(512, sizeof(float));
    fread(conv_9_biases, sizeof(float), 512, fp);
    float *conv_9_scales = calloc(512,sizeof(float));
    fread(conv_9_scales, sizeof(float), 512, fp);
    float *conv_9_mean = calloc(512,sizeof(float));
    fread(conv_9_mean, sizeof(float), 512, fp);
    float *conv_9_variance = calloc(512,sizeof(float));
    fread(conv_9_variance, sizeof(float), 512, fp);
    float *conv_9_weights = calloc(conv_num_9,sizeof(float));
    fread(conv_9_weights, sizeof(float), conv_num_9, fp);

    int conv_num_10 = 512*255*1*1;
    float *conv_10_biases = calloc(255, sizeof(float));
    fread(conv_10_biases, sizeof(float), 255, fp);
    float *conv_10_scales = calloc(255,sizeof(float));
   // fread(conv_10_scales, sizeof(float), 255, fp);
    float *conv_10_mean = calloc(255,sizeof(float));
    //fread(conv_10_mean, sizeof(float), 255, fp);
    float *conv_10_variance = calloc(255,sizeof(float));
    //fread(conv_10_variance, sizeof(float), 255, fp);
    float *conv_10_weights = calloc(conv_num_10,sizeof(float));
    fread(conv_10_weights, sizeof(float), conv_num_10, fp);

    int conv_num_11 = 256*128*1*1;
    float *conv_11_biases = calloc(128, sizeof(float));
    fread(conv_11_biases, sizeof(float), 128, fp);
    float *conv_11_scales = calloc(128,sizeof(float));
    fread(conv_11_scales, sizeof(float), 128, fp);
    float *conv_11_mean = calloc(128,sizeof(float));
    fread(conv_11_mean, sizeof(float), 128, fp);
    float *conv_11_variance = calloc(128,sizeof(float));
    fread(conv_11_variance, sizeof(float), 128, fp);
    float *conv_11_weights = calloc(conv_num_11,sizeof(float));
    fread(conv_11_weights, sizeof(float), conv_num_11, fp);

    int conv_num_12 = 384*256*3*3;
    float *conv_12_biases = calloc(256, sizeof(float));
    fread(conv_12_biases, sizeof(float), 256, fp);
    float *conv_12_scales = calloc(256,sizeof(float));
    fread(conv_12_scales, sizeof(float), 256, fp);
    float *conv_12_mean = calloc(256,sizeof(float));
    fread(conv_12_mean, sizeof(float), 256, fp);
    float *conv_12_variance = calloc(256,sizeof(float));
    fread(conv_12_variance, sizeof(float), 256, fp);
    float *conv_12_weights = calloc(conv_num_12,sizeof(float));
    fread(conv_12_weights, sizeof(float), conv_num_12, fp);\

    int conv_num_13 = 256*255*1*1;
    float *conv_13_biases = calloc(255, sizeof(float));
    fread(conv_13_biases, sizeof(float), 255, fp);
    float *conv_13_scales = calloc(255,sizeof(float));
    //fread(conv_13_scales, sizeof(float), 255, fp);
    float *conv_13_mean = calloc(255,sizeof(float));
    //fread(conv_13_mean, sizeof(float), 255, fp);
    float *conv_13_variance = calloc(255,sizeof(float));
    //fread(conv_13_variance, sizeof(float), 255, fp);
    float *conv_13_weights = calloc(conv_num_13,sizeof(float));
    fread(conv_13_weights, sizeof(float), conv_num_13, fp);

    fclose(fp);



 	//create_network
    //输入416x416的图片，绝对够了
    size_t workspace_size_1 = (size_t)416*416*3*3*1024*sizeof(float);
    float *workspace0 = calloc(1,workspace_size_1);

    int *conv_1_out = calloc(3,sizeof(int));
    //actination:0,1,2
    float *conv_1_output = calloc(416*416*16,sizeof(float));

    conv_1_out = conv_bn_activation_layer(X,1,416,416,3,0,16,3,1,1,conv_1_mean,conv_1_variance,conv_1_scales,conv_1_weights,conv_1_biases,workspace0,conv_1_output);

    int *maxpool_1_out = calloc(3,sizeof(int));
    float *maxpool_1_output = calloc(208*208*16,sizeof(float));
    maxpool_1_out = maxpool_layer(conv_1_output,1,416,416,16,2,2,1,maxpool_1_output);

    int *conv_2_out = calloc(3,sizeof(int));
    float *conv_2_output = calloc(208*208*32,sizeof(float));

    conv_2_out = conv_bn_activation_layer(maxpool_1_output,1,208,208,16,0,32,3,1,1,conv_2_mean,conv_2_variance,conv_2_scales,conv_2_weights,conv_2_biases,workspace0,conv_2_output);
 
    int *maxpool_2_out = calloc(3,sizeof(int));
    float *maxpool_2_output = calloc(104*104*32,sizeof(float));
    maxpool_2_out = maxpool_layer(conv_2_output,1,208,208,32,2,2,1,maxpool_2_output);
  
    int *conv_3_out = calloc(3,sizeof(int));
    float *conv_3_output = calloc(104*104*64,sizeof(float));
    conv_3_out = conv_bn_activation_layer(maxpool_2_output,1,104,104,32,0,64,3,1,1,conv_3_mean,conv_3_variance,conv_3_scales,conv_3_weights,conv_3_biases,workspace0,conv_3_output);
 
    int *maxpool_3_out = calloc(3,sizeof(int));
    float *maxpool_3_output = calloc(52*52*64,sizeof(float));
    maxpool_3_out = maxpool_layer(conv_3_output,1,104,104,64,2,2,1,maxpool_3_output);
 
    int *conv_4_out = calloc(3,sizeof(int));
    float *conv_4_output = calloc(52*52*128,sizeof(float));
    conv_4_out = conv_bn_activation_layer(maxpool_3_output,1,52,52,64,0,128,3,1,1,conv_4_mean,conv_4_variance,conv_4_scales,conv_4_weights,conv_4_biases,workspace0,conv_4_output);
 
    int *maxpool_4_out = calloc(3,sizeof(int));
    float *maxpool_4_output = calloc(26*26*128,sizeof(float));
    maxpool_4_out = maxpool_layer(conv_4_output,1,52,52,128,2,2,1,maxpool_4_output);

    int *conv_5_out = calloc(3,sizeof(int));
    float *conv_5_output = calloc(26*26*256,sizeof(float));
    conv_5_out = conv_bn_activation_layer(maxpool_4_output,1,26,26,128,0,256,3,1,1,conv_5_mean,conv_5_variance,conv_5_scales,conv_5_weights,conv_5_biases,workspace0,conv_5_output);

    int *maxpool_5_out = calloc(3,sizeof(int));
    float *maxpool_5_output = calloc(13*13*256,sizeof(float));
    maxpool_5_out = maxpool_layer(conv_5_output,1,26,26,256,2,2,1,maxpool_5_output);
 
    int *conv_6_out = calloc(3,sizeof(int));
    float *conv_6_output = calloc(13*13*512,sizeof(float));
    conv_6_out = conv_bn_activation_layer(maxpool_5_output,1,13,13,256,0,512,3,1,1,conv_6_mean,conv_6_variance,conv_6_scales,conv_6_weights,conv_6_biases,workspace0,conv_6_output);

    int *maxpool_6_out = calloc(3,sizeof(int));
    float *maxpool_6_output = calloc(13*13*512,sizeof(float));
    maxpool_6_out = maxpool_layer(conv_6_output,1,13,13,512,2,1,1,maxpool_6_output);
 
    int *conv_7_out = calloc(3,sizeof(int));
    float *conv_7_output = calloc(13*13*1024,sizeof(float));
    conv_7_out = conv_bn_activation_layer(maxpool_6_output,1,13,13,512,0,1024,3,1,1,conv_7_mean,conv_7_variance,conv_7_scales,conv_7_weights,conv_7_biases,workspace0,conv_7_output);
   
    int *conv_8_out = calloc(3,sizeof(int));
    float *conv_8_output = calloc(13*13*256,sizeof(float));
    conv_8_out = conv_bn_activation_layer(conv_7_output,1,13,13,1024,0,256,1,1,0,conv_8_mean,conv_8_variance,conv_8_scales,conv_8_weights,conv_8_biases,workspace0,conv_8_output);
   
    int *conv_9_out = calloc(3,sizeof(int));
    float *conv_9_output = calloc(13*13*512,sizeof(float));
    conv_9_out = conv_bn_activation_layer(conv_8_output,1,13,13,256,0,512,3,1,1,conv_9_mean,conv_9_variance,conv_9_scales,conv_9_weights,conv_9_biases,workspace0,conv_9_output);

    int *conv_10_out = calloc(3,sizeof(int));
    float *conv_10_output = calloc(13*13*255,sizeof(float));
    conv_10_out = conv_activation_layer(conv_9_output,1,13,13,512,2,255,1,1,0,conv_10_mean,conv_10_variance,conv_10_scales,conv_10_weights,conv_10_biases,workspace0,conv_10_output);
   
    int *yolo_1_out = calloc(3,sizeof(int));
    float *yolo_1_output = calloc(13*13*3*(80+4+1),sizeof(float));
    
    yolo_1_out = yolo_layer_1(conv_10_output,yolo_1_output,1,13,13,3,80);
 
    //这里的route实际上啥都没干。。
  
    int *conv_11_out = calloc(3,sizeof(int));
    float *conv_11_output = calloc(13*13*128,sizeof(float));
    
   conv_11_out = conv_bn_activation_layer(conv_8_output,1,13,13,256,0,128,1,1,0,conv_11_mean,conv_11_variance,conv_11_scales,conv_11_weights,conv_11_biases,workspace0,conv_11_output);

   int *upsample_1_out = calloc(3,sizeof(int));
   float *upsample_1_output = calloc(26*26*128,sizeof(float));
   upsample_1_out = upsample_layer(conv_11_output,upsample_1_output,1,2,13,13,128);
 

    int *route_2_out = calloc(3,sizeof(int));
    float *route_2_output = calloc(26*26*384,sizeof(float));
    route_2_out = route_layer_2(upsample_1_output,26,26,128,conv_5_output,26,26,256,1,route_2_output);
    
    int *conv_12_out = calloc(3,sizeof(int));
    float *conv_12_output = calloc(26*26*256,sizeof(float));
    conv_12_out = conv_bn_activation_layer(route_2_output,1,26,26,384,0,256,3,1,1,conv_12_mean,conv_12_variance,conv_12_scales,conv_12_weights,conv_12_biases,workspace0,conv_12_output);
  
    int *conv_13_out = calloc(3,sizeof(int));
    float *conv_13_output = calloc(26*26*255,sizeof(float));
    conv_13_out = conv_activation_layer(conv_12_output,1,26,26,256,2,255,1,1,0,conv_13_mean,conv_13_variance,conv_13_scales,conv_13_weights,conv_13_biases,workspace0,conv_13_output);
 
    int *yolo_2_out = calloc(3,sizeof(int));
    float *yolo_2_output = calloc(26*26*3*(80+4+1),sizeof(float));
    yolo_2_out = yolo_layer_2(conv_13_output,yolo_2_output,1,26,26,3,80);
    //printf("yolo_2_out %.3f\n",yolo_2_output[26*26*255-1]);
    //printf("yolo_2_out_100 %.3f\n",yolo_2_output[26*26*255-100]);
    //printf("yolo_2_out index %d\n",26*26*255);


    //net->layers[16].output = yolo_1_output;
    //net->layers[23].output = yolo_2_output;



    //post processor

    char *coco_names[] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",   "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    
    int nboxes = 0;
       
	detection *dets = get_network_boxes_1(yolo_1_output,yolo_2_output, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    printf("nboxes %d\n", nboxes);
      
    if (nms) do_nms_sort_1(dets, nboxes, 80, nms);
        
	draw_detections_1(im, dets, nboxes, thresh, coco_names,80);

    //free memory

    free_detections_1(dets, nboxes);

    save_image_1(im, "predictions");

    free_image_1(im);
    free_image_1(sized);

    
    free(conv_1_mean);
    free(conv_1_variance);
    free(conv_1_scales);
    free(conv_1_weights);
    free(conv_1_biases);
    free(conv_1_out);
    free(conv_1_output);

    free(conv_2_mean);
    free(conv_2_variance);
    free(conv_2_scales);
    free(conv_2_weights);
    free(conv_2_biases);
    free(conv_2_out);
    free(conv_2_output);

    free(conv_3_mean);
    free(conv_3_variance);
    free(conv_3_scales);
    free(conv_3_weights);
    free(conv_3_biases);
    free(conv_3_out);
    free(conv_3_output);

    free(conv_4_mean);
    free(conv_4_variance);
    free(conv_4_scales);
    free(conv_4_weights);
    free(conv_4_biases);
    free(conv_4_out);
    free(conv_4_output);

    free(conv_5_mean);
    free(conv_5_variance);
    free(conv_5_scales);
    free(conv_5_weights);
    free(conv_5_biases);
    free(conv_5_out);
    free(conv_5_output);

    free(conv_6_mean);
    free(conv_6_variance);
    free(conv_6_scales);
    free(conv_6_weights);
    free(conv_6_biases);
    free(conv_6_out);
    free(conv_6_output);

    free(conv_7_mean);
    free(conv_7_variance);
    free(conv_7_scales);
    free(conv_7_weights);
    free(conv_7_biases);
    free(conv_7_out);
    free(conv_7_output);

    free(conv_8_mean);
    free(conv_8_variance);
    free(conv_8_scales);
    free(conv_8_weights);
    free(conv_8_biases);
    free(conv_8_out);
    free(conv_8_output);

    free(conv_9_mean);
    free(conv_9_variance);
    free(conv_9_scales);
    free(conv_9_weights);
    free(conv_9_biases);
    free(conv_9_out);
    free(conv_9_output);

    free(conv_10_mean);
    free(conv_10_variance);
    free(conv_10_scales);
    free(conv_10_weights);
    free(conv_10_biases);
    free(conv_10_out);
    free(conv_10_output);

    free(conv_11_mean);
    free(conv_11_variance);
    free(conv_11_scales);
    free(conv_11_weights);
    free(conv_11_biases);
    free(conv_11_out);
    free(conv_11_output);

    free(conv_12_mean);
    free(conv_12_variance);
    free(conv_12_scales);
    free(conv_12_weights);
    free(conv_12_biases);
    free(conv_12_out);
    free(conv_12_output);

    free(conv_13_mean);
    free(conv_13_variance);
    free(conv_13_scales);
    free(conv_13_weights);
    free(conv_13_biases);
    free(conv_13_out);
    free(conv_13_output);

    free(workspace0);

    free(maxpool_1_out);
    free(maxpool_1_output);

    free(maxpool_2_out);
    free(maxpool_2_output);

    free(maxpool_3_out);
    free(maxpool_3_output);

    free(maxpool_4_out);
    free(maxpool_4_output);

    free(maxpool_5_out);
    free(maxpool_5_output);

    free(maxpool_6_out);
    free(maxpool_6_output);

    free(upsample_1_out);
    free(upsample_1_output);

    //free(route_1_out);
    //free(route_1_output);

    free(route_2_out);
    free(route_2_output);

    free(yolo_1_out);
    free(yolo_1_output);

    free(yolo_2_out);
    free(yolo_2_output);


}



int main(int argc, char **argv)
{
//check input argv..
    if (0 == strcmp(argv[1], "-infer")){

	float thresh = 0.15;
	 
        test_detector(argv[2],thresh, .5);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}

