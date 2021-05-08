#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <device_launch_parameters.h>
//#include "cuda_head.h"
//#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
//#include <assert.h> 

void fill_gpu(int N, float ALPHA, float * X, int INCX);
void activate_layer_1_gpu(float *x, int n, int y) ;
void maxpool_layer_gpu(float *layer_input,int batch, int input_h, int input_w, int input_c, int size, int stride, int padding, float *layer_output,int *indexes_gpu);
void im2col_gpu_1(float* data_im,int channels,  int height,  int width,int ksize,  int stride, int pad, float* data_col);
void normalize_1_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void scale_bias_1_gpu(float *output, float *biases, int batch, int n, int size);
void add_bias_1_gpu(float *output, float *biases, int batch, int n, int size);
void copy_gpu_offset(int N, float *X,int OFFX,int INCX,float *Y,int OFFY,int INCY);
void upsample_gpu_1(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

static void error(const char *s){
perror(s);
assert(0);
exit(-1);
}

static void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    //__host__ ​ __device__ ​cudaError_t cudaGetLastError ( void )
    //返回运行时调用中的最后一个错误
    cudaError_t status2 = cudaGetLastError();
    //如果失败的话。。。。
    if (status != cudaSuccess)
    {   
        //__host__ ​ __device__ ​const char* cudaGetErrorString ( cudaError_t error )
        //返回错误码的描述字符串，也就是上面的cudaError_t中返回的实际上是相应的错误代码或者说是id..
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

static dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / 512 + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*512) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

