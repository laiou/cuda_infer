#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>
extern "C" {
#include "cuda_head.h"
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
  
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  
    if(i < N) X[i*INCX] = ALPHA;
}


void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    
    fill_kernel<<<cuda_gridsize(N), 512>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}



__device__ float linear_activate_kernel(float x){return x;}
__device__ float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
__device__ float relu_activate_kernel(float x){return x*(x>0);}
__device__ float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}

__device__ float activate_kernel(float x, int y)

{

switch(y){
        case 0:
            return leaky_activate_kernel(x);
        case 1:
	    return relu_activate_kernel(x);
	case 2:
	    return linear_activate_kernel(x);
	case 3:
	    return logistic_activate_kernel(x);
}
return 0;

}


__global__ void activate_array_kernel(float *x, int n, int y)
{
   
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  
    if(i < n) x[i] = activate_kernel(x[i], y);
}

void activate_layer_1_gpu(float *x, int n, int y) 
{   
  
    activate_array_kernel<<<cuda_gridsize(n), 512>>>(x, n, y);
    check_error(cudaPeekAtLastError());
}


__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output, int *indexes)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    __syncthreads();
    output[out_index] = max;
    indexes[out_index] = max_i;
}

 void maxpool_layer_gpu(float *layer_input,int batch, int input_h, int input_w, int input_c, int size, int stride, int padding, float *layer_output,int *indexes_gpu){
    int b,i,j,k,m;
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
    
    size_t n = output_h*output_w*output_c*batch;
    forward_maxpool_layer_kernel<<<cuda_gridsize(n), 512>>>(n, input_h, input_w, input_c, stride, size, padding, layer_input, layer_output,indexes_gpu);
    check_error(cudaPeekAtLastError());


}

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
       
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    
    for(; index < n; index += blockDim.x*gridDim.x){
   
        int w_out = index % width_col;
        int h_index = index / width_col;
       
        int h_out = h_index % height_col;
      
        int channel_in = h_index / height_col;
     
        int channel_out = channel_in * ksize * ksize;
       
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
     
        float* data_col_ptr = data_col;
     
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
      
        const float* data_im_ptr = data_im;
      
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
     
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
          
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
              
                data_col_ptr += height_col * width_col;
            }
        }
    }
}


 void im2col_gpu_1(float* im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;

    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    
    im2col_gpu_kernel<<<(num_kernels+512-1)/512,512>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}


__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
   
    int f = (index/spatial)%filters;

    x[index] = (x[index] - mean[f])/(sqrtf(variance[f] + .00001f));
}

__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

void normalize_1_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
   
    size_t N = batch*filters*spatial;
   
    normalize_kernel<<<cuda_gridsize(N), 512>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

 void scale_bias_1_gpu(float *output, float *biases, int batch, int n, int size)
{
  
    dim3 dimGrid((size-1)/512 + 1, n, batch);
    dim3 dimBlock(512, 1, 1);


    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}


void add_bias_1_gpu(float *output, float *biases, int batch, int n, int size)
{
   
    int num = n*size*batch;
   
    add_bias_kernel<<< cuda_gridsize(num), 512>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}


__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
  
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
   
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

void copy_gpu_offset(int N, float *X,int OFFX,int INCX,float *Y,int OFFY,int INCY)
{
    
    copy_kernel<<<cuda_gridsize(N), 512>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

 __global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if(forward) out[out_index] += scale * x[in_index];
    }

 void upsample_gpu_1(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel<<<cuda_gridsize(size), 512>>>(size, in, w, h, c, batch, stride, forward, scale, out);
    check_error(cudaPeekAtLastError());
}
