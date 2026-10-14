/* SPDX-License-Identifier: GPL-3.0-only */
/* Own CUDA kernels: batched tensors [B,T,C], tiled im2col, one LSTM block/record. */
#include "internal.h"
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
struct DeviceModel {
    pnsn_layer layer[25];
    size_t window,max_width,capacity,patch_rows;
    int bits;
    float *a,*b,*c,*patch,*gates,*scales;
    int8_t *q;
    void *raw;
    int *invalid;
    cudaStream_t stream;
    cudaEvent_t before,normalized,after;
    double prep_ms,network_ms;
};
static int check(cudaError_t result) {
    if(result==cudaSuccess)return 0;
    char message[256];snprintf(message,sizeof(message),"CUDA: %s",cudaGetErrorString(result));return pnsn_error(message);
}
#define TRY(x) do {if(check(x))goto fail;}while(0)
__global__ void im2col(const float *x,float *patch,int8_t *q,float *scale,int rows,int begin,int t,int ni,int k,int pad,int stride,int width,int bits) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=rows)return;
    int global=begin+local,per_record=stride==2?t*2:t,row=global%per_record,batch=global/per_record;
    float maximum=0;
    for(int j=0;j<width;j++) {
        int kernel=j/ni,c=j%ni,source=row+kernel-pad;bool valid=kernel<k;
        if(stride==2) {source=row+pad-kernel;valid=valid && !(source%2);source/=2;}
        float value=valid && source>=0 && source<t?x[(batch*t+source)*ni+c]:0;
        patch[local*width+j]=value;maximum=fmaxf(maximum,fabsf(value));
    }
    if(bits==8) {
        float s=fmaxf(maximum/127,1e-30f);scale[local]=s;
        for(int j=0;j<width;j++)q[local*width+j]=(int8_t)fmaxf(-127,fminf(127,nearbyintf(patch[local*width+j]/s)));
    }
}
__global__ void matrix(const float *patch,const int8_t *q,const float *xs,pnsn_layer l,float *out,int rows,int bits,int relu) {
    int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=rows*(int)l.no)return;
    int row=at/l.no,o=at%l.no;float total=l.bias[o];
    if(bits==8) {
        int sum=0;const int *x=(const int*)(q+row*l.width),*w=(const int*)(l.q+o*l.width);
        for(int j=0;j<(int)l.width/4;j++) {
#if __CUDA_ARCH__ >= 610
            sum=__dp4a(x[j],w[j],sum);
#else
            const int8_t *a=(const int8_t*)(x+j),*b=(const int8_t*)(w+j);
            for(int k=0;k<4;k++)sum+=(int)a[k]*b[k];
#endif
        }
        total+=(float)sum*xs[row]*l.scale[o];
    } else for(int j=0;j<(int)l.width;j++)total+=patch[row*l.width+j]*l.w[o*l.width+j];
    out[at]=relu?fmaxf(total,0):total;
}
__global__ void pool(const float *x,float *y,int rows,int channels) {
    int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=rows*channels)return;
    int row=at/channels,c=at%channels;y[at]=fmaxf(x[row*2*channels+c],x[(row*2+1)*channels+c]);
}
__device__ float sigmoid_gpu(float x){return 1/(1+expf(-x));}
__global__ void recurrence(const float *input,float *output,pnsn_layer hh,int t,int reverse,int bits) {
    __shared__ float hidden[96],cell[96],gate[384],scale;
    __shared__ __align__(4) int8_t qhidden[96];
    int j=threadIdx.x,batch=blockIdx.x;if(j<96){hidden[j]=0;cell[j]=0;}__syncthreads();
    for(int step=0;step<t;step++) {
        int at=reverse?t-1-step:step;
        if(bits==8) {
            if(j==0){float maxv=0;for(int k=0;k<96;k++)maxv=fmaxf(maxv,fabsf(hidden[k]));scale=fmaxf(maxv/127,1e-30f);}
            __syncthreads();if(j<96)qhidden[j]=(int8_t)fmaxf(-127,fminf(127,nearbyintf(hidden[j]/scale)));__syncthreads();
        }
        float total=hh.bias[j];
        if(bits==8) {
            int sum=0;const int *x=(const int*)qhidden,*w=(const int*)(hh.q+j*96);
            for(int k=0;k<24;k++) {
#if __CUDA_ARCH__ >= 610
                sum=__dp4a(x[k],w[k],sum);
#else
                for(int m=0;m<4;m++)sum+=(int)qhidden[k*4+m]*hh.q[j*96+k*4+m];
#endif
            }
            total+=(float)sum*scale*hh.scale[j];
        } else for(int k=0;k<96;k++)total+=hidden[k]*hh.w[j*96+k];
        gate[j]=total+input[(batch*t+at)*384+j];__syncthreads();
        if(j<96) {
            cell[j]=sigmoid_gpu(gate[96+j])*cell[j]+sigmoid_gpu(gate[j])*tanhf(gate[192+j]);
            hidden[j]=sigmoid_gpu(gate[288+j])*tanhf(cell[j]);output[(batch*t+at)*192+reverse*96+j]=hidden[j];
        }__syncthreads();
    }
}
__global__ void softmax_gpu(float *x,int rows) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    float maximum=x[row*5];for(int c=1;c<5;c++)maximum=fmaxf(maximum,x[row*5+c]);
    float sum=0;for(int c=0;c<5;c++){x[row*5+c]=expf(x[row*5+c]-maximum);sum+=x[row*5+c];}
    for(int c=0;c<5;c++)x[row*5+c]/=sum;
}
template<typename T>
__global__ void preprocess(const T *x,float *y,int w,int normalize,int diff,int *invalid) {
    __shared__ double reduction[128],mean,scale;
    int tid=threadIdx.x,b=blockIdx.x/3,c=blockIdx.x%3;
    const T *record=x+(size_t)b*3*w+c*w;double origin=(double)record[0],sum=0;
    for(int t=tid;t<w;t+=128) {
        double raw=(double)record[t],v=diff?(double)record[t+1<w?t+1:t]-raw:raw-origin;
        if(!isfinite(raw) || !isfinite(v))atomicExch(invalid,1);
        if(!normalize)y[((size_t)b*w+t)*3+c]=(float)raw;else sum+=v;
    }
    if(!normalize)return;
    reduction[tid]=sum;__syncthreads();
    for(int s=64;s;s/=2){if(tid<s)reduction[tid]+=reduction[tid+s];__syncthreads();}
    if(tid==0)mean=reduction[0]/w;__syncthreads();sum=0;
    for(int t=tid;t<w;t+=128) {
        double raw=(double)record[t],v=(diff?(double)record[t+1<w?t+1:t]-raw:raw-origin)-mean;sum+=v*v;
    }
    reduction[tid]=sum;__syncthreads();
    for(int s=64;s;s/=2){if(tid<s)reduction[tid]+=reduction[tid+s];__syncthreads();}
    if(tid==0){scale=sqrt(reduction[0]/(w-1))+1e-6;if(!isfinite(scale) || !isfinite(mean))atomicExch(invalid,1);}__syncthreads();
    for(int t=tid;t<w;t+=128) {
        double raw=(double)record[t],v=(diff?(double)record[t+1<w?t+1:t]-raw:raw-origin)-mean;
        y[((size_t)b*w+t)*3+c]=(float)(v/scale);
    }
}
static void convolution(DeviceModel *d,int index,const float *x,float *y,int t,int relu,int batch) {
    pnsn_layer l=d->layer[index];int total=(l.kind==2?t*2:t)*batch;
    for(int begin=0;begin<total;begin+=(int)d->patch_rows) {
        int rows=total-begin<(int)d->patch_rows?total-begin:(int)d->patch_rows;
        im2col<<<(rows+127)/128,128,0,d->stream>>>(x,d->patch,d->q,d->scales,rows,begin,t,l.ni,l.kernel,l.pad,l.kind==2?2:1,l.width,d->bits);
        matrix<<<(rows*l.no+127)/128,128,0,d->stream>>>(d->patch,d->q,d->scales,l,y+(size_t)begin*l.no,rows,d->bits,relu);
    }
}
static float *network(DeviceModel *d,int batch) {
    int t=(int)d->window;float *x=d->a,*y=d->b,*swap;
    for(int i=0;i<7;i++) {
        convolution(d,i,x,y,t,1,batch);
        int count=t/2*d->layer[i].no*batch;
        pool<<<(count+127)/128,128,0,d->stream>>>(y,x,t/2*batch,d->layer[i].no);t/=2;
    }
    for(int base=7;base<=12;base+=5) {
        for(int dir=0;dir<2;dir++) {
            convolution(d,base+dir*2,x,d->gates,t,0,batch);
            recurrence<<<batch,384,0,d->stream>>>(d->gates,d->c,d->layer[base+dir*2+1],t,dir,d->bits);
        }convolution(d,base+4,d->c,y,t,0,batch);swap=x;x=y;y=swap;
    }
    for(int i=17;i<25;i++){convolution(d,i,x,y,t,i!=24,batch);if(i!=24)t*=2;swap=x;x=y;y=swap;}
    softmax_gpu<<<(t*batch+127)/128,128,0,d->stream>>>(x,t*batch);return x;
}
static void free_buffers(DeviceModel *d) {
    cudaFree(d->a);cudaFree(d->b);cudaFree(d->c);cudaFree(d->patch);cudaFree(d->q);cudaFree(d->gates);cudaFree(d->scales);cudaFree(d->raw);cudaFree(d->invalid);
}
static int reserve(DeviceModel *d,size_t batch) {
    if(batch<=d->capacity)return 0;
    if(!batch || batch>INT_MAX/(d->window*8))return pnsn_error("CUDA batch tensor exceeds supported indexing range");
    DeviceModel n={};n.patch_rows=d->window*(batch<32?batch:32);n.capacity=batch;
    if(n.patch_rows>INT_MAX/d->max_width)n.patch_rows=INT_MAX/d->max_width;
    TRY(cudaMalloc(&n.a,d->window*8*batch*sizeof(float)));TRY(cudaMalloc(&n.b,d->window*8*batch*sizeof(float)));TRY(cudaMalloc(&n.c,d->window*8*batch*sizeof(float)));
    TRY(cudaMalloc(&n.gates,d->window/128*384*batch*sizeof(float)));
    TRY(cudaMalloc(&n.patch,n.patch_rows*d->max_width*sizeof(float)));TRY(cudaMalloc(&n.q,n.patch_rows*d->max_width));TRY(cudaMalloc(&n.scales,n.patch_rows*sizeof(float)));
    TRY(cudaMalloc(&n.raw,d->window*3*batch*sizeof(double)));TRY(cudaMalloc(&n.invalid,sizeof(int)));
    free_buffers(d);d->a=n.a;d->b=n.b;d->c=n.c;d->gates=n.gates;d->patch=n.patch;d->q=n.q;d->scales=n.scales;d->raw=n.raw;d->invalid=n.invalid;d->patch_rows=n.patch_rows;d->capacity=batch;return 0;
fail:free_buffers(&n);return -1;
}
void pnsn_cuda_close(void *context) {
    DeviceModel *d=(DeviceModel*)context;if(!d)return;
    if(d->stream)cudaStreamSynchronize(d->stream);
    for(int i=0;i<25;i++){cudaFree(d->layer[i].bias);cudaFree(d->layer[i].scale);cudaFree(d->layer[i].q);cudaFree(d->layer[i].w);}free_buffers(d);
    if(d->before)cudaEventDestroy(d->before);if(d->normalized)cudaEventDestroy(d->normalized);if(d->after)cudaEventDestroy(d->after);
    if(d->stream)cudaStreamDestroy(d->stream);free(d);
}
void *pnsn_cuda_open(pnsn_context *ctx) {
    if(ctx->model){pnsn_error("China UNet is supported by CPU/cuDNN, not the custom PNSN CUDA backend");return NULL;}
    DeviceModel *d=(DeviceModel*)calloc(1,sizeof(*d));if(!d){pnsn_error("CUDA host out of memory");return NULL;}
    d->window=ctx->options.window;d->bits=ctx->bits;d->max_width=ctx->max_width;
    TRY(cudaStreamCreateWithFlags(&d->stream,cudaStreamNonBlocking));TRY(cudaEventCreate(&d->before));TRY(cudaEventCreate(&d->normalized));TRY(cudaEventCreate(&d->after));
    for(int i=0;i<25;i++) {
        pnsn_layer h=ctx->layers[i],*l=&d->layer[i];*l=h;l->bias=NULL;l->scale=NULL;l->q=NULL;l->w=NULL;
        TRY(cudaMalloc(&l->bias,h.no*sizeof(float)));TRY(cudaMalloc(&l->scale,h.no*sizeof(float)));
        TRY(cudaMemcpy(l->bias,h.bias,h.no*sizeof(float),cudaMemcpyHostToDevice));TRY(cudaMemcpy(l->scale,h.scale,h.no*sizeof(float),cudaMemcpyHostToDevice));
        if(d->bits==8){TRY(cudaMalloc(&l->q,h.no*h.width));TRY(cudaMemcpy(l->q,h.q,h.no*h.width,cudaMemcpyHostToDevice));}
        else {TRY(cudaMalloc(&l->w,h.no*h.width*sizeof(float)));TRY(cudaMemcpy(l->w,h.w,h.no*h.width*sizeof(float),cudaMemcpyHostToDevice));}
    }
    if(reserve(d,1))goto fail;return d;
fail:pnsn_cuda_close(d);return NULL;
}
int pnsn_cuda_forward(void *context,const float *input,float *out) {
    DeviceModel *d=(DeviceModel*)context;size_t size=d->window;
    if(check(cudaMemcpyAsync(d->a,input,size*3*sizeof(float),cudaMemcpyHostToDevice,d->stream)))return -1;
    float *x=network(d,1);if(check(cudaGetLastError()))return -1;
    if(check(cudaMemcpyAsync(out,x,size*5*sizeof(float),cudaMemcpyDeviceToHost,d->stream)))return -1;
    return check(cudaStreamSynchronize(d->stream));
}
int pnsn_cuda_forward_batch(void *context,const void *input,pnsn_dtype dtype,size_t batch,int normalize,int diff,float *out) {
    DeviceModel *d=(DeviceModel*)context;if(reserve(d,batch))return -1;
    size_t bytes=batch*d->window*3*(dtype==PNSN_F64?8:4);int invalid=0;float prep,net;
    if(check(cudaMemcpyAsync(d->raw,input,bytes,cudaMemcpyHostToDevice,d->stream)) || check(cudaMemsetAsync(d->invalid,0,sizeof(int),d->stream)) || check(cudaEventRecord(d->before,d->stream)))return -1;
    if(dtype==PNSN_F64)preprocess<<<batch*3,128,0,d->stream>>>((const double*)d->raw,d->a,d->window,normalize,diff,d->invalid);
    else if(dtype==PNSN_I32)preprocess<<<batch*3,128,0,d->stream>>>((const int32_t*)d->raw,d->a,d->window,normalize,diff,d->invalid);
    else preprocess<<<batch*3,128,0,d->stream>>>((const float*)d->raw,d->a,d->window,normalize,diff,d->invalid);
    if(check(cudaEventRecord(d->normalized,d->stream)))return -1;
    float *x=network(d,(int)batch);if(check(cudaEventRecord(d->after,d->stream)) || check(cudaGetLastError()))return -1;
    if(check(cudaMemcpyAsync(out,x,batch*d->window*5*sizeof(float),cudaMemcpyDeviceToHost,d->stream)) || check(cudaMemcpyAsync(&invalid,d->invalid,sizeof(int),cudaMemcpyDeviceToHost,d->stream)) || check(cudaStreamSynchronize(d->stream)))return -1;
    if(invalid)return pnsn_error("Non-finite batch waveform or overflowing float64 normalization");
    if(check(cudaEventElapsedTime(&prep,d->before,d->normalized)) || check(cudaEventElapsedTime(&net,d->normalized,d->after)))return -1;
    d->prep_ms=prep;d->network_ms=net;return 0;
}
int pnsn_cuda_batch_stats(void *context,double *prep,double *net,size_t *bytes) {
    DeviceModel *d=(DeviceModel*)context;*prep=d->prep_ms;*net=d->network_ms;
    *bytes=d->window*d->capacity*(24*4+3*4+3*8)+d->patch_rows*(d->max_width*5+4)+sizeof(int);
    for(int i=0;i<25;i++)*bytes+=d->layer[i].no*(8+d->layer[i].width*(d->bits==8?1:4));return 0;
}
