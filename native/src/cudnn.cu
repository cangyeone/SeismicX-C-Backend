/* SPDX-License-Identifier: GPL-3.0-only */
/* Vendor cuDNN convolutions and bidirectional LSTM behind the public C API.
 * FP32 features are NCT; INT8-convolution features are NTC (NHWC) to avoid transposes.
 * Only preprocessing/layout/activation use small CUDA kernels.
 * Optional PNSN_CUDNN_INT8: true W8A8 convolutions, INT32 accumulation; FP32 LSTM.
 * Transposed convolutions use two polyphase INT8 forward convolutions. */
#include "internal.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include <algorithm>
#include <new>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
struct TensorPair {
    cudnnTensorDescriptor_t m=nullptr,b=nullptr;
    ~TensorPair(){if(m)cudnnDestroyTensorDescriptor(m);if(b)cudnnDestroyTensorDescriptor(b);}
};
struct IntConv {
    cudnnTensorDescriptor_t x=nullptr,y=nullptr,last_x=nullptr,last_y=nullptr;
    cudnnFilterDescriptor_t filter=nullptr;
    cudnnConvolutionDescriptor_t op=nullptr;
    cudnnConvolutionFwdAlgo_t algo;
    int8_t *weight=nullptr;
    int k=0,pad=0,input_t=0,first=0,rows=0,last_rows=0;
    size_t workspace=0,bytes=0;
    ~IntConv(){if(x)cudnnDestroyTensorDescriptor(x);if(y)cudnnDestroyTensorDescriptor(y);if(last_x)cudnnDestroyTensorDescriptor(last_x);if(last_y)cudnnDestroyTensorDescriptor(last_y);if(filter)cudnnDestroyFilterDescriptor(filter);if(op)cudnnDestroyConvolutionDescriptor(op);cudaFree(weight);}
};
struct Conv {
    cudnnTensorDescriptor_t x=nullptr,y=nullptr;
    cudnnFilterDescriptor_t f=nullptr;
    cudnnConvolutionDescriptor_t op=nullptr;
    cudnnConvolutionFwdAlgo_t forward;
    cudnnConvolutionBwdDataAlgo_t backward;
    float *weight=nullptr,*bias=nullptr,*scale=nullptr;
    IntConv integer[2];
    int nic=0;
    int ni=0,no=0,k=0,pad=0,transpose=0,upsample=0,stride=1,t=0,u=0,source_t=0;
    size_t workspace=0,bytes=0;
};
struct RNN {
    cudnnRNNDescriptor_t op=nullptr;
    cudnnRNNDataDescriptor_t x=nullptr,y=nullptr;
    cudnnTensorDescriptor_t hidden=nullptr;
    float *weight=nullptr;
    size_t bytes=0,workspace=0;
    int ni=0;
};
struct DNN {
    cudaStream_t stream=nullptr;
    cudaEvent_t before=nullptr,normalized=nullptr,after=nullptr;
    cudnnHandle_t handle=nullptr;
    cudnnDropoutDescriptor_t dropout=nullptr;
    cudnnPoolingDescriptor_t pool=nullptr;
    cudnnTensorDescriptor_t px[7]={},py[7]={};
    Conv conv[25]; RNN rnn[2];
    float *a=nullptr,*b=nullptr,*seq=nullptr,*rout=nullptr,*output=nullptr;
    void *raw=nullptr,*work=nullptr;
    int *invalid=nullptr,*lengths=nullptr;
    size_t window=0,capacity=0,workbytes=0;
    int batch=0,int8=0,model=0,classes=5;
    float *skips=nullptr,*up=nullptr;
    int8_t *quant=nullptr;float *iscale=nullptr,*accum=nullptr;
    size_t quant_bytes=0,accum_bytes=0,scale_bytes=0;
    double prep_ms=0,network_ms=0;
    ~DNN() {
        if(stream)cudaStreamSynchronize(stream);
        for(auto &c:conv){if(c.x)cudnnDestroyTensorDescriptor(c.x);if(c.y)cudnnDestroyTensorDescriptor(c.y);if(c.f)cudnnDestroyFilterDescriptor(c.f);if(c.op)cudnnDestroyConvolutionDescriptor(c.op);cudaFree(c.weight);cudaFree(c.bias);cudaFree(c.scale);}
        for(auto &r:rnn){if(r.op)cudnnDestroyRNNDescriptor(r.op);if(r.x)cudnnDestroyRNNDataDescriptor(r.x);if(r.y)cudnnDestroyRNNDataDescriptor(r.y);if(r.hidden)cudnnDestroyTensorDescriptor(r.hidden);cudaFree(r.weight);}
        for(int i=0;i<7;i++){if(px[i])cudnnDestroyTensorDescriptor(px[i]);if(py[i])cudnnDestroyTensorDescriptor(py[i]);}
        if(pool)cudnnDestroyPoolingDescriptor(pool);if(dropout)cudnnDestroyDropoutDescriptor(dropout);if(handle)cudnnDestroy(handle);
        cudaFree(a);cudaFree(b);cudaFree(seq);cudaFree(rout);cudaFree(output);cudaFree(raw);cudaFree(work);cudaFree(invalid);cudaFree(lengths);cudaFree(quant);cudaFree(iscale);cudaFree(accum);cudaFree(skips);cudaFree(up);
        if(before)cudaEventDestroy(before);if(normalized)cudaEventDestroy(normalized);if(after)cudaEventDestroy(after);if(stream)cudaStreamDestroy(stream);
    }
};
static int check(cudaError_t s,int line){if(s==cudaSuccess)return 0;char msg[256];snprintf(msg,sizeof(msg),"cuDNN backend CUDA at line %d: %s",line,cudaGetErrorString(s));return pnsn_error(msg);}
static int check(cudnnStatus_t s,int line){if(s==CUDNN_STATUS_SUCCESS)return 0;char msg[256];snprintf(msg,sizeof(msg),"cuDNN at line %d: %s",line,cudnnGetErrorString(s));return pnsn_error(msg);}
#define OK(x) do{if(check((x),__LINE__))return -1;}while(0)
static int tensor(cudnnTensorDescriptor_t d,int b,int c,int t,int interleaved=0){OK(cudnnSetTensor4dDescriptor(d,interleaved?CUDNN_TENSOR_NHWC:CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b,c,1,t));return 0;}
static float value(const pnsn_context *ctx,const pnsn_layer &l,int o,int i){return ctx->bits==8?l.q[o*l.width+i]*l.scale[o]:l.w[o*l.width+i];}
static int init(DNN *d,pnsn_context *ctx){
    d->window=ctx->options.window;d->model=ctx->model;d->classes=ctx->classes;
#ifdef PNSN_CUDNN_INT8
    d->int8=ctx->bits==8;
#endif
    OK(cudaStreamCreateWithFlags(&d->stream,cudaStreamNonBlocking));OK(cudaEventCreate(&d->before));OK(cudaEventCreate(&d->normalized));OK(cudaEventCreate(&d->after));
    OK(cudnnCreate(&d->handle));OK(cudnnSetStream(d->handle,d->stream));OK(cudnnCreatePoolingDescriptor(&d->pool));
    OK(cudnnSetPooling2dDescriptor(d->pool,CUDNN_POOLING_MAX,CUDNN_NOT_PROPAGATE_NAN,1,2,0,0,1,2));
    for(int i=0;i<7;i++){OK(cudnnCreateTensorDescriptor(&d->px[i]));OK(cudnnCreateTensorDescriptor(&d->py[i]));}
    for(int i=0;i<25;i++){
        const pnsn_layer &l=ctx->layers[i];if(l.kind!=1 && l.kind!=2 && l.kind!=3)continue;
        Conv &c=d->conv[i];c.ni=l.ni;c.no=l.no;c.k=l.kernel;c.pad=l.pad;c.transpose=l.kind==2;c.upsample=l.kind==3;c.stride=c.upsample?1:l.stride;
        OK(cudnnCreateTensorDescriptor(&c.x));OK(cudnnCreateTensorDescriptor(&c.y));OK(cudnnCreateFilterDescriptor(&c.f));OK(cudnnCreateConvolutionDescriptor(&c.op));
        OK(cudnnSetFilter4dDescriptor(c.f,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,c.transpose?c.ni:c.no,c.transpose?c.no:c.ni,1,c.k));
        OK(cudnnSetConvolution2dDescriptor(c.op,0,c.pad,1,c.stride,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));OK(cudnnSetConvolutionMathType(c.op,CUDNN_FMA_MATH));
        std::vector<float>w(c.ni*c.no*c.k);
        for(int o=0;o<c.no;o++)for(int k=0;k<c.k;k++)for(int in=0;in<c.ni;in++)w[c.transpose?(in*c.no+o)*c.k+k:(o*c.ni+in)*c.k+k]=value(ctx,l,o,k*c.ni+in);
        c.bytes=w.size()*4+c.no*4;OK(cudaMalloc(&c.weight,w.size()*4));OK(cudaMalloc(&c.bias,c.no*4));
        OK(cudaMemcpy(c.weight,w.data(),w.size()*4,cudaMemcpyHostToDevice));OK(cudaMemcpy(c.bias,l.bias,c.no*4,cudaMemcpyHostToDevice));
        if(d->int8){
            c.nic=d->model?l.width:(c.ni+3)/4*4;OK(cudaMalloc(&c.scale,c.no*4));OK(cudaMemcpy(c.scale,l.scale,c.no*4,cudaMemcpyHostToDevice));c.bytes+=c.no*4;
            for(int phase=0;phase<(c.transpose?2:1);phase++){
                IntConv &ic=c.integer[phase];ic.k=d->model?1:c.transpose?(c.k+1-phase)/2:c.k;
                ic.first=(phase+(c.pad&1))&1;int offset=(ic.first+c.pad-phase)/2;ic.pad=d->model?0:c.transpose?ic.k-1-offset:c.pad;
                OK(cudnnCreateTensorDescriptor(&ic.x));OK(cudnnCreateTensorDescriptor(&ic.y));OK(cudnnCreateFilterDescriptor(&ic.filter));OK(cudnnCreateConvolutionDescriptor(&ic.op));
                OK(cudnnSetFilter4dDescriptor(ic.filter,CUDNN_DATA_INT8,CUDNN_TENSOR_NHWC,c.no,c.nic,1,ic.k));
                OK(cudnnSetConvolution2dDescriptor(ic.op,0,ic.pad,1,(d->model || c.transpose)?1:c.stride,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_INT32));OK(cudnnSetConvolutionMathType(ic.op,CUDNN_DEFAULT_MATH));
                if(d->model){OK(cudnnCreateTensorDescriptor(&ic.last_x));OK(cudnnCreateTensorDescriptor(&ic.last_y));}
                std::vector<int8_t>q(c.no*ic.k*c.nic,0);
                if(d->model)memcpy(q.data(),l.q,q.size());
                else for(int o=0;o<c.no;o++)for(int k=0;k<ic.k;k++)for(int in=0;in<c.ni;in++){
                    int original=c.transpose?(ic.k-1-k)*2+phase:k;
                    q[(o*ic.k+k)*c.nic+in]=l.q[o*l.width+original*c.ni+in];
                }
                ic.bytes=q.size();OK(cudaMalloc(&ic.weight,ic.bytes));OK(cudaMemcpy(ic.weight,q.data(),ic.bytes,cudaMemcpyHostToDevice));c.bytes+=ic.bytes;
            }
        }
    }
    if(d->model)return 0;
    OK(cudnnCreateDropoutDescriptor(&d->dropout));OK(cudnnSetDropoutDescriptor(d->dropout,d->handle,0,nullptr,0,0));
    for(int q=0;q<2;q++){
        int base=7+q*5;RNN &r=d->rnn[q];r.ni=ctx->layers[base].ni;
        OK(cudnnCreateRNNDescriptor(&r.op));OK(cudnnCreateRNNDataDescriptor(&r.x));OK(cudnnCreateRNNDataDescriptor(&r.y));OK(cudnnCreateTensorDescriptor(&r.hidden));
        OK(cudnnSetRNNDescriptor_v8(r.op,CUDNN_RNN_ALGO_STANDARD,CUDNN_LSTM,CUDNN_RNN_DOUBLE_BIAS,CUDNN_BIDIRECTIONAL,CUDNN_LINEAR_INPUT,CUDNN_DATA_FLOAT,CUDNN_DATA_FLOAT,CUDNN_FMA_MATH,r.ni,96,96,1,d->dropout,CUDNN_RNN_PADDED_IO_DISABLED));
        OK(cudnnGetRNNWeightSpaceSize(d->handle,r.op,&r.bytes));OK(cudaMalloc(&r.weight,r.bytes));OK(cudaMemset(r.weight,0,r.bytes));
        TensorPair desc;OK(cudnnCreateTensorDescriptor(&desc.m));OK(cudnnCreateTensorDescriptor(&desc.b));
        for(int dir=0;dir<2;dir++)for(int gate=0;gate<8;gate++){
            void *mw,*bw;OK(cudnnGetRNNWeightParams(d->handle,r.op,dir,r.bytes,r.weight,gate,desc.m,&mw,desc.b,&bw));
            const pnsn_layer &l=ctx->layers[base+dir*2+(gate>=4)];int g=gate%4;
            std::vector<float>w(96*l.ni);for(int o=0;o<96;o++)for(int in=0;in<(int)l.ni;in++)w[o*l.ni+in]=value(ctx,l,g*96+o,in);
            OK(cudaMemcpy(mw,w.data(),w.size()*4,cudaMemcpyHostToDevice));if(bw)OK(cudaMemcpy(bw,l.bias+g*96,96*4,cudaMemcpyHostToDevice));
        }
    }
    return 0;
}
static int reserve(DNN *d,int batch){
    if(batch==d->batch)return 0;
    if(batch<=0 || (size_t)batch>INT_MAX/(d->window*(d->model?32:8)))return pnsn_error("cuDNN batch tensor exceeds indexing range");
    d->batch=0; /* A failed shape/workspace change must not leave a stale cache hit. */
    if((size_t)batch>d->capacity){
        /* Allocate replacement buffers before releasing the old set. */
        float *a=nullptr,*b=nullptr,*seq=nullptr,*rout=nullptr,*out=nullptr,*skips=nullptr,*up=nullptr;void *raw=nullptr;int *invalid=nullptr,*lengths=nullptr;
        cudaError_t status=cudaMalloc(&a,d->window*(d->model?16:8)*batch*4);
        if(status==cudaSuccess)status=cudaMalloc(&b,d->window*(d->model?16:8)*batch*4);
        if(status==cudaSuccess && !d->model)status=cudaMalloc(&seq,d->window/128*96*batch*4);
        if(status==cudaSuccess && !d->model)status=cudaMalloc(&rout,d->window/128*192*batch*4);
        if(status==cudaSuccess)status=cudaMalloc(&out,d->window*d->classes*batch*4);
        if(status==cudaSuccess && d->model)status=cudaMalloc(&skips,d->window*15*batch*4);
        if(status==cudaSuccess && d->model && !d->int8)status=cudaMalloc(&up,d->window*32*batch*4);
        if(status==cudaSuccess)status=cudaMalloc(&raw,d->window*3*batch*8);
        if(status==cudaSuccess)status=cudaMalloc(&invalid,4);
        if(status==cudaSuccess)status=cudaMalloc(&lengths,batch*4);
        if(status!=cudaSuccess){cudaFree(a);cudaFree(b);cudaFree(seq);cudaFree(rout);cudaFree(out);cudaFree(raw);cudaFree(invalid);cudaFree(lengths);cudaFree(skips);cudaFree(up);return check(status,__LINE__);}
        cudaFree(d->a);cudaFree(d->b);cudaFree(d->seq);cudaFree(d->rout);cudaFree(d->output);cudaFree(d->raw);cudaFree(d->invalid);cudaFree(d->lengths);cudaFree(d->skips);cudaFree(d->up);
        d->skips=skips;d->up=up;d->a=a;d->b=b;d->seq=seq;d->rout=rout;d->output=out;d->raw=raw;d->invalid=invalid;d->lengths=lengths;d->capacity=batch;
    }
    size_t need=0;int t=d->window;
    for(int i=0;i<25;i++){
        Conv &c=d->conv[i];if(!c.op)continue;c.t=t;c.source_t=c.upsample?t*4:t;c.u=c.transpose?t*2:(c.source_t+2*c.pad-c.k)/c.stride+1;
        if(tensor(c.x,batch,c.ni,c.source_t) || tensor(c.y,batch,c.no,c.u))return -1;
        int count=0;bool found=false;
        if(d->int8){
            for(int phase=0;phase<(c.transpose?2:1);phase++){
                IntConv &ic=c.integer[phase];ic.input_t=c.source_t+(c.transpose && !(ic.k&1));
                ic.rows=(batch<128?batch:128)*c.u;ic.last_rows=(batch%(batch<128?batch:128))*c.u;
                if(!ic.last_rows)ic.last_rows=ic.rows;
                OK(cudnnSetTensor4dDescriptor(ic.x,CUDNN_TENSOR_NHWC,CUDNN_DATA_INT8,d->model?1:batch,c.nic,1,d->model?ic.rows:ic.input_t));
                OK(cudnnSetTensor4dDescriptor(ic.y,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,d->model?1:batch,c.no,1,d->model?ic.rows:c.transpose?t:c.u));
                if(d->model){
                    OK(cudnnSetTensor4dDescriptor(ic.last_x,CUDNN_TENSOR_NHWC,CUDNN_DATA_INT8,1,c.nic,1,ic.last_rows));
                    OK(cudnnSetTensor4dDescriptor(ic.last_y,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,1,c.no,1,ic.last_rows));
                }
                cudnnConvolutionFwdAlgoPerf_t p[8];OK(cudnnGetConvolutionForwardAlgorithm_v7(d->handle,ic.x,ic.filter,ic.op,ic.y,8,&count,p));found=false;
                for(int j=0;j<count;j++)if(p[j].status==CUDNN_STATUS_SUCCESS && p[j].memory<=512ULL*1024*1024){ic.algo=p[j].algo;found=true;break;}
                if(!found)return pnsn_error("No cuDNN INT8 forward algorithm within 512 MiB workspace; no FP32 fallback");
                OK(cudnnGetConvolutionForwardWorkspaceSize(d->handle,ic.x,ic.filter,ic.op,ic.y,ic.algo,&ic.workspace));need=std::max(need,ic.workspace);
                if(d->model){size_t last_work;OK(cudnnGetConvolutionForwardWorkspaceSize(d->handle,ic.last_x,ic.filter,ic.op,ic.last_y,ic.algo,&last_work));ic.workspace=std::max(ic.workspace,last_work);need=std::max(need,ic.workspace);}
            }
        } else if(c.transpose){
            cudnnConvolutionBwdDataAlgoPerf_t p[8];OK(cudnnGetConvolutionBackwardDataAlgorithm_v7(d->handle,c.f,c.x,c.op,c.y,8,&count,p));
            for(int j=0;j<count;j++)if(p[j].status==CUDNN_STATUS_SUCCESS && p[j].memory<=512ULL*1024*1024){c.backward=p[j].algo;found=true;break;}
            if(!found)return pnsn_error("No cuDNN transposed convolution algorithm within 512 MiB workspace");
            OK(cudnnGetConvolutionBackwardDataWorkspaceSize(d->handle,c.f,c.x,c.op,c.y,c.backward,&c.workspace));
        }else{
            cudnnConvolutionFwdAlgoPerf_t p[8];OK(cudnnGetConvolutionForwardAlgorithm_v7(d->handle,c.x,c.f,c.op,c.y,8,&count,p));
            for(int j=0;j<count;j++)if(p[j].status==CUDNN_STATUS_SUCCESS && p[j].memory<=512ULL*1024*1024){c.forward=p[j].algo;found=true;break;}
            if(!found)return pnsn_error("No cuDNN convolution algorithm within 512 MiB workspace");
            OK(cudnnGetConvolutionForwardWorkspaceSize(d->handle,c.x,c.f,c.op,c.y,c.forward,&c.workspace));
        }
        need=std::max(need,c.workspace);
        if(!d->model && i<7){if(tensor(d->px[i],batch,c.no,t,d->int8) || tensor(d->py[i],batch,c.no,t/2,d->int8))return -1;t/=2;}
        if(d->model)t=c.u;else if(c.transpose)t*=2;
    }
    if(d->int8){
        size_t qbytes=0,abytes=0;
        for(const auto &c:d->conv)if(c.op){
            for(const auto &ic:c.integer)if(ic.op)qbytes=std::max(qbytes,(size_t)c.nic*(d->model?ic.rows:batch*ic.input_t));
            if(c.transpose)abytes=std::max(abytes,(size_t)batch*c.no*c.t*4);
        }
        if(qbytes>d->quant_bytes){int8_t *q=nullptr;OK(cudaMalloc(&q,qbytes));cudaFree(d->quant);d->quant=q;d->quant_bytes=qbytes;}
        if(abytes>d->accum_bytes){float *acc=nullptr;OK(cudaMalloc(&acc,abytes));cudaFree(d->accum);d->accum=acc;d->accum_bytes=abytes;}
        size_t scales=(d->model?(size_t)(batch<128?batch:128)*d->window:INT_MAX/(d->window*8))*sizeof(float);
        if(scales>d->scale_bytes){float *s=nullptr;OK(cudaMalloc(&s,scales));cudaFree(d->iscale);d->iscale=s;d->scale_bytes=scales;}
    }
    if(d->model){if(need>d->workbytes){void *work=nullptr;OK(cudaMalloc(&work,need));cudaFree(d->work);d->work=work;d->workbytes=need;}d->batch=batch;return 0;}
    t=d->window/128;std::vector<int>lens(batch,t);OK(cudaMemcpyAsync(d->lengths,lens.data(),batch*4,cudaMemcpyHostToDevice,d->stream));
    for(auto &r:d->rnn){
        OK(cudnnSetRNNDataDescriptor(r.x,CUDNN_DATA_FLOAT,CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,t,batch,r.ni,lens.data(),nullptr));
        OK(cudnnSetRNNDataDescriptor(r.y,CUDNN_DATA_FLOAT,CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,t,batch,192,lens.data(),nullptr));
        int shape[3]={2,batch,96},strides[3]={batch*96,96,1};OK(cudnnSetTensorNdDescriptor(r.hidden,CUDNN_DATA_FLOAT,3,shape,strides));
        size_t reservebytes;OK(cudnnGetRNNTempSpaceSizes(d->handle,r.op,CUDNN_FWD_MODE_INFERENCE,r.x,&r.workspace,&reservebytes));need=std::max(need,r.workspace);
    }
    if(need>d->workbytes){void *work=nullptr;OK(cudaMalloc(&work,need));cudaFree(d->work);d->work=work;d->workbytes=need;}
    d->batch=batch;return 0;
}
__global__ void bias_relu(float *x,const float *bias,int total,int channels,int t,int relu){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at<total){float v=x[at]+bias[(at/t)%channels];x[at]=relu?fmaxf(v,0):v;}}
__global__ void layout(const float *x,float *y,int batch,int c,int t,int to_seq,int interleaved){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=batch*c*t)return;int time=interleaved?(at/c)%t:at%t,channel=interleaved?at%c:(at/t)%c,b=at/(t*c),seq=(time*batch+b)*c+channel;if(to_seq)y[seq]=x[at];else y[at]=x[seq];}
__global__ void probabilities(const float *x,float *y,int batch,int t,int interleaved,int classes){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=batch*t)return;int b=at/t,time=at%t;float maxv=x[interleaved?at*classes:b*classes*t+time],sum=0;for(int c=1;c<classes;c++)maxv=fmaxf(maxv,x[interleaved?at*classes+c:(b*classes+c)*t+time]);for(int c=0;c<classes;c++){float v=expf(x[interleaved?at*classes+c:(b*classes+c)*t+time]-maxv);y[at*classes+c]=v;sum+=v;}for(int c=0;c<classes;c++)y[at*classes+c]/=sum;}
template<typename T>
__global__ void preprocess(const T *x,float *y,int w,int normalize,int diff,int *invalid,int interleaved) {
    __shared__ double reduction[128],mean,scale;
    int tid=threadIdx.x,b=blockIdx.x/3,c=blockIdx.x%3;
    const T *record=x+(size_t)b*3*w+c*w;double origin=(double)record[0],sum=0;
    for(int t=tid;t<w;t+=128) {
        double raw=(double)record[t],v=diff?(double)record[t+1<w?t+1:t]-raw:raw-origin;
        if(!isfinite(raw) || !isfinite(v))atomicExch(invalid,1);
        if(!normalize)y[interleaved?((size_t)b*w+t)*3+c:((size_t)b*3+c)*w+t]=(float)raw;else sum+=v;
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
        y[interleaved?((size_t)b*w+t)*3+c:((size_t)b*3+c)*w+t]=(float)(v/scale);
    }
}
__global__ void quantize_record(const float *x,int8_t *q,float *scales,int ni,int nic,int t,int input_t,int repeat){
    __shared__ float maxima[256],scale;int b=blockIdx.x,tid=threadIdx.x;float maxv=0;
    for(int at=tid;at<ni*t;at+=256)maxv=fmaxf(maxv,fabsf(x[(size_t)b*ni*t+at]));
    maxima[tid]=maxv;__syncthreads();for(int s=128;s;s/=2){if(tid<s)maxima[tid]=fmaxf(maxima[tid],maxima[tid+s]);__syncthreads();}
    if(tid==0){scale=fmaxf(maxima[0]/127,1e-30f);scales[b]=scale;}__syncthreads();
    /* Quantize nearest-neighbor source once, then replicate integer values.
     * This is bit-identical to quantizing all four repeated floats separately. */
    for(int at=tid;at<nic*t;at+=256){
        int time=at/nic,c=at%nic;float v=c<ni?x[((size_t)b*t+time)*ni+c]:0;
        int8_t code=(int8_t)fmaxf(-127,fminf(127,nearbyintf(v/scale)));
        for(int r=0;r<repeat;r++)q[((size_t)b*input_t+time*repeat+r)*nic+c]=code;
    }
    for(int at=tid;at<(input_t-t*repeat)*nic;at+=256)q[(size_t)b*nic*input_t+t*repeat*nic+at]=0;
}
__global__ void dequantize_conv(const float *acc,float *y,const float *xs,const float *ws,const float *bias,int batch,int no,int t,int transpose,int first,int relu){
    int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=batch*no*t)return;
    int c=at%no,time=at/no%t,b=at/(no*t);float v=__fadd_rn(__fmul_rn(__fmul_rn(acc[at],xs[b]),ws[c]),bias[c]);if(relu)v=fmaxf(v,0);
    y[((size_t)b*(transpose?t*2:t)+(transpose?time*2+first:time))*no+c]=v;
}
__global__ void repeat_time(const float *x,float *y,int batch,int ni,int t){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=batch*ni*t*4)return;int time=at%(t*4),c=(at/(t*4))%ni,b=at/(t*4*ni);y[at]=x[(b*ni+c)*t+time/4];}
__global__ void merge_skip(const float *skip,const float *up,float *out,int batch,int no,int t,int interleaved){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=batch*no*t*2)return;int b=at/(no*t*2),time=interleaved?(at/(no*2))%t:at%t,c=interleaved?at%(no*2):(at/t)%(no*2);int index=interleaved?(b*t+time)*no+c%no:(b*no+c%no)*t+time;out[at]=(c<no?skip:up)[index];}
__device__ float patch_value(const float *x,int global,int j,int ni,int t,int u,int k,int pad,int stride,int repeat){
    int b=global/u,row=global%u,kernel=j/ni,c=j%ni;if(kernel>=k)return 0;
    int source=row*(repeat==4?1:stride)+kernel-pad;if(source<0 || source>=t*repeat)return 0;source/=repeat;
    return x[((size_t)b*t+source)*ni+c];
}
__global__ void quantize_patches(const float *x,int8_t *q,float *scales,int begin,int rows,int ni,int width,int t,int u,int k,int pad,int stride,int repeat){
    int lane=threadIdx.x%32,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(row>=rows)return;
    float maxv=0;for(int j=lane;j<width;j+=32)maxv=fmaxf(maxv,fabsf(patch_value(x,begin+row,j,ni,t,u,k,pad,stride,repeat)));
    for(int s=16;s;s/=2)maxv=fmaxf(maxv,__shfl_down_sync(0xffffffff,maxv,s));
    float scale=fmaxf(__shfl_sync(0xffffffff,maxv,0)/127,1e-30f);if(lane==0)scales[row]=scale;
    for(int j=lane;j<width;j+=32){float v=patch_value(x,begin+row,j,ni,t,u,k,pad,stride,repeat);q[(size_t)row*width+j]=(int8_t)fmaxf(-127,fminf(127,nearbyintf(v/scale)));}
}
__global__ void dequantize_rows(float *y,const float *xs,const float *ws,const float *bias,int rows,int no,int relu){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=rows*no)return;int row=at/no,c=at%no;float v=__fadd_rn(__fmul_rn(__fmul_rn(y[at],xs[row]),ws[c]),bias[c]);y[at]=relu?fmaxf(v,0):v;}
static int convolution(DNN *d,int index,const float *x,float *y,int relu){
    Conv &c=d->conv[index];float one=1,zero=0;
    if(d->int8 && d->model){
        IntConv &ic=c.integer[0];int total=d->batch*c.u;
        for(int begin=0;begin<total;begin+=ic.rows){
            int rows=std::min(ic.rows,total-begin);float *out=y+(size_t)begin*c.no;
            quantize_patches<<<(rows+7)/8,256,0,d->stream>>>(x,d->quant,d->iscale,begin,rows,c.ni,c.nic,c.t,c.u,c.k,c.pad,c.stride,c.upsample?4:1);
            OK(cudnnConvolutionForward(d->handle,&one,rows==ic.rows?ic.x:ic.last_x,d->quant,ic.filter,ic.weight,ic.op,ic.algo,d->work,ic.workspace,&zero,rows==ic.rows?ic.y:ic.last_y,out));
            int n=rows*c.no;dequantize_rows<<<(n+255)/256,256,0,d->stream>>>(out,d->iscale,c.scale,c.bias,rows,c.no,relu);
        }return 0;
    }
    if(d->int8){
        for(int phase=0;phase<(c.transpose?2:1);phase++){
            IntConv &ic=c.integer[phase];quantize_record<<<d->batch,256,0,d->stream>>>(x,d->quant,d->iscale,c.ni,c.nic,c.t,ic.input_t,c.upsample?4:1);
            float *acc=c.transpose?d->accum:y; /* Normal convolution dequantizes in place. */
            OK(cudnnConvolutionForward(d->handle,&one,ic.x,d->quant,ic.filter,ic.weight,ic.op,ic.algo,d->work,ic.workspace,&zero,ic.y,acc));
            int rows=c.transpose?c.t:c.u;int count=d->batch*c.no*rows;dequantize_conv<<<(count+255)/256,256,0,d->stream>>>(acc,y,d->iscale,c.scale,c.bias,d->batch,c.no,rows,c.transpose,ic.first,relu);
        }
        return 0;
    }
    if(c.upsample){int n=d->batch*c.ni*c.source_t;repeat_time<<<(n+255)/256,256,0,d->stream>>>(x,d->up,d->batch,c.ni,c.t);x=d->up;}
    if(c.transpose){OK(cudnnConvolutionBackwardData(d->handle,&one,c.f,c.weight,c.x,x,c.op,c.backward,d->work,c.workspace,&zero,c.y,y));}
    else {OK(cudnnConvolutionForward(d->handle,&one,c.x,x,c.f,c.weight,c.op,c.forward,d->work,c.workspace,&zero,c.y,y));}
    int count=d->batch*c.no*c.u;bias_relu<<<(count+255)/256,256,0,d->stream>>>(y,c.bias,count,c.no,c.u,relu);return 0;
}
static int china_network(DNN *d){
    float *x=d->a,*y=d->b,*skips[4];size_t offset=0;
    for(int i=0;i<6;i++){
        Conv &c=d->conv[i];if(convolution(d,i,x,y,1))return -1;
        if(i>=1 && i<=4){size_t bytes=(size_t)d->batch*c.no*c.u*4;skips[i-1]=d->skips+offset;OK(cudaMemcpyAsync(skips[i-1],y,bytes,cudaMemcpyDeviceToDevice,d->stream));offset+=bytes/4;}
        std::swap(x,y);
    }
    for(int i=6;i<10;i++){
        Conv &c=d->conv[i];if(convolution(d,i,x,y,1))return -1;
        int n=d->batch*c.no*c.u*2;merge_skip<<<(n+255)/256,256,0,d->stream>>>(skips[9-i],y,x,d->batch,c.no,c.u,d->int8);
    }
    if(convolution(d,10,x,y,0))return -1;
    probabilities<<<(d->batch*d->window+255)/256,256,0,d->stream>>>(y,d->output,d->batch,d->window,d->int8,d->classes);OK(cudaGetLastError());return 0;
}
static int network(DNN *d){
    if(d->model)return china_network(d);
    float *x=d->a,*y=d->b,one=1,zero=0;int t=d->window,batch=d->batch;
    for(int i=0;i<7;i++){
        if(convolution(d,i,x,y,1))return -1;OK(cudnnPoolingForward(d->handle,d->pool,&one,d->px[i],y,&zero,d->py[i],x));t/=2;
    }
    for(int q=0;q<2;q++){
        RNN &r=d->rnn[q];int count=batch*t*r.ni;layout<<<(count+255)/256,256,0,d->stream>>>(x,d->seq,batch,r.ni,t,1,d->int8);
        OK(cudnnRNNForward(d->handle,r.op,CUDNN_FWD_MODE_INFERENCE,d->lengths,r.x,d->seq,r.y,d->rout,r.hidden,nullptr,nullptr,r.hidden,nullptr,nullptr,r.bytes,r.weight,d->workbytes,d->work,0,nullptr));
        count=batch*t*192;layout<<<(count+255)/256,256,0,d->stream>>>(d->rout,y,batch,192,t,0,d->int8);
        if(convolution(d,11+q*5,y,x,0))return -1;
    }
    for(int i=17;i<25;i++){if(convolution(d,i,x,y,i!=24))return -1;std::swap(x,y);if(i!=24)t*=2;}
    probabilities<<<(batch*t+255)/256,256,0,d->stream>>>(x,d->output,batch,t,d->int8,d->classes);OK(cudaGetLastError());return 0;
}
void pnsn_cuda_close(void *context){delete (DNN*)context;}
void *pnsn_cuda_open(pnsn_context *ctx){
    DNN *d=new(std::nothrow)DNN;if(!d){pnsn_error("cuDNN host out of memory");return nullptr;}
    try {if(init(d,ctx)){delete d;return nullptr;}return d;}
    catch(const std::bad_alloc &){delete d;pnsn_error("cuDNN host out of memory");return nullptr;}
}
static int forward_batch_impl(void *context,const void *input,pnsn_dtype dtype,size_t batch,int normalize,int diff,float *out){
    DNN *d=(DNN*)context;if(batch>INT_MAX || reserve(d,(int)batch))return -1;
    size_t bytes=batch*d->window*3*(dtype==PNSN_F64?8:4);int invalid=0;float prep,net;
    OK(cudaMemcpyAsync(d->raw,input,bytes,cudaMemcpyHostToDevice,d->stream));OK(cudaMemsetAsync(d->invalid,0,4,d->stream));OK(cudaEventRecord(d->before,d->stream));
    if(dtype==PNSN_F64)preprocess<<<batch*3,128,0,d->stream>>>((const double*)d->raw,d->a,d->window,normalize,diff,d->invalid,d->int8);
    else if(dtype==PNSN_I32)preprocess<<<batch*3,128,0,d->stream>>>((const int32_t*)d->raw,d->a,d->window,normalize,diff,d->invalid,d->int8);
    else preprocess<<<batch*3,128,0,d->stream>>>((const float*)d->raw,d->a,d->window,normalize,diff,d->invalid,d->int8);
    OK(cudaEventRecord(d->normalized,d->stream));if(network(d))return -1;OK(cudaEventRecord(d->after,d->stream));
    OK(cudaMemcpyAsync(out,d->output,batch*d->window*d->classes*4,cudaMemcpyDeviceToHost,d->stream));OK(cudaMemcpyAsync(&invalid,d->invalid,4,cudaMemcpyDeviceToHost,d->stream));OK(cudaStreamSynchronize(d->stream));
    if(invalid)return pnsn_error("Non-finite batch waveform or overflowing float64 normalization");
    OK(cudaEventElapsedTime(&prep,d->before,d->normalized));OK(cudaEventElapsedTime(&net,d->normalized,d->after));d->prep_ms=prep;d->network_ms=net;return 0;
}
int pnsn_cuda_forward_batch(void *context,const void *input,pnsn_dtype dtype,size_t batch,int normalize,int diff,float *out){
    try{return forward_batch_impl(context,input,dtype,batch,normalize,diff,out);}
    catch(const std::bad_alloc &){return pnsn_error("cuDNN host out of memory");}
}
int pnsn_cuda_forward(void *context,const float *input,float *out){
    try {
    DNN *d=(DNN*)context;std::vector<float>planar(d->window*3);
    for(size_t t=0;t<d->window;t++)for(int c=0;c<3;c++)planar[c*d->window+t]=input[t*3+c];
    return pnsn_cuda_forward_batch(d,planar.data(),PNSN_F32,1,0,0,out);
    } catch(const std::bad_alloc &){return pnsn_error("cuDNN host out of memory");}
}
int pnsn_cuda_batch_stats(void *context,double *prep,double *net,size_t *bytes){
    DNN *d=(DNN*)context;*prep=d->prep_ms;*net=d->network_ms;
    *bytes=d->capacity*(d->model?d->window*((32+15+3+(!d->int8?32:0))*4+3*8)+4:d->window*(16*4+5*4+3*8)+d->window/128*288*4+4)+4+d->workbytes;
    *bytes+=d->quant_bytes+d->accum_bytes+d->scale_bytes;
    for(const auto &c:d->conv)*bytes+=c.bytes;for(const auto &r:d->rnn)*bytes+=r.bytes;return 0;
}
