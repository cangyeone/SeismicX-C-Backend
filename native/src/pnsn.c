#include "internal.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL _Thread_local
#endif
static THREAD_LOCAL char error_text[256];
int pnsn_error(const char *message) { snprintf(error_text, sizeof(error_text), "%s", message); return -1; }
const char *pnsn_last_error(void) { return error_text; }
pnsn_options pnsn_default_options(void) {
    pnsn_options o = {10240, 1024, 2000000, .1f, 300, 1, 0}; return o;
}
static int read_u32(FILE *f, uint32_t *v) {
    unsigned char b[4]; if (fread(b, 1, 4, f)!=4) return 0;
    *v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24); return 1;
}
static int read_floats(FILE *f, float *v, size_t n) {
    for(size_t i=0;i<n;i++) { uint32_t bits; if(!read_u32(f,&bits)) return 0; memcpy(v+i,&bits,4); if(!isfinite(v[i])) return 0; } return 1;
}
static int valid_layer(int model,int index, pnsn_layer *l) {
    static const uint32_t e[25][6] = {
        {1,3,8,11,1,5},{1,8,16,9,1,4},{1,16,16,7,1,3},{1,16,32,7,1,3},
        {1,32,32,5,1,2},{1,32,64,5,1,2},{1,64,64,3,1,1},
        {0,64,384,1,1,0},{0,96,384,1,1,0},{0,64,384,1,1,0},{0,96,384,1,1,0},{1,192,96,1,1,0},
        {0,96,384,1,1,0},{0,96,384,1,1,0},{0,96,384,1,1,0},{0,96,384,1,1,0},{1,192,96,1,1,0},
        {2,96,96,3,2,1},{2,96,96,5,2,2},{2,96,32,5,2,2},{2,32,32,7,2,3},
        {2,32,16,7,2,3},{2,16,16,9,2,4},{2,16,8,11,2,5},{1,8,5,11,1,5}};
    static const uint32_t china[11][6]={
        {1,3,8,7,1,3},{1,8,8,7,1,3},{1,8,16,7,4,3},{1,16,32,7,4,3},
        {1,32,64,7,4,3},{1,64,128,7,4,3},{3,128,64,7,4,3},
        {3,128,32,7,4,3},{3,64,16,7,4,3},{3,32,8,7,4,3},{1,16,3,7,1,3}};
    uint32_t a[]={l->kind,l->ni,l->no,l->kernel,l->stride,l->pad};
    return !memcmp(model?china[index]:e[index],a,sizeof(a)) && l->width==((l->ni*l->kernel+15)/16)*16;
}
void pnsn_close(pnsn_context *c) {
    if(!c) return;
#ifdef PNSN_CUDA
    if(c->gpu) pnsn_cuda_close(c->gpu);
#endif
    for(int i=0;i<PNSN_LAYERS;i++) { free(c->layers[i].scale); free(c->layers[i].bias); free(c->layers[i].w); free(c->layers[i].q); }
    for(int i=0;i<7;i++)for(int p=0;p<2;p++) {free(c->transpose[i][p].w);free(c->transpose[i][p].q);}
    free(c->a);free(c->b);free(c->c);free(c->patch);free(c->prob);free(c->quant);free(c->skips);free(c);
}
pnsn_context *pnsn_open(const char *path, const pnsn_options *options) {
    error_text[0]=0;
    if(!path) {pnsn_error("Null model path");return NULL;}
    pnsn_options o=options?*options:pnsn_default_options();
    if(o.window<128 || o.window>1048576 || o.window%128 || o.overlap>=o.window || o.threads<1 || o.threads>256 ||
       o.max_candidates==0 || o.max_candidates>SIZE_MAX/sizeof(pnsn_pick) || !isfinite(o.threshold) || o.threshold<0 || o.threshold>1 || o.min_gap<0 || (o.difference!=0 && o.difference!=1)) {
        pnsn_error("Invalid options: window must be a multiple of 128 in [128,1048576]"); return NULL;
    }
    FILE *f=fopen(path,"rb"); if(!f) {pnsn_error("Cannot open model");return NULL;}
    pnsn_context *c=calloc(1,sizeof(*c)); if(!c) {fclose(f);pnsn_error("Out of memory");return NULL;} c->options=o;
    unsigned char magic[8];uint32_t count,bits;
    if(fread(magic,1,8,f)!=8 || !read_u32(f,&count) || !read_u32(f,&bits) || (bits!=8 && bits!=32)) goto bad;
    if(!memcmp(magic,"PNSNC01\0",8) && count==25){c->model=0;c->classes=5;}
    else if(!memcmp(magic,"SXUNET1\0",8) && count==11){
        if(o.window%256){pnsn_error("China UNet window must be a multiple of 256");fclose(f);pnsn_close(c);return NULL;}
        c->model=1;c->classes=3;
    }
    else goto bad;
    c->bits=(int)bits;
    for(int i=0;i<(int)count;i++) {
        pnsn_layer *l=&c->layers[i]; uint32_t d[7];
        for(int j=0;j<7;j++) if(!read_u32(f,d+j)) goto bad;
        l->kind=d[0];l->ni=d[1];l->no=d[2];l->kernel=d[3];l->stride=d[4];l->pad=d[5];l->width=d[6];
        if(!valid_layer(c->model,i,l)) goto bad;
        size_t n=(size_t)l->no*l->width;
        l->scale=malloc(l->no*sizeof(float));l->bias=malloc(l->no*sizeof(float));
        if(bits==8) l->q=malloc(n);else l->w=malloc(n*sizeof(float));
        if(!l->scale || !l->bias || (bits==8?!l->q:!l->w)) goto oom;
        if(!read_floats(f,l->scale,l->no) || !read_floats(f,l->bias,l->no)) goto bad;
        for(uint32_t j=0;j<l->no;j++) if(l->scale[j]<=0) goto bad;
        if(bits==8) {if(fread(l->q,1,n,f)!=n) goto bad;} else if(!read_floats(f,l->w,n)) goto bad;
        if(l->width>c->max_width) c->max_width=l->width;
    }
    if(fgetc(f)!=EOF) goto bad;
    fclose(f);f=NULL;
    size_t feature=c->model?16:8;
    c->a=calloc(o.window*feature,sizeof(float));c->b=calloc(o.window*feature,sizeof(float));c->c=calloc(o.window*feature,sizeof(float));
    if(c->model)c->skips=calloc(o.window*15,sizeof(float));
    c->patch=calloc(PNSN_TILE*c->max_width,sizeof(float));c->quant=calloc(PNSN_TILE*c->max_width,1);c->prob=calloc(o.window*c->classes,sizeof(float));
    if(!c->a || !c->b || !c->c || !c->patch || !c->quant || !c->prob || (c->model && !c->skips)) goto oom;
    if(pnsn_cpu_init(c)) {pnsn_close(c);return NULL;}
#ifdef PNSN_CUDA
    c->gpu=pnsn_cuda_open(c);if(!c->gpu) {pnsn_close(c);return NULL;}
#endif
    return c;
bad: pnsn_error("Invalid, truncated or incompatible SeismicX model");if(f) fclose(f);pnsn_close(c);return NULL;
oom: pnsn_error("Out of memory");if(f) fclose(f);pnsn_close(c);return NULL;
}
int pnsn_classes(const pnsn_context *ctx){return ctx?ctx->classes:0;}
const char *pnsn_model_name(const pnsn_context *ctx){return !ctx?"unknown":ctx->model?"China PhaseNet-style UNet":"PNSN v3";}
size_t pnsn_memory_bytes(const pnsn_context *ctx) {
    if(!ctx)return 0;
    size_t bytes=sizeof(*ctx)+ctx->options.window*(ctx->model?66:29)*sizeof(float)+PNSN_TILE*ctx->max_width*(sizeof(float)+1);
    for(int i=0;i<25;i++)bytes+=ctx->layers[i].no*(2*sizeof(float)+ctx->layers[i].width*(ctx->bits==8?1:sizeof(float)));
    for(int i=0;i<7;i++)for(int p=0;p<2;p++)bytes+=(size_t)ctx->transpose[i][p].no*ctx->transpose[i][p].width*(ctx->bits==8?1:sizeof(float));
    return bytes;
}
static double sample_at(const void *x,pnsn_dtype dtype,size_t index) {
    if(dtype==PNSN_F64) return ((const double*)x)[index];
    if(dtype==PNSN_I32) return ((const int32_t*)x)[index];
    return ((const float*)x)[index];
}
static int normalize_layout(const void *input,pnsn_dtype dtype,size_t n,size_t start,size_t w,int diff,float *out,int planar) {
    if(!input || !n || n>SIZE_MAX/3 || dtype<PNSN_F32 || dtype>PNSN_I32) return pnsn_error("Invalid waveform");
    /* Shift first, then Welford in float64: preserve small signals on huge ADC offsets. */
    for(size_t c=0;c<3;c++) {
        size_t first=start<n?start:n-1;
        double origin=sample_at(input,dtype,planar?c*n+first:first*3+c),mean=0,m2=0;
        if(!isfinite(origin)) return pnsn_error("Non-finite waveform");
        for(size_t j=0;j<w;j++) {
            size_t i=j<n-start?start+j:n-1;
            double raw=sample_at(input,dtype,planar?c*n+i:i*3+c);
            double v=raw-origin;
            if(diff) {double next=sample_at(input,dtype,planar?c*n+(i+1<n?i+1:i):(i+1<n?i+1:i)*3+c);v=next-raw;}
            if(!isfinite(raw) || !isfinite(v)) return pnsn_error("Non-finite waveform or overflowing difference");
            double delta=v-mean;mean+=delta/(double)(j+1);m2+=delta*(v-mean);
        }
        double scale=sqrt(fmax(0,m2)/(double)(w-1))+1e-6;
        if(!isfinite(mean) || !isfinite(scale)) return pnsn_error("Waveform range exceeds float64 normalization");
        for(size_t j=0;j<w;j++) {
            size_t i=j<n-start?start+j:n-1;
            double raw=sample_at(input,dtype,planar?c*n+i:i*3+c),v=raw-origin;
            if(diff) v=sample_at(input,dtype,planar?c*n+(i+1<n?i+1:i):(i+1<n?i+1:i)*3+c)-raw;
            out[j*3+c]=(float)((v-mean)/scale);
        }
    }
    return 0;
}
int pnsn_normalize(const void *input,pnsn_dtype dtype,size_t n,size_t start,size_t w,int diff,float *out) {
    return normalize_layout(input,dtype,n,start,w,diff,out,0);
}
int pnsn_preprocess(const void *input,pnsn_dtype dtype,size_t n,size_t start,size_t w,int diff,float *out) {
    if(!out || !n || start>=n || w<2 || w>1048576 || (diff!=0 && diff!=1)) return pnsn_error("Invalid preprocessing arguments");
    return pnsn_normalize(input,dtype,n,start,w,diff,out);
}
int pnsn_preprocess_channels(const void *input,pnsn_dtype dtype,size_t n,size_t start,size_t w,int diff,float *out) {
    if(!out || !n || start>=n || w<2 || w>1048576 || (diff!=0 && diff!=1))return pnsn_error("Invalid planar preprocessing arguments");
    return normalize_layout(input,dtype,n,start,w,diff,out,1);
}
int pnsn_preprocess_batch(const void *input,pnsn_dtype dtype,size_t batch,size_t w,int diff,float *out) {
    if(dtype<PNSN_F32 || dtype>PNSN_I32 || w<2 || w>1048576 || (diff!=0 && diff!=1) || (batch && (!input || !out)) || batch>SIZE_MAX/(w*3*(dtype==PNSN_F64?8:4)))return pnsn_error("Invalid batch preprocessing arguments");
    size_t bytes=w*3*(dtype==PNSN_F64?8:4);
    for(size_t b=0;b<batch;b++)if(normalize_layout((const unsigned char*)input+b*bytes,dtype,w,0,w,diff,out+b*w*3,1))return -1;
    return 0;
}
static int forward(pnsn_context *c,float *out) {
#ifdef PNSN_CUDA
    return pnsn_cuda_forward(c->gpu,c->a,out);
#else
    pnsn_cpu_forward(c,out);return 0;
#endif
}
int pnsn_predict(pnsn_context *c,const void *input,pnsn_dtype dtype,size_t n,float *out) {
    if(!c || !out || n!=c->options.window) return pnsn_error("Fixed inference requires samples == configured window");
    if(pnsn_normalize(input,dtype,n,0,n,0,c->a)) return -1;
    return forward(c,out);
}
static int pick_history_layout(pnsn_context *c,const void *input,pnsn_dtype dtype,size_t n,pnsn_pick *out,size_t cap,size_t *count,int planar) {
    if(!c || !count || (!input && n) || (cap && !out) || dtype<PNSN_F32 || dtype>PNSN_I32 || n>INT64_MAX || n>SIZE_MAX/3) return pnsn_error("Invalid picking arguments");
    *count=0;if(!n) return 0;
    pnsn_pick *candidates=NULL;size_t used=0,allocated=0,w=c->options.window,step=w-c->options.overlap;
    for(size_t start=0;;) {
        for(int diff=0;diff<=c->options.difference;diff++) {
            if(normalize_layout(input,dtype,n,start,w,diff,c->a,planar) || forward(c,c->prob)) {free(candidates);return -1;}
            size_t valid=n-start<w?n-start:w;
            for(size_t t=0;t<valid;t++) for(int phase=0;phase<c->classes-1;phase++) {
                float s=c->prob[t*c->classes+phase+1];if(s<=c->options.threshold) continue;
                if(used==c->options.max_candidates) {free(candidates);return pnsn_error("Candidate memory budget exceeded; raise max_candidates or threshold");}
                if(used==allocated) {
                    size_t next=allocated?allocated*2:1024;
                    if(next>c->options.max_candidates || next<allocated) next=c->options.max_candidates;
                    pnsn_pick *p=realloc(candidates,next*sizeof(*p));if(!p) {free(candidates);return pnsn_error("Out of memory");} candidates=p;allocated=next;
                }
                candidates[used++]=(pnsn_pick){phase,(int64_t)(start+t),s};
            }
        }
        if(n-start<=w) break;
        start+=step;
    }
    int status=pnsn_nms(candidates,used,c->options.min_gap,out,cap,count);free(candidates);return status;
}

int pnsn_pick_trace(pnsn_context *c,const void *input,pnsn_dtype dtype,size_t n,pnsn_pick *out,size_t cap,size_t *count) {
    return pick_history_layout(c,input,dtype,n,out,cap,count,0);
}
int pnsn_pick_history(pnsn_context *c,const void *input,pnsn_dtype dtype,size_t n,pnsn_pick *out,size_t cap,size_t *count) {
    return pick_history_layout(c,input,dtype,n,out,cap,count,1);
}
static int predict_batch_impl(pnsn_context *c,const void *input,pnsn_dtype dtype,size_t batch,size_t w,float *out,int normalize,int diff) {
    if(!c || w!=c->options.window || dtype<PNSN_F32 || dtype>PNSN_I32 || (batch && (!input || !out)) ||
       batch>SIZE_MAX/(w*c->classes*sizeof(float)) || batch>SIZE_MAX/(w*3*(dtype==PNSN_F64?8:4))) return pnsn_error("Invalid batch: expected contiguous [B,3,configured_window]");
    if(!batch)return 0;
#ifdef PNSN_CUDA
    return pnsn_cuda_forward_batch(c->gpu,input,dtype,batch,normalize,diff,out);
#else
    size_t item_size=dtype==PNSN_F64?sizeof(double):sizeof(float);
    for(size_t b=0;b<batch;b++) {
        const unsigned char *record=(const unsigned char*)input+b*3*w*item_size;
        if(normalize) {if(normalize_layout(record,dtype,w,0,w,diff,c->a,1))return -1;}
        else for(size_t t=0;t<w;t++)for(size_t channel=0;channel<3;channel++) {
            float value=((const float*)record)[channel*w+t];
            if(!isfinite(value))return pnsn_error("Non-finite normalized batch");
            c->a[t*3+channel]=value;
        }
        if(forward(c,out+b*w*c->classes))return -1;
    }
    return 0;
#endif
}
int pnsn_predict_batch(pnsn_context *c,const void *input,pnsn_dtype dtype,size_t b,size_t w,float *out) {
    return predict_batch_impl(c,input,dtype,b,w,out,1,0);
}
int pnsn_predict_batch_normalized(pnsn_context *c,const float *input,size_t b,size_t w,float *out) {
    return predict_batch_impl(c,input,PNSN_F32,b,w,out,0,0);
}
int pnsn_batch_device_stats(pnsn_context *c,double *prep,double *network,size_t *bytes) {
    if(!c || !prep || !network || !bytes)return pnsn_error("Invalid stats arguments");
#ifdef PNSN_CUDA
    return pnsn_cuda_batch_stats(c->gpu,prep,network,bytes);
#else
    *prep=0;*network=0;*bytes=0;return 0;
#endif
}
int pnsn_pick_batch(pnsn_context *c,const void *input,pnsn_dtype dtype,size_t b,size_t w,pnsn_pick *out,size_t cap,size_t *offsets,size_t *count) {
    if(!c || !count || !offsets || w!=c->options.window || (cap && !out) || b>SIZE_MAX/(w*c->classes*sizeof(float)) || b>SIZE_MAX/sizeof(size_t)-1) return pnsn_error("Invalid batch picking arguments");
    *count=0;if(!b){offsets[0]=0;return 0;}
    size_t per_wave=(c->classes-1)*(w/((uint64_t)c->options.min_gap+1)+1);
    if(per_wave>c->options.max_candidates)per_wave=c->options.max_candidates;
    if(b>SIZE_MAX/(per_wave*sizeof(pnsn_pick)))return pnsn_error("Batch output overflow");
    size_t max_c=(c->classes-1)*w*(c->options.difference+1);
    if(max_c>c->options.max_candidates)max_c=c->options.max_candidates;
    float *p=malloc(b*w*c->classes*sizeof(float)),*diff=c->options.difference?malloc(b*w*c->classes*sizeof(float)):NULL;
    pnsn_pick *candidates=malloc(max_c*sizeof(*candidates)),*selected=malloc(b*per_wave*sizeof(*selected));
    size_t *bounds=malloc((b+1)*sizeof(*bounds));int status=-1;size_t used=0;
    if(!p || (c->options.difference && !diff) || !candidates || !selected || !bounds){pnsn_error("Out of memory in batch NMS");goto done;}
    if(predict_batch_impl(c,input,dtype,b,w,p,1,0) || (diff && predict_batch_impl(c,input,dtype,b,w,diff,1,1)))goto done;
    for(size_t row=0;row<b;row++) {
        bounds[row]=used;size_t n=0;
        for(int pass=0;pass<=c->options.difference;pass++)for(size_t t=0;t<w;t++)for(int phase=0;phase<c->classes-1;phase++) {
            float score=(pass?diff:p)[(row*w+t)*c->classes+phase+1];if(score<=c->options.threshold)continue;
            if(n==max_c){pnsn_error("Per-record candidate memory budget exceeded");goto done;}
            candidates[n++]=(pnsn_pick){phase,(int64_t)t,score};
        }
        size_t picked=0;
        if(pnsn_nms(candidates,n,c->options.min_gap,selected+used,per_wave,&picked))goto done;
        used+=picked;
    }
    bounds[b]=used;*count=used;
    if(used>cap){pnsn_error("Batch pick capacity insufficient");status=-2;goto done;}
    if(used)memcpy(out,selected,used*sizeof(*out));
    memcpy(offsets,bounds,(b+1)*sizeof(*bounds));status=0;
done:free(p);free(diff);free(candidates);free(selected);free(bounds);return status;
}
