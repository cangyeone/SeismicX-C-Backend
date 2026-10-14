#include "internal.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#if (defined(__ARM_NEON) && defined(__aarch64__))
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
static float dot_f32(const float *a,const float *b,size_t n) {
#if (defined(__ARM_NEON) && defined(__aarch64__))
    float32x4_t s0=vdupq_n_f32(0),s1=s0,s2=s0,s3=s0;
    for(size_t i=0;i<n;i+=16) {
        s0=vmlaq_f32(s0,vld1q_f32(a+i),vld1q_f32(b+i));
        s1=vmlaq_f32(s1,vld1q_f32(a+i+4),vld1q_f32(b+i+4));
        s2=vmlaq_f32(s2,vld1q_f32(a+i+8),vld1q_f32(b+i+8));
        s3=vmlaq_f32(s3,vld1q_f32(a+i+12),vld1q_f32(b+i+12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0,s1),vaddq_f32(s2,s3)));
#elif defined(__SSE2__)
    __m128 s=_mm_setzero_ps();
    for(size_t i=0;i<n;i+=4) s=_mm_add_ps(s,_mm_mul_ps(_mm_loadu_ps(a+i),_mm_loadu_ps(b+i)));
    float v[4];_mm_storeu_ps(v,s);return v[0]+v[1]+v[2]+v[3];
#else
    float s=0;for(size_t i=0;i<n;i++) s+=a[i]*b[i];return s;
#endif
}
static int32_t dot_i8_base(const int8_t *a,const int8_t *b,size_t n) {
#if (defined(__ARM_NEON) && defined(__aarch64__))
    int32x4_t s=vdupq_n_s32(0);
    for(size_t i=0;i<n;i+=16) {
        int8x16_t x=vld1q_s8(a+i),y=vld1q_s8(b+i);
        s=vpadalq_s16(s,vmull_s8(vget_low_s8(x),vget_low_s8(y)));
        s=vpadalq_s16(s,vmull_s8(vget_high_s8(x),vget_high_s8(y)));
    }
    return vaddvq_s32(s);
#else
    int32_t s=0;for(size_t i=0;i<n;i++) s+=(int32_t)a[i]*b[i];return s;
#endif
}

#if (defined(__ARM_NEON) && defined(__aarch64__)) && (defined(__GNUC__) || defined(__clang__))
static int has_dotprod(void) {
#if defined(__APPLE__)
    int supported=0;size_t n=sizeof(supported);
    return sysctlbyname("hw.optional.arm.FEAT_DotProd",&supported,&n,NULL,0)==0 && supported;
#elif defined(__linux__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP)&HWCAP_ASIMDDP)!=0;
#else
    return 0;
#endif
}
#if defined(__clang__)
__attribute__((target("dotprod")))
#else
__attribute__((target("arch=armv8.2-a+dotprod")))
#endif
static int32_t dot_i8_dotprod(const int8_t *a,const int8_t *b,size_t n) {
    int32x4_t s=vdupq_n_s32(0);
    for(size_t i=0;i<n;i+=16)s=vdotq_s32(s,vld1q_s8(a+i),vld1q_s8(b+i));
    return vaddvq_s32(s);
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static int32_t dot_i8_avx2(const int8_t *a,const int8_t *b,size_t n) {
    __m256i s=_mm256_setzero_si256();
    for(size_t i=0;i<n;i+=16) {
        __m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+i)));
        __m256i y=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+i)));
        s=_mm256_add_epi32(s,_mm256_madd_epi16(x,y));
    }
    int32_t v[8];_mm256_storeu_si256((__m256i*)v,s);
    int32_t total=0;for(int i=0;i<8;i++)total+=v[i];return total;
}
#endif
const char *pnsn_cpu_features(void) {
#if (defined(__ARM_NEON) && defined(__aarch64__))
    #if defined(__GNUC__) || defined(__clang__)
    return has_dotprod()?"ARM NEON + DOTPROD int8 / NEON fp32":"ARM NEON fp32/int8";
#else
    return "ARM NEON fp32/int8";
#endif
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2")?"x86 AVX2 int8 / SSE fp32":"x86 scalar int8 / SSE fp32";
#else
    return "portable scalar";
#endif
}
int pnsn_cpu_init(pnsn_context *ctx) {
    ctx->simd=0;
#if (defined(__ARM_NEON) && defined(__aarch64__)) && (defined(__GNUC__) || defined(__clang__))
    ctx->simd=has_dotprod()?2:0;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    ctx->simd=__builtin_cpu_supports("avx2")?1:0;
#endif
    if(ctx->model)return 0;
    /* Split even/odd transpose kernels: avoid multiplying stride-inserted zeros. */
    for(int index=0;index<7;index++) for(int p=0;p<2;p++) {
        pnsn_layer *source=&ctx->layers[17+index],*l=&ctx->transpose[index][p];
        *l=*source;l->kernel=(source->kernel+1-(uint32_t)p)/2;
        l->width=((l->ni*l->kernel+15)/16)*16;l->w=NULL;l->q=NULL;
        size_t n=(size_t)l->width*l->no;
        if(ctx->bits==8)l->q=calloc(n,1);else l->w=calloc(n,sizeof(float));
        if(ctx->bits==8?!l->q:!l->w)return pnsn_error("Out of memory packing transpose weights");
        for(size_t o=0;o<l->no;o++) for(size_t k=0;k<l->kernel;k++) {
            size_t src=o*source->width+(k*2+p)*source->ni,dst=o*l->width+k*l->ni;
            if(ctx->bits==8)memcpy(l->q+dst,source->q+src,l->ni);
            else memcpy(l->w+dst,source->w+src,l->ni*sizeof(float));
        }
    }
    return 0;
}
static float quantize(const float *x,int8_t *q,size_t width) {
#if (defined(__ARM_NEON) && defined(__aarch64__))
    float32x4_t maximum=vdupq_n_f32(0);
    for(size_t i=0;i<width;i+=4)maximum=vmaxq_f32(maximum,vabsq_f32(vld1q_f32(x+i)));
    float scale=fmaxf(vmaxvq_f32(maximum)/127,1e-30f);float32x4_t divisor=vdupq_n_f32(scale);
    for(size_t i=0;i<width;i+=16) {
        int32x4_t a=vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x+i),divisor));
        int32x4_t b=vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x+i+4),divisor));
        int32x4_t c=vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x+i+8),divisor));
        int32x4_t d=vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x+i+12),divisor));
        int16x8_t lo=vcombine_s16(vmovn_s32(a),vmovn_s32(b)),hi=vcombine_s16(vmovn_s32(c),vmovn_s32(d));
        vst1q_s8(q+i,vcombine_s8(vmovn_s16(lo),vmovn_s16(hi)));
    }return scale;
#else
    float maximum=0;for(size_t i=0;i<width;i++) {float v=fabsf(x[i]);if(v>maximum)maximum=v;}
    float scale=maximum/127;if(scale<1e-30f)scale=1e-30f;
    for(size_t i=0;i<width;i++) {
#if defined(__SSE2__)
        int v=_mm_cvtss_si32(_mm_set_ss(x[i]/scale));
#else
        int v=(int)nearbyintf(x[i]/scale);
#endif
        q[i]=(int8_t)(v<-127?-127:v>127?127:v);
    }return scale;
#endif
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static float quantize_avx2(const float *x,int8_t *q,size_t width) {
    __m256 maximum=_mm256_setzero_ps(),sign=_mm256_set1_ps(-0.0f);
    for(size_t i=0;i<width;i+=8)maximum=_mm256_max_ps(maximum,_mm256_andnot_ps(sign,_mm256_loadu_ps(x+i)));
    float values[8];_mm256_storeu_ps(values,maximum);float maxv=0;
    for(int i=0;i<8;i++)if(values[i]>maxv)maxv=values[i];
    float scale=maxv/127;if(scale<1e-30f)scale=1e-30f;__m256 divisor=_mm256_set1_ps(scale);
    for(size_t i=0;i<width;i+=8) {
        __m256i v=_mm256_cvtps_epi32(_mm256_div_ps(_mm256_loadu_ps(x+i),divisor));
        __m128i packed=_mm_packs_epi32(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
        packed=_mm_packs_epi16(packed,packed);_mm_storel_epi64((__m128i*)(q+i),packed);
    }return scale;
}
#endif
static void matmul(pnsn_context *ctx,const pnsn_layer *l,const float *x,float *y,size_t rows,int relu,size_t output_stride) {
    float scales[PNSN_TILE];
    int32_t (*dot)(const int8_t*,const int8_t*,size_t)=dot_i8_base;
    float (*quantizer)(const float*,int8_t*,size_t)=quantize;
#if (defined(__ARM_NEON) && defined(__aarch64__)) && (defined(__GNUC__) || defined(__clang__))
    if(ctx->simd==2)dot=dot_i8_dotprod;
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(ctx->simd==1) {dot=dot_i8_avx2;quantizer=quantize_avx2;}
#endif
    if(ctx->bits==8) for(size_t t=0;t<rows;t++) scales[t]=quantizer(x+t*l->width,ctx->quant+t*l->width,l->width);
#if defined(_MSC_VER) && defined(_OPENMP)
    /* MSVC's C OpenMP parser requires the signed counter declared before
       the directive. Rows <= PNSN_TILE (128), validated outputs <= 384. */
    int index;
    const int jobs=(int)(rows*l->no);
#endif
#ifdef _OPENMP
#pragma omp parallel for num_threads(ctx->options.threads) if(ctx->options.threads>1 && rows*l->no>=128)
#endif
#if defined(_MSC_VER) && defined(_OPENMP)
    for(index=0;index<jobs;index++) {
#else
    for(size_t index=0;index<rows*l->no;index++) {
#endif
        size_t t=index/l->no,o=index%l->no;
        float s=l->bias[o];
        if(ctx->bits==8) s+=(float)dot(ctx->quant+t*l->width,l->q+o*l->width,l->width)*scales[t]*l->scale[o];
        else s+=dot_f32(x+t*l->width,l->w+o*l->width,l->width);
        y[t*output_stride+o]=relu?fmaxf(s,0):s;
    }
}
static void transpose_conv(pnsn_context *ctx,int index,const float *x,float *y,size_t t) {
    for(int parity=0;parity<2;parity++) {
        const pnsn_layer *l=&ctx->transpose[index-17][parity];
        size_t first=((size_t)parity+(l->pad&1))&1;
        for(size_t begin=0;begin<t;begin+=PNSN_TILE) {
            size_t batch=t-begin<PNSN_TILE?t-begin:PNSN_TILE;
            memset(ctx->patch,0,batch*l->width*sizeof(float));
            for(size_t row=0;row<batch;row++)for(size_t k=0;k<l->kernel;k++) {
                int64_t source=((int64_t)((begin+row)*2+first+l->pad)-(int64_t)(k*2+parity))/2;
                if(source>=0 && (uint64_t)source<t)memcpy(ctx->patch+row*l->width+k*l->ni,x+(size_t)source*l->ni,l->ni*sizeof(float));
            }
            matmul(ctx,l,ctx->patch,y+(begin*2+first)*l->no,batch,1,l->no*2);
        }
    }
}
static void conv(pnsn_context *ctx,const pnsn_layer *l,const float *x,float *y,size_t t,int relu) {
    size_t rows=l->kind==2?t*2:l->kind==3?t*4:(t+2*l->pad-l->kernel)/l->stride+1;
    for(size_t begin=0;begin<rows;begin+=PNSN_TILE) {
        size_t batch=rows-begin<PNSN_TILE?rows-begin:PNSN_TILE;
        memset(ctx->patch,0,batch*l->width*sizeof(float));
        for(size_t row=0;row<batch;row++) for(size_t k=0;k<l->kernel;k++) {
            int64_t source=(int64_t)(begin+row)+(int64_t)l->pad-(int64_t)k;
            if(l->kind!=2) {
                source=(int64_t)(begin+row)*(l->kind==3?1:l->stride)+(int64_t)k-(int64_t)l->pad;
                if(l->kind==3){if(source<0 || source>=(int64_t)t*4)continue;source/=4;}
            }
            else {if(source%2) continue;source/=2;}
            if(source>=0 && (uint64_t)source<t) memcpy(ctx->patch+row*l->width+k*l->ni,x+(size_t)source*l->ni,l->ni*sizeof(float));
        }
        matmul(ctx,l,ctx->patch,y+begin*l->no,batch,relu,l->no);
    }
}
static float sigmoid(float x) { return 1/(1+expf(-x)); }
static void brnn(pnsn_context *ctx,int base,const float *x,float *out,size_t t) {
    /* Recurrent state stays fp32, including in int8 mode. */
    for(int direction=0;direction<2;direction++) {
        const pnsn_layer *ih=&ctx->layers[base+direction*2],*hh=ih+1;
        float state[96]={0},cell[96]={0},gates[384],recur[384];
        conv(ctx,ih,x,out,t,0); /* out is temporary [T,384] */
        for(size_t step=0;step<t;step++) {
            size_t at=direction?t-1-step:step;
            matmul(ctx,hh,state,recur,1,0,hh->no);
            memcpy(gates,out+at*384,sizeof(gates));
            for(int j=0;j<96;j++) {
                float i=sigmoid(gates[j]+recur[j]),f=sigmoid(gates[96+j]+recur[96+j]);
                float g=tanhf(gates[192+j]+recur[192+j]),o=sigmoid(gates[288+j]+recur[288+j]);
                cell[j]=f*cell[j]+i*g;state[j]=o*tanhf(cell[j]);
                ctx->c[at*192+direction*96+j]=state[j];
            }
        }
    }
    conv(ctx,&ctx->layers[base+4],ctx->c,out,t,0);
}
static void china_unet(pnsn_context *ctx,float *out){
    float *x=ctx->a,*y=ctx->b,*swap;size_t t=ctx->options.window;
    float *skip[4];size_t offset=0;
    for(int i=0;i<6;i++){
        pnsn_layer *l=&ctx->layers[i];conv(ctx,l,x,y,t,1);t=(t+2*l->pad-l->kernel)/l->stride+1;
        if(i>=1 && i<=4){skip[i-1]=ctx->skips+offset;memcpy(skip[i-1],y,t*l->no*sizeof(float));offset+=t*l->no;}
        swap=x;x=y;y=swap;
    }
    for(int i=6;i<10;i++){
        pnsn_layer *l=&ctx->layers[i];conv(ctx,l,x,y,t,1);t*=4;
        for(size_t row=0;row<t;row++){
            memcpy(ctx->c+row*l->no*2,skip[9-i]+row*l->no,l->no*sizeof(float));
            memcpy(ctx->c+row*l->no*2+l->no,y+row*l->no,l->no*sizeof(float));
        }x=ctx->c;
    }
    conv(ctx,&ctx->layers[10],x,y,t,0);
    for(size_t row=0;row<t;row++){
        float maxv=fmaxf(y[row*3],fmaxf(y[row*3+1],y[row*3+2])),sum=0;
        for(int c=0;c<3;c++){out[row*3+c]=expf(y[row*3+c]-maxv);sum+=out[row*3+c];}
        for(int c=0;c<3;c++)out[row*3+c]/=sum;
    }
}
void pnsn_cpu_forward(pnsn_context *ctx,float *out) {
    if(ctx->model){china_unet(ctx,out);return;}
    float *x=ctx->a,*y=ctx->b,*swap;size_t t=ctx->options.window;
    for(int layer=0;layer<7;layer++) {
        pnsn_layer *l=&ctx->layers[layer];conv(ctx,l,x,y,t,1);
        for(size_t j=0;j<t/2;j++) for(size_t c=0;c<l->no;c++) y[j*l->no+c]=fmaxf(y[j*2*l->no+c],y[(j*2+1)*l->no+c]);
        t/=2;swap=x;x=y;y=swap;
    }
    for(int layer=7;layer<=12;layer+=5) {brnn(ctx,layer,x,y,t);swap=x;x=y;y=swap;}
    for(int layer=17;layer<25;layer++) {if(layer!=24){transpose_conv(ctx,layer,x,y,t);t*=2;}else conv(ctx,&ctx->layers[layer],x,y,t,0);swap=x;x=y;y=swap;}
    for(size_t j=0;j<t;j++) {
        float maximum=x[j*5];for(int k=1;k<5;k++)maximum=fmaxf(maximum,x[j*5+k]);
        float sum=0;for(int k=0;k<5;k++){out[j*5+k]=expf(x[j*5+k]-maximum);sum+=out[j*5+k];}
        for(int k=0;k<5;k++)out[j*5+k]/=sum;
    }
}
