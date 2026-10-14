/* SPDX-License-Identifier: GPL-3.0-only */
#ifndef PNSN_INTERNAL_H
#define PNSN_INTERNAL_H
#include "pnsn.h"
#define PNSN_LAYERS 25
#define PNSN_TILE 32
typedef struct {
    uint32_t kind, ni, no, kernel, stride, pad, width;
    float *scale, *bias, *w;
    int8_t *q;
} pnsn_layer;
struct pnsn_context {
    pnsn_options options;
    int bits, simd, model, classes; /* 0 PNSN v3, 1 China PhaseNet-style UNet */
    pnsn_layer layers[PNSN_LAYERS];
    pnsn_layer transpose[7][2];
    float *a, *b, *c, *patch, *prob, *skips;
    int8_t *quant;
    size_t max_width;
#ifdef PNSN_CUDA
    void *gpu;
#endif
};
#ifdef __cplusplus
extern "C" {
#endif
int pnsn_error(const char *message);
int pnsn_normalize(const void *input, pnsn_dtype dtype, size_t samples,
                   size_t start, size_t window, int diff, float *out);
int pnsn_cpu_init(pnsn_context *ctx);
void pnsn_cpu_forward(pnsn_context *ctx, float *out);
#ifdef PNSN_CUDA
void *pnsn_cuda_open(pnsn_context *ctx);
int pnsn_cuda_forward(void *gpu, const float *input, float *output);
void pnsn_cuda_close(void *gpu);
int pnsn_cuda_forward_batch(void *gpu,const void *input,pnsn_dtype dtype,size_t batch,int normalize,int difference,float *output);
int pnsn_cuda_batch_stats(void *gpu,double *prep,double *network,size_t *bytes);
#endif
#ifdef __cplusplus
}
#endif
#endif
