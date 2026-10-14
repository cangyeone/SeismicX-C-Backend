#ifndef PNSN_H
#define PNSN_H
#include <stddef.h>
#include <stdint.h>
#if defined(PNSN_STATIC)
# define PNSN_API
#elif defined(_WIN32)
# ifdef PNSN_BUILD
# define PNSN_API __declspec(dllexport)
# else
# define PNSN_API __declspec(dllimport)
# endif
#else
# define PNSN_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct pnsn_context pnsn_context;
typedef enum { PNSN_F32=0, PNSN_F64=1, PNSN_I32=2 } pnsn_dtype;
typedef struct { int32_t phase; int64_t sample; float confidence; } pnsn_pick;
typedef struct {
    size_t window;       /* PNSN: multiple of 128; China UNet: 256; default 10240 */
    size_t overlap;      /* default 1024 */
    size_t max_candidates; /* hard memory budget; never silently truncate */
    float threshold;
    int64_t min_gap;     /* same-class suppression distance, inclusive */
    int threads;        /* OpenMP workers; default 1 for edge devices */
    int difference;     /* 0 raw, 1 raw + first difference */
} pnsn_options;
/* Each context owns workspace; separate contexts can run concurrently. */
PNSN_API pnsn_options pnsn_default_options(void);
PNSN_API pnsn_context *pnsn_open(const char *model_path, const pnsn_options *options);
PNSN_API const char *pnsn_last_error(void);
PNSN_API void pnsn_close(pnsn_context *ctx);
PNSN_API const char *pnsn_cpu_features(void);
/* CPU context bytes, excluding caller buffers, transient NMS and CUDA device memory. */
/* Model-dependent output classes: PNSN=5, China UNet=3 (Noise,P,S). */
PNSN_API int pnsn_classes(const pnsn_context *ctx);
PNSN_API const char *pnsn_model_name(const pnsn_context *ctx);
PNSN_API size_t pnsn_memory_bytes(const pnsn_context *ctx);
/* Preprocess any raw window, pad short input by its final sample. */
PNSN_API int pnsn_preprocess(const void *input, pnsn_dtype dtype, size_t samples,
                            size_t start, size_t window, int difference, float *output);
/* Interleaved E/N/Z input [samples,3], preprocessing included, output [samples,C].
   C=pnsn_classes(ctx).
   Fixed call requires samples==window. Zero variance becomes zeros. Reject NaN/Inf. */
PNSN_API int pnsn_predict(pnsn_context *ctx, const void *input, pnsn_dtype dtype,
                         size_t samples, float *probabilities);
/* Arbitrary length, bounded per-window workspace; returns time-sorted picks.
   Output remains untouched if capacity is insufficient; *count gives required size. */
PNSN_API int pnsn_pick_trace(pnsn_context *ctx, const void *input, pnsn_dtype dtype,
                           size_t samples, pnsn_pick *output, size_t capacity, size_t *count);
PNSN_API int pnsn_preprocess_channels(const void *input,pnsn_dtype dtype,size_t samples,
                                     size_t start,size_t window,int difference,float *output);
PNSN_API int pnsn_preprocess_batch(const void *input,pnsn_dtype dtype,size_t batch,
                                  size_t length,int difference,float *output);
/* Official history layout [3,N], planar E/N/Z; global sliding-window picks. */
PNSN_API int pnsn_pick_history(pnsn_context *ctx, const void *input, pnsn_dtype dtype,
                             size_t samples, pnsn_pick *output, size_t capacity, size_t *count);
/* Official realtime layout [B,3,L]. L must equal configured window. Output [B,L,C].
   C=pnsn_classes(ctx).
   CUDA executes a real batch; CPU fallback processes each record with SIMD. */
PNSN_API int pnsn_predict_batch(pnsn_context *ctx, const void *input, pnsn_dtype dtype,
                              size_t batch, size_t length, float *probabilities);
/* Expert benchmark path: input [B,3,L] is already normalized float32. */
PNSN_API int pnsn_predict_batch_normalized(pnsn_context *ctx, const float *input,
                                         size_t batch, size_t length, float *probabilities);
/* Independent NMS per record; offsets[B+1] partitions flat output into batches.
   count gives required total; output and offsets unchanged on insufficient capacity. */
PNSN_API int pnsn_pick_batch(pnsn_context *ctx, const void *input, pnsn_dtype dtype,
                           size_t batch, size_t length, pnsn_pick *output, size_t capacity,
                           size_t *offsets, size_t *count);
/* Last CUDA batch event times, excluding host transfers/NMS. CPU returns zeros.
   buffer_bytes counts allocated CUDA tensor buffers/weights, not driver/process RSS. */
PNSN_API int pnsn_batch_device_stats(pnsn_context *ctx, double *preprocess_ms,
                                    double *network_ms, size_t *buffer_bytes);
/* Independent heap NMS: min-heap keyed by (-confidence,time,phase).
   Candidates may be unsorted and should already be threshold-filtered; ties prefer earliest sample. */
PNSN_API int pnsn_nms(const pnsn_pick *candidates, size_t n, int64_t min_gap,
                     pnsn_pick *output, size_t capacity, size_t *count);
#ifdef __cplusplus
}
#endif
#endif
