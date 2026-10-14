#ifndef SEISMICX_H
#define SEISMICX_H
/* SeismicX C backend API. The pnsn_* ABI remains available for compatibility.
 * Output classes are read from the loaded model: PNSN=5, China UNet=3.
 * Fixed windows: PNSN multiples of 128; China UNet multiples of 256.
 * Historical input [3,N]; realtime input [B,3,L]; probability output [B,L,C]. */
#include "pnsn.h"
typedef pnsn_context sx_context;
typedef pnsn_options sx_options;
typedef pnsn_dtype sx_dtype;
typedef pnsn_pick sx_pick;
#define SX_F32 PNSN_F32
#define SX_F64 PNSN_F64
#define SX_I32 PNSN_I32
#define sx_default_options pnsn_default_options
#define sx_open pnsn_open
#define sx_close pnsn_close
#define sx_last_error pnsn_last_error
#define sx_classes pnsn_classes
#define sx_model_name pnsn_model_name
#define sx_cpu_features pnsn_cpu_features
#define sx_memory_bytes pnsn_memory_bytes
#define sx_predict pnsn_predict
#define sx_predict_batch pnsn_predict_batch
#define sx_predict_batch_normalized pnsn_predict_batch_normalized
#define sx_pick_history pnsn_pick_history
#define sx_pick_batch pnsn_pick_batch
#define sx_nms pnsn_nms
#define sx_batch_device_stats pnsn_batch_device_stats
#endif
