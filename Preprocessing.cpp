// =============================================================================
// preprocessing.cpp — Vitis HLS Image Preprocessing Kernel
// Target: AMD Kria KV260 / KR260 — DPU Companion Accelerator
// Purpose: Resize + Normalize an input image frame before DPU inference.
//          Offloading these CPU bottlenecks to PL (Programmable Logic) frees
//          the ARM cores for orchestration and UI tasks.
// =============================================================================

#include "preprocessing.h"
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// ---------------------------------------------------------------------------
// Internal fixed-point type for normalized pixel values [-1.0, 1.0]
// ap_fixed<16, 2> gives 14 fractional bits — sufficient for INT8 DPU input.
// ---------------------------------------------------------------------------
typedef ap_fixed<16, 2> fixed_pixel_t;

// ---------------------------------------------------------------------------
// bilinear_interpolate
// Computes one output pixel via bilinear interpolation from a flat src buffer.
// Fully unrolled inner loop — HLS will map this to a small LUT chain.
// ---------------------------------------------------------------------------
static fixed_pixel_t bilinear_interpolate(
    const uint8_t  src[SRC_H * SRC_W * CHANNELS],
    float          src_x,
    float          src_y,
    int            ch)
{
#pragma HLS INLINE

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = (x0 + 1 < SRC_W) ? x0 + 1 : x0;
    int y1 = (y0 + 1 < SRC_H) ? y0 + 1 : y0;

    float dx = src_x - x0;
    float dy = src_y - y0;

    float p00 = src[(y0 * SRC_W + x0) * CHANNELS + ch];
    float p01 = src[(y0 * SRC_W + x1) * CHANNELS + ch];
    float p10 = src[(y1 * SRC_W + x0) * CHANNELS + ch];
    float p11 = src[(y1 * SRC_W + x1) * CHANNELS + ch];

    float interp = p00 * (1 - dx) * (1 - dy)
                 + p01 *      dx  * (1 - dy)
                 + p10 * (1 - dx) *      dy
                 + p11 *      dx  *      dy;

    // Normalize to [-1.0, 1.0]: (pixel / 127.5) - 1.0
    fixed_pixel_t norm = (fixed_pixel_t)(interp / 127.5f - 1.0f);
    return norm;
}

// ---------------------------------------------------------------------------
// preprocess_image  — TOP-LEVEL HLS KERNEL
//
// Ports (AXI4-Lite slave for scalar args, AXI4 master for array args):
//   src_img  : raw uint8 BGR frame  [SRC_H  × SRC_W  × CHANNELS]
//   dst_img  : normalized output    [DST_H  × DST_W  × CHANNELS]  (int8)
//
// Pipeline strategy:
//   - II=1 pipeline on the innermost channel loop (3 channels/cycle)
//   - DATAFLOW pragma enables the resize and normalize stages to overlap
//     in a producer-consumer pipeline across function calls.
// ---------------------------------------------------------------------------
extern "C" void preprocess_image(
    const uint8_t  src_img[SRC_H * SRC_W * CHANNELS],
    int8_t         dst_img[DST_H * DST_W * CHANNELS])
{
// ---- Interface pragmas -----------------------------------------------
#pragma HLS INTERFACE m_axi     port=src_img  offset=slave bundle=gmem0 depth=SRC_DEPTH
#pragma HLS INTERFACE m_axi     port=dst_img  offset=slave bundle=gmem1 depth=DST_DEPTH
#pragma HLS INTERFACE s_axilite port=return

// ---- Array partitioning — allow simultaneous channel reads -----------
#pragma HLS ARRAY_PARTITION variable=src_img cyclic factor=CHANNELS dim=1
#pragma HLS ARRAY_PARTITION variable=dst_img cyclic factor=CHANNELS dim=1

// ---- Dataflow enables the two loop nests to pipeline across calls ----
#pragma HLS DATAFLOW

    // Scale factors (compile-time constants → synthesised as multipliers)
    const float scale_x = (float)SRC_W / (float)DST_W;
    const float scale_y = (float)SRC_H / (float)DST_H;

ROW_LOOP:
    for (int dy = 0; dy < DST_H; dy++) {
COL_LOOP:
        for (int dx = 0; dx < DST_W; dx++) {
#pragma HLS PIPELINE II=1          // Target: one output pixel per clock cycle
#pragma HLS LOOP_FLATTEN off       // Keep loop hierarchy visible to scheduler

            float src_x = (dx + 0.5f) * scale_x - 0.5f;
            float src_y = (dy + 0.5f) * scale_y - 0.5f;

            // Clamp to valid source range
            if (src_x < 0.0f) src_x = 0.0f;
            if (src_y < 0.0f) src_y = 0.0f;
            if (src_x > SRC_W - 1) src_x = (float)(SRC_W - 1);
            if (src_y > SRC_H - 1) src_y = (float)(SRC_H - 1);

CH_LOOP:
            for (int ch = 0; ch < CHANNELS; ch++) {
#pragma HLS UNROLL factor=CHANNELS  // All 3 channels computed in parallel

                fixed_pixel_t norm_val = bilinear_interpolate(
                    src_img, src_x, src_y, ch);

                // Quantize to INT8 for DPU (scale factor = 64, i.e., Q6 fixed)
                int8_t quantized = (int8_t)(norm_val * DPU_QUANT_SCALE);
                dst_img[(dy * DST_W + dx) * CHANNELS + ch] = quantized;
            }
        }
    }
}