#pragma once
#include <cstdint>

// ---- Source frame dimensions (e.g., 720p camera) ----
static const int SRC_H    = 720;
static const int SRC_W    = 1280;
static const int CHANNELS = 3;      // BGR

// ---- DPU model input dimensions (e.g., MobileNetV2 224×224) ----
static const int DST_H    = 224;
static const int DST_W    = 224;

// ---- Derived depth values for AXI port depth pragmas ----
static const int SRC_DEPTH = SRC_H * SRC_W * CHANNELS;
static const int DST_DEPTH = DST_H * DST_W * CHANNELS;

// ---- DPU INT8 quantization scale (fixed-point Q6 = multiply by 64) ----
static const float DPU_QUANT_SCALE = 64.0f;

// ---- Kernel entry-point declaration ----
extern "C" void preprocess_image(
    const uint8_t src_img[SRC_H * SRC_W * CHANNELS],
    int8_t        dst_img[DST_H * DST_W * CHANNELS]);