#pragma once

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdint.h>

namespace kernel_params {

constexpr int kPeCount = 8;
constexpr int kDoublePeCount = 16;
constexpr int kLogPeCount = 3;
constexpr int kMaxNodeCountLittle = 65536;
constexpr int kMaxNodeCountBig = 524288;
constexpr int kReductionLevels = 4;
constexpr int kSrcBufferSize = 4096;
constexpr int kLogSrcBufferSize = 12;
constexpr int kNodeIdBitwidth = 32;
constexpr int kDistanceBitwidth = 32;
constexpr int kDistanceIntegerPart = 16;
constexpr int kWeightBitwidth = kDistanceBitwidth;
constexpr int kWeightIntegerPart = kDistanceIntegerPart;
constexpr int kOutEndMarkerBitwidth = 4;
constexpr int kAxiBusWidth = 512;
constexpr int kDistancesPerWord = kAxiBusWidth / kDistanceBitwidth;
constexpr int kLogDistancesPerWord = 4;
constexpr int kReduceMemWidth = 64;
constexpr int kInfinityDistance = 16384;

constexpr int kLittleKernelCount = 3;
constexpr int kBigKernelCount = 2;
constexpr int kTotalKernelCount = kLittleKernelCount + kBigKernelCount;

using BusWord = ap_uint<kAxiBusWidth>;
using ReduceWord = ap_uint<kReduceMemWidth>;
using DistancePod = ap_uint<kDistanceBitwidth>;
using DistanceFixed = ap_fixed<kDistanceBitwidth, kDistanceIntegerPart>;
using NodeId = ap_uint<kNodeIdBitwidth>;
using EdgeId = ap_uint<32>;
using CachelineDataPacket = ap_axiu<kAxiBusWidth, 0, 0, 0>;
using WriteBurstPacket = ap_axiu<kAxiBusWidth, 0, 0, 0>;
using LittleRequestPacket = ap_axiu<32, 0, 0, 0>;
using LittleResponsePacket = ap_axiu<kAxiBusWidth, 0, 0, 32>;
using BigRequestPacket = ap_axiu<32, 0, 0, 8>;
using BigResponsePacket = ap_axiu<kAxiBusWidth, 0, 0, 8>;

} // namespace kernel_params
