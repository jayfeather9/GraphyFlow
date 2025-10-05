#ifndef __FPGA_EXECUTOR_H__
#define __FPGA_EXECUTOR_H__

#include "common.h"
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

// 通用的 FPGA 执行函数。
// 它通过 AlgorithmHost 类来处理所有与具体算法相关的操作。
std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 std::vector<GraphCSR> &graphs,
                                 int start_node,
                                 double &total_kernel_time_sec, int &iter_count,
                                 int device_no, int max_iterations) ;

#endif // __FPGA_EXECUTOR_H__
