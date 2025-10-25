#ifndef __ACC_SETUP_H__
#define __ACC_SETUP_H__

#include "common.h"
#include "host_config.h"
#include "xcl2.h"

typedef struct AccDescriptor {
    cl::CommandQueue q;

    std::vector<cl::CommandQueue> big_gs_queue;
    // std::vector<cl::CommandQueue> little_gs_queue;
    std::vector<cl::CommandQueue> apply_queue;
    cl::CommandQueue hbm_writer_queue;

    int num_big_krnl = BIG_KERNEL_NUM;
    int num_little_krnl = LITTLE_KERNEL_NUM;

    // std::string big_gs_kernel_name = "bigKernel";
    // std::string little_gs_kernel_name = "littleKernel";
    // std::string writer_kernel_name = "hbm_writer";
    // std::string apply_kernel_name = "applyKernel";

    std::vector<cl::Kernel> big_gs_krnls;
    // std::vector<cl::Kernel> little_gs_krnls;
    std::vector<cl::Kernel> apply_krnls;
    cl::Kernel hbm_writer_krnl;

    cl::Context context;

    // 新增
    std::vector<int> big_kernel_hbm_edge_id = BIG_KERNEL_HBM_EDGE_ID;
    std::vector<int> big_kernel_hbm_node_id = BIG_KERNEL_HBM_NODE_ID;

    // std::vector<int> little_kernel_hbm_edge_id = LITTLE_KERNEL_HBM_EDGE_ID;
    // std::vector<int> little_kernel_hbm_node_id = LITTLE_KERNEL_HBM_NODE_ID;

    //
    // std::vector<int> big_kernel_hbm_stop_flag_id =
    // BIG_KERNEL_HBM_STOP_FLAG_ID; std::vector<int>
    // little_kernel_hbm_stop_flag_id = LITTLE_KERNEL_HBM_STOP_FLAG_ID;

} AccDescriptor;

AccDescriptor initAccelerator(std::string xcl_file);

#endif
