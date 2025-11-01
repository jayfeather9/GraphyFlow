#ifndef __ACC_SETUP_H__
#define __ACC_SETUP_H__

#include "common.h"
#include "host_config.h"
#include "xcl2.h"

typedef struct AccDescriptor {
    cl::CommandQueue q;

    std::vector<cl::CommandQueue> big_gs_queue;
    std::vector<cl::CommandQueue> little_gs_queue;
    std::vector<cl::CommandQueue> hbm_writer_little_queue;
    std::vector<cl::CommandQueue> hbm_writer_big_queue;

    int num_big_krnl = BIG_KERNEL_NUM;
    int num_little_krnl = LITTLE_KERNEL_NUM;

    std::vector<cl::Kernel> big_gs_krnls;
    std::vector<cl::Kernel> little_gs_krnls;
    std::vector<cl::Kernel> hbm_writer_little_krnls;
    std::vector<cl::Kernel> hbm_writer_big_krnls;

    cl::Context context;

    // 新增
    std::vector<int> big_kernel_hbm_edge_id = BIG_KERNEL_HBM_EDGE_ID;
    std::vector<int> big_kernel_hbm_node_id = BIG_KERNEL_HBM_NODE_ID;

    std::vector<int> little_kernel_hbm_edge_id = LITTLE_KERNEL_HBM_EDGE_ID;
    std::vector<int> little_kernel_hbm_node_id = LITTLE_KERNEL_HBM_NODE_ID;

} AccDescriptor;

AccDescriptor initAccelerator(std::string xcl_file);

#endif
