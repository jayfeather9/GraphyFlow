Project Path: scripts

Source Tree:

```txt
scripts
├── clean.mk
├── help.mk
├── host
│   ├── acc_setup
│   │   ├── acc_setup.cpp
│   │   └── acc_setup.h
│   ├── common.h
│   ├── fpga_executor.cpp
│   ├── fpga_executor.h
│   ├── generated_host.cpp
│   ├── generated_host.h
│   ├── graph_loader.cpp
│   ├── graph_loader.h
│   ├── graph_preprocess
│   │   ├── graph_preprocess.cpp
│   │   └── graph_preprocess.h
│   ├── host.cpp
│   ├── host.mk
│   ├── host_bellman_ford.cpp
│   ├── host_bellman_ford.h
│   ├── host_config.h
│   ├── host_verifier.cpp
│   ├── host_verifier.h
│   ├── xcl2.cpp
│   └── xcl2.h
├── kernel
│   ├── apply_kernel.cpp
│   ├── big_merger.cpp
│   ├── graphyflow_big.cpp
│   ├── graphyflow_big.h
│   ├── graphyflow_little.cpp
│   ├── graphyflow_little.h
│   ├── hbm_writer.cpp
│   ├── kernel.mk
│   ├── little_merger.cpp
│   └── shared_kernel_params.h
├── main.mk
└── utils.mk

```

`scripts/clean.mk`:

```mk
cleanexe:
	-$(RMDIR) $(EXECUTABLE)

clean:
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) .Xil
	-$(RMDIR) *.zip
	-$(RMDIR) *.str
	-$(RMDIR) ./_x
	-$(RMDIR) ./membership.out
	-$(RMDIR) .run
	-$(RMDIR) makefile_gen
	-$(RMDIR) .ipcache
	-$(RMDIR) ./scripts/host/*.o
	-$(RMDIR) ./scripts/host/acc_setup/acc_setup.o
	-$(RMDIR) ./scripts/host/graph_preprocess/graph_preprocess.o

cleanall:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*,*hw*} 
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) .Xil
	-$(RMDIR) *.zip
	-$(RMDIR) *.str
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) ./_x
	-$(RMDIR) ./membership.out
	-$(RMDIR) xclbin*
	-$(RMDIR) .run
	-$(RMDIR) makefile_gen
	-$(RMDIR) .ipcache
	-$(RMDIR) *.csv
	-$(RMDIR) *.protoinst
	-$(RMDIR) ./scripts/host/*.o
	-$(RMDIR) ./scripts/host/acc_setup/acc_setup.o
	-$(RMDIR) ./scripts/host/graph_preprocess/graph_preprocess.o

```

`scripts/help.mk`:

```mk
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""

```

`scripts/host/acc_setup/acc_setup.cpp`:

```cpp
#include "acc_setup.h"

AccDescriptor initAccelerator(const std::string xclbin_path) {
    cl_int err;
    AccDescriptor acc;

    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    //  为选定的设备创建上下文和主命令队列
    OCL_CHECK(err, acc.context =
                       cl::Context(device, nullptr, nullptr, nullptr, &err));
    OCL_CHECK(err, acc.q = cl::CommandQueue(acc.context, device,
                                            CL_QUEUE_PROFILING_ENABLE, &err));

    // 为每个 "big" 内核实例创建一个专用的命令队列 ---
    acc.big_gs_queue.resize(acc.num_big_krnl);
    for (int k = 0; k < acc.num_big_krnl; k++) {
        cl::CommandQueue tmp_q;
        OCL_CHECK(err,
                  tmp_q = cl::CommandQueue(acc.context, device,
                                           CL_QUEUE_PROFILING_ENABLE, &err));
        acc.big_gs_queue[k] = tmp_q;
    }

    // 为每个 "little" 内核实例创建一个专用的命令队列 ---
    acc.little_gs_queue.resize(acc.num_little_krnl);
    for (int k = 0; k < acc.num_little_krnl; k++) {
        cl::CommandQueue tmp_q;
        OCL_CHECK(err,
                  tmp_q = cl::CommandQueue(acc.context, device,
                                           CL_QUEUE_PROFILING_ENABLE, &err));
        acc.little_gs_queue[k] = tmp_q;
    }

    // 为每个 "apply" 内核实例创建一个专用的命令队列 ---
    OCL_CHECK(err, acc.apply_queue = cl::CommandQueue(
                       acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    // 为统一的 hbm_writer 内核创建专用的命令队列 ---
    OCL_CHECK(err, acc.hbm_writer_queue = cl::CommandQueue(
                       acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    std::cout << "Attempting to program device: "
              << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(acc.context, {device}, bins, nullptr, &err);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file!\n";
        exit(EXIT_FAILURE);
    } else {
        std::cout << "Device program successful!\n";

        // 创建 acc.num_big_krnl 个 "graphyflow_big" 内核实例 ---
        for (int i = 0; i < acc.num_big_krnl; i++) {
            std::string cu_id = std::to_string(i + 1);
            // 构造内核名称，格式为 "kernel_name:{instance_name_ID}"
            std::string krnl_name_full = std::string("graphyflow_big:{") +
                                         "graphyflow_big_" + cu_id + "}";

            cl::Kernel tmp_gs_krnl;
            printf("Creating a big kernel [%s] for CU(%d)\n",
                   krnl_name_full.c_str(), i + 1);
            OCL_CHECK(err, tmp_gs_krnl = cl::Kernel(
                               program, krnl_name_full.c_str(), &err));
            acc.big_gs_krnls.push_back(tmp_gs_krnl);
        }

        // 创建 acc.num_little_krnl 个 "graphyflow_little" 内核实例 ---
        for (int i = 0; i < acc.num_little_krnl; i++) {
            std::string cu_id = std::to_string(i + 1);
            std::string krnl_name_full = std::string("graphyflow_little:{") +
                                         "graphyflow_little_" + cu_id + "}";

            cl::Kernel tmp_gs_krnl;
            printf("Creating a little kernel [%s] for CU(%d)\n",
                   krnl_name_full.c_str(), i + 1);
            OCL_CHECK(err, tmp_gs_krnl = cl::Kernel(
                               program, krnl_name_full.c_str(), &err));
            acc.little_gs_krnls.push_back(tmp_gs_krnl);
        }

        // 创建 acc.num_apply_krnl 个 "apply_kernel" 内核实例 ---
        OCL_CHECK(err, acc.apply_krnl = cl::Kernel(
                           program, "apply_kernel:{apply_kernel_1}", &err));

        // 创建统一的 hbm_writer 内核实例 ---
        OCL_CHECK(err, acc.hbm_writer_krnl = cl::Kernel(
                           program, "hbm_writer:{hbm_writer_1}", &err));
    }

    return acc;
}
```

`scripts/host/acc_setup/acc_setup.h`:

```h
#ifndef __ACC_SETUP_H__
#define __ACC_SETUP_H__

#include "common.h"
#include "host_config.h"
#include "xcl2.h"

typedef struct AccDescriptor {
    cl::CommandQueue q;

    std::vector<cl::CommandQueue> big_gs_queue;
    std::vector<cl::CommandQueue> little_gs_queue;
    cl::CommandQueue apply_queue;
    cl::CommandQueue hbm_writer_queue;

    int num_big_krnl = BIG_KERNEL_NUM;
    int num_little_krnl = LITTLE_KERNEL_NUM;

    std::vector<cl::Kernel> big_gs_krnls;
    std::vector<cl::Kernel> little_gs_krnls;
    cl::Kernel apply_krnl;
    cl::Kernel hbm_writer_krnl;

    std::vector<std::vector<cl::Event>> big_kernel_events;
    std::vector<std::vector<cl::Event>> little_kernel_events;
    cl::Event apply_kernel_event;
    cl::Event hbm_writer_event;

    cl::Context context;

    // 新增
    std::vector<int> big_kernel_hbm_edge_id = BIG_KERNEL_HBM_EDGE_ID;
    std::vector<int> big_kernel_hbm_node_id = BIG_KERNEL_HBM_NODE_ID;

    std::vector<int> little_kernel_hbm_edge_id = LITTLE_KERNEL_HBM_EDGE_ID;
    std::vector<int> little_kernel_hbm_node_id = LITTLE_KERNEL_HBM_NODE_ID;

} AccDescriptor;

AccDescriptor initAccelerator(std::string xcl_file);

#endif

```

`scripts/host/common.h`:

```h
#ifndef __COMMON_H__
#define __COMMON_H__

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <ap_fixed.h>
#include <ap_int.h>
#include <stdint.h>

#ifndef __SYNTHESIS__
#include "xcl2.h"
#endif

// --- Customizable Bitwidth Macros ---
// These macros define the bitwidths for core data types.
// They are used by the host for data packing and by the kernel for synthesis.
#define NODE_ID_BITWIDTH 32
#define DISTANCE_BITWIDTH 32
#define DISTANCE_INTEGER_PART                                                  \
    16 // Number of bits for the integer part of distance
#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH
#define WEIGHT_INTEGER_PART                                                    \
    DISTANCE_INTEGER_PART // Number of bits for the integer part of weight
#define OUT_END_MARKER_BITWIDTH 4
#define SRC_BUFFER_SIZE 4096

#ifdef EMULATION
const int LITTLE_MAX_DST = 32;
const int BIG_MAX_DST = 32;
#else
const int LITTLE_MAX_DST = 65536;
const int BIG_MAX_DST = 524288;
#endif

// --- Host-side definition for the AXI bus word ---
#define AXI_BUS_WIDTH 512
typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<DISTANCE_BITWIDTH> ap_fixed_pod_t;
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
typedef ap_fixed<WEIGHT_BITWIDTH, WEIGHT_INTEGER_PART> weight_t;
typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;

// A constant representing infinity for distance initialization
const int INFINITY_DIST = 16384;

// --- Graph Type Definitions ---
typedef uint32_t edge_id_t;
typedef uint32_t node_id_t;
// typedef uint32_t ap_fixed_pod_t;

// Structure to hold the graph in Compressed Sparse Row (CSR) format
struct GraphCSR {
    int num_vertices;
    int num_edges;
    int num_dsts;
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<int> weights;
    // Map from original global vertex ID to compressed local ID
    std::unordered_map<int, int> vtx_map;
    // Map from compressed local ID to original global vertex ID
    std::unordered_map<int, int> vtx_map_rev;
};

#define KERNEL_OUTPUT_BATCH_TYPE KernelOutputBatch
#define BATCH_TYPE edge_des_burst_t
#define EDGE_TYPE edge_t
#define NODE_TYPE node_t

#define PE_NUM 8

#endif // __COMMON_H__
```

`scripts/host/fpga_executor.cpp`:

```cpp

#include "fpga_executor.h"
#include "acc_setup/acc_setup.h"
#include "generated_host.h"
#include "graph_preprocess/graph_preprocess.h"
#include <chrono>
#include <iostream>

#define KERNEL_NAME "graphyflow"

std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 const GraphCSR &graph, int start_node,
                                 double &total_kernel_time_sec,
                                 int &iter_count) {

    // GraphPartiton
    PartitionContainer partition_container = partitionGraph(&graph);

    // init accelerator
    AccDescriptor acc = initAccelerator(xclbin_path);

    AlgorithmHost algo_host(acc);
    algo_host.prepare_data(partition_container, start_node);
    algo_host.setup_buffers(partition_container);
    acc.big_kernel_events.resize(partition_container.num_sparse_partitions);
    acc.little_kernel_events.resize(partition_container.num_dense_partitions);
    for (int i = 0; i < partition_container.num_sparse_partitions; ++i) {
        acc.big_kernel_events[i].resize(acc.num_big_krnl);
    }
    for (int i = 0; i < partition_container.num_dense_partitions; ++i) {
        acc.little_kernel_events[i].resize(acc.num_little_krnl);
    }
    total_kernel_time_sec = 0;
    double current_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices;
    int iter = 0;
    std::cout << "\nStarting FPGA execution..." << std::endl;

    for (iter = 0; iter < max_iterations; ++iter) {
        auto iteration_start = std::chrono::high_resolution_clock::now();

        algo_host.update_data(partition_container);
        algo_host.transfer_data_to_fpga(partition_container);

        std::cout << "--- [Host] Phase 3: Enqueuing kernel tasks ---"
                  << std::endl;

        algo_host.execute_kernel_iteration(partition_container);
        auto kernel_enqueue_start = std::chrono::high_resolution_clock::now();

        // Wait for all kernels to finish
        for (auto &q : acc.big_gs_queue)
            q.finish();
        // auto big_finish = std::chrono::high_resolution_clock::now();
        for (auto &q : acc.little_gs_queue)
            q.finish();
        acc.apply_queue.finish();
        acc.hbm_writer_queue.finish();

        auto kernel_finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> end_to_end_time =
            kernel_finish - kernel_enqueue_start;

        algo_host.transfer_data_from_fpga();

        // iv. 性能统计
        int cnt = 0;
        int partition_cnt = 0;
        for (auto &event_vec : acc.big_kernel_events) {
            for (auto &event : event_vec) {
                unsigned long start = 0, end = 0;
                event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
                event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
                double iteration_time_ns = end - start;
                current_kernel_time_sec = std::max(current_kernel_time_sec,
                                                   iteration_time_ns * 1.0e-9);
                double mteps = (double)partition_container.SPs[cnt].num_edges /
                               (iteration_time_ns * 1.0e-9) / 1.0e6;

                std::cout << "FPGA Iteration " << iter << ": "
                          << "Sparse Partition " << partition_cnt << ", "
                          << "Big Kernel " << cnt++ << ", "
                          << "Time = " << (iteration_time_ns * 1.0e-6)
                          << " ms, "
                          << "Throughput = " << mteps << " MTEPS" << std::endl;
            }
            partition_cnt++;
        }
        cnt = 0;
        partition_cnt = 0;
        for (auto &event_vec : acc.little_kernel_events) {
            for (auto &event : event_vec) {
                unsigned long start = 0, end = 0;
                event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
                event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
                double iteration_time_ns = end - start;
                current_kernel_time_sec = std::max(current_kernel_time_sec,
                                                   iteration_time_ns * 1.0e-9);
                double mteps = (double)partition_container.DPs[cnt].num_edges /
                               (iteration_time_ns * 1.0e-9) / 1.0e6;

                std::cout << "FPGA Iteration " << iter << ": "
                          << "Dense Partition " << partition_cnt << ", "
                          << "Little Kernel " << cnt++ << ", "
                          << "Time = " << (iteration_time_ns * 1.0e-6)
                          << " ms, "
                          << "Throughput = " << mteps << " MTEPS" << std::endl;
            }
            partition_cnt++;
        }

        {
            auto event = acc.apply_kernel_event;
            unsigned long start = 0, end = 0;
            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            double iteration_time_ns = end - start;
            current_kernel_time_sec =
                std::max(current_kernel_time_sec, iteration_time_ns * 1.0e-9);
            double mteps =
                (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;

            std::cout << "FPGA Iteration " << iter << ": "
                      << "Apply Kernel, "
                      << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                      << std::endl;
        }

        {
            auto event = acc.hbm_writer_event;
            unsigned long start = 0, end = 0;
            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            double iteration_time_ns = end - start;
            current_kernel_time_sec =
                std::max(current_kernel_time_sec, iteration_time_ns * 1.0e-9);

            std::cout << "FPGA Iteration " << iter << ": "
                      << "HBM Writer Kernel Time = "
                      << (iteration_time_ns * 1.0e-6) << " ms" << std::endl;
        }

        double iteration_time_ns = current_kernel_time_sec * 1.0e9;
        total_kernel_time_sec += end_to_end_time.count();
        current_kernel_time_sec = 0;
        double mteps =
            (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;

        std::cout << "FPGA Iteration " << iter << ": "
                  << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                  << "Throughput = " << mteps << " MTEPS" << std::endl;

        // Print end-to-end timing for the iteration
        std::cout << "FPGA Iteration " << iter
                  << " End-to-End Time: " << (end_to_end_time.count() * 1000.0)
                  << " ms" << " Throughput = "
                  << (double)graph.num_edges / end_to_end_time.count() / 1.0e6
                  << " MTEPS" << std::endl;

        // v. 检查是否收敛。如果未收敛，此函数会更新 partition_container
        // 为下次迭代做准备
        if (algo_host.check_convergence_and_update(partition_container)) {
            std::cout << "FPGA computation converged after " << iter + 1
                      << " iteration(s)." << std::endl;
            break;
        }
    }

    iter_count = iter + 1;

    const std::vector<int> &final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;

    return final_results;
}

```

`scripts/host/fpga_executor.h`:

```h
#ifndef __FPGA_EXECUTOR_H__
#define __FPGA_EXECUTOR_H__

#include "common.h"
#include <vector>

// 通用的 FPGA 执行函数。
// 它通过 AlgorithmHost 类来处理所有与具体算法相关的操作。
std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 const GraphCSR &graph, int start_node,
                                 double &total_kernel_time_sec,
                                 int &iter_count);

#endif // __FPGA_EXECUTOR_H__

```

`scripts/host/generated_host.cpp`:

```cpp
#include "generated_host.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

AlgorithmHost::AlgorithmHost(AccDescriptor &acc) : acc(acc) {}

void AlgorithmHost::prepare_data(const PartitionContainer &container,
                                 int start_node) {
    std::cout << "--- [Host] Phase 0: Preparing data structures ---"
              << std::endl;

    auto start_time = std::chrono::system_clock::now();
    auto current_time = start_time;

    // 1. Initialize algorithm state
    m_num_vertices = container.num_graph_vertices;
    h_distances.assign(m_num_vertices, distance_t(INFINITY_DIST));
    if (start_node < m_num_vertices) {
        h_distances[start_node] = 0;
    }

    // 2. Prepare host-side input buffers for each pipeline
    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    dense_buffers.resize(container.num_dense_partitions);
    sparse_buffers.resize(container.num_sparse_partitions);

    for (int i = 0; i < container.num_dense_partitions; ++i) {
        dense_buffers[i].pipelines.resize(LITTLE_KERNEL_NUM);
        dense_buffers[i].node_prop_offset = 0;
        dense_buffers[i].dst_prop_offset = 0;
    }
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        sparse_buffers[i].pipelines.resize(BIG_KERNEL_NUM);
        sparse_buffers[i].node_prop_offset = 0;
        sparse_buffers[i].dst_prop_offset = 0;
    }

    size_t sparse_node_offset = 0;
    size_t sparse_dst_offset = 0;
    size_t dense_node_offset = 0;
    size_t dense_src_buf_offset = 0;
    size_t dense_dst_offset = 0;

    // --- 2.1: Prepare BIG partition data (shared node props, separate edge
    // props per pipeline) ---
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        const auto &big_partition = container.SPs[i];

        // Pack node distances ONCE for the big partition (shared)
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t word_number =
                (big_partition.num_vertices + dist_per_word - 1) /
                dist_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < big_partition.num_vertices; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }

                int global_id = big_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                // printf("Putting Data: SP No.%d, big pipe No.%d, local_id %d,
                // global_id %d, dist %.3f\n",
                //        i, 0,  j, global_id, (float)dist_val);

                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            size_t node_word_count =
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;
            sparse_buffers[i].packed_node_props.resize(node_word_count, 0);
            std::memcpy(sparse_buffers[i].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
            sparse_buffers[i].node_prop_offset = sparse_node_offset;
            sparse_node_offset += node_word_count;

            const size_t dst_word_number =
                (big_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < big_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = big_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            size_t dst_word_count =
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;
            sparse_buffers[i].packed_dst_props.resize(dst_word_count, 0);
            std::memcpy(sparse_buffers[i].packed_dst_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
            sparse_buffers[i].dst_prop_offset = sparse_dst_offset;
            sparse_dst_offset += dst_word_count;
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared shared big node props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;

        // Pack edge properties for EACH big pipeline
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            const auto &pipeline_edges = big_partition.pipeline_edges[pip];
            // Pack this pipeline's edge properties
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number =
                (pipeline_edges.num_edges + edges_per_word - 1) /
                edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            std::cout << "[Host] Packing edge props for SP No." << i
                      << ", big pipe No." << pip << ", total edges "
                      << pipeline_edges.num_edges << std::endl;

            // Iterate through vertices, then their edges
            for (int v = 0; v < big_partition.num_vertices; ++v) {
                node_id_t src_id = v;
                for (int edge_idx = pipeline_edges.offsets[v];
                     edge_idx < pipeline_edges.offsets[v + 1]; ++edge_idx) {
                    if ((temp_byte_buffer.size() % bytes_per_word) +
                            bytes_per_edge >
                        bytes_per_word) {
                        size_t padding_needed =
                            bytes_per_word -
                            (temp_byte_buffer.size() % bytes_per_word);
                        temp_byte_buffer.insert(temp_byte_buffer.end(),
                                                padding_needed, 0);
                    }

                    char edge_bytes[bytes_per_edge];
                    uint32_t dest_id = pipeline_edges.columns[edge_idx];

                    std::cout <<
                        "[Host]   Edge " << edge_idx << ": src_id " << src_id
                        << ", dest_id " << dest_id << std::endl;

                    // Pack dst_id (first NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[b] = (dest_id >> (8 * b)) & 0xFF;
                    }

                    // Pack src_id (next NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[(NODE_ID_BITWIDTH / 8) + b] =
                            (src_id >> (8 * b)) & 0xFF;
                    }

                    temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                            edge_bytes + bytes_per_edge);
                }
            }
            sparse_buffers[i].pipelines[pip].packed_edge_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(
                sparse_buffers[i].pipelines[pip].packed_edge_props.data(),
                temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared " << BIG_KERNEL_NUM
            << " big pipeline edge props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;
    }

    // --- 2.2: Prepare LITTLE partition data (shared node props, separate edge
    // props per pipeline) ---
    for (int i = 0; i < container.num_dense_partitions; ++i) {
        const auto &little_partition = container.DPs[i];

        // Pack node distances ONCE for the little partition (shared)
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t word_number =
                (little_partition.num_vertices + dist_per_word - 1) /
                dist_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < little_partition.num_vertices; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = little_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                // printf("Putting Data: DP No.%d, little pipe No.%d, local_id
                // %d, global_id %d, dist %.3f\n",
                //        i, 0,  j, global_id, (float)dist_val);
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            // pack temp_byte_buffer to SRC_BUFFER_SIZE nodes
            size_t buffer_size = SRC_BUFFER_SIZE * bytes_per_dist;
            if (temp_byte_buffer.size() % buffer_size != 0) {
                size_t padding_needed =
                    buffer_size - (temp_byte_buffer.size() % buffer_size);
                temp_byte_buffer.insert(temp_byte_buffer.end(), padding_needed,
                                        0);
            }
            size_t src_buf_cnt = temp_byte_buffer.size() / buffer_size;

            size_t node_word_count =
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;
            dense_buffers[i].packed_node_props.resize(node_word_count, 0);
            std::memcpy(dense_buffers[i].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
            dense_buffers[i].node_prop_offset = dense_node_offset;
            dense_node_offset += node_word_count;
            dense_buffers[i].src_buf_offset = dense_src_buf_offset;
            dense_src_buf_offset += src_buf_cnt;

            const size_t dst_word_number =
                (little_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < little_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = little_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            size_t dst_word_count =
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;
            dense_buffers[i].packed_dst_props.resize(dst_word_count, 0);
            std::memcpy(dense_buffers[i].packed_dst_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
            dense_buffers[i].dst_prop_offset = dense_dst_offset;
            dense_dst_offset += dst_word_count;
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared shared little node props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;

        // Pack edge properties for EACH little pipeline
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            const auto &pipeline_edges = little_partition.pipeline_edges[pip];

            // Pack this pipeline's edge properties
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number =
                (pipeline_edges.num_edges + edges_per_word - 1) /
                edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            // Iterate through vertices, then their edges
            for (int v = 0; v < little_partition.num_vertices; ++v) {
                node_id_t src_id = v;
                for (int edge_idx = pipeline_edges.offsets[v];
                     edge_idx < pipeline_edges.offsets[v + 1]; ++edge_idx) {
                    if ((temp_byte_buffer.size() % bytes_per_word) +
                            bytes_per_edge >
                        bytes_per_word) {
                        size_t padding_needed =
                            bytes_per_word -
                            (temp_byte_buffer.size() % bytes_per_word);
                        temp_byte_buffer.insert(temp_byte_buffer.end(),
                                                padding_needed, 0);
                    }

                    char edge_bytes[bytes_per_edge];
                    uint32_t dest_id = pipeline_edges.columns[edge_idx];

                    // Pack dst_id (first NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[b] = (dest_id >> (8 * b)) & 0xFF;
                    }

                    // Pack src_id (next NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[(NODE_ID_BITWIDTH / 8) + b] =
                            (src_id >> (8 * b)) & 0xFF;
                    }

                    temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                            edge_bytes + bytes_per_edge);
                }
            }
            dense_buffers[i].pipelines[pip].packed_edge_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(
                dense_buffers[i].pipelines[pip].packed_edge_props.data(),
                temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared " << LITTLE_KERNEL_NUM
            << " little pipeline edge props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;
    }

    // apply kernel node props should be first each little partition's dst
    // props, then each big partition's dst props
    size_t total_little_dst_words = dense_dst_offset;
    size_t total_big_dst_words = sparse_dst_offset;
    size_t total_little_node_words = dense_node_offset;
    size_t total_big_node_words = sparse_node_offset;

    apply_kernel_node_props.assign(total_little_dst_words + total_big_dst_words,
                                   0);
    for (int i = 0; i < container.num_dense_partitions; ++i) {
        std::copy(dense_buffers[i].packed_dst_props.begin(),
                  dense_buffers[i].packed_dst_props.end(),
                  apply_kernel_node_props.begin() +
                      dense_buffers[i].dst_prop_offset);
    }
    big_dst_offset = total_little_dst_words;
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        std::copy(sparse_buffers[i].packed_dst_props.begin(),
                  sparse_buffers[i].packed_dst_props.end(),
                  apply_kernel_node_props.begin() + big_dst_offset +
                      sparse_buffers[i].dst_prop_offset);
    }

    writer_kernel_node_props.resize(LITTLE_KERNEL_NUM + BIG_KERNEL_NUM);
    for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
        writer_kernel_node_props[pip].assign(total_little_node_words, 0);
    }
    for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
        writer_kernel_node_props[pip + LITTLE_KERNEL_NUM].assign(
            total_big_node_words, 0);
    }

    for (int i = 0; i < container.num_dense_partitions; ++i) {
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            std::copy(dense_buffers[i].packed_node_props.begin(),
                      dense_buffers[i].packed_node_props.end(),
                      writer_kernel_node_props[pip].begin() +
                          dense_buffers[i].node_prop_offset);
        }
    }
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            std::copy(
                sparse_buffers[i].packed_node_props.begin(),
                sparse_buffers[i].packed_node_props.end(),
                writer_kernel_node_props[pip + LITTLE_KERNEL_NUM].begin() +
                    sparse_buffers[i].node_prop_offset);
            printf("SP No.%d, big pipe No.%d, node_prop_offset %d\n", i, pip,
                   sparse_buffers[i].node_prop_offset);
        }
    }
    current_time = std::chrono::system_clock::now();
    std::cout
        << "--- [Host] Phase 0: Prepared apply kernel node props ("
        << std::chrono::duration<double>(current_time - start_time).count()
        << " sec) ---" << std::endl;
}

// --- PHASE 1: BUFFER SETUP ---
// MODIFIED: Create separate edge buffers for each pipeline, but share node
// buffers within partition
void AlgorithmHost::setup_buffers(const PartitionContainer &container) {
    cl_int err;
    std::cout
        << "--- [Host] Phase 1: Setting up HBM buffers for all pipelines ---"
        << std::endl;

    // 1.1: Clear old buffer handles and resize host-side result vectors
    writer_kernel_node_prop_buffers.clear();

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    size_t dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
    size_t total_output_words = apply_kernel_node_props.size();

    // --- 1.2: Setup buffers for LITTLE pipelines ---
    for (int i = 0; i < container.num_dense_partitions; ++i) {
        const auto &little_partition = container.DPs[i];

        // Create edge buffers for each little pipeline
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            cl::Buffer edge_props_buf;

            cl_mem_ext_ptr_t hbm_ext_edge;
            hbm_ext_edge.flags =
                XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_edge_id[pip];
            hbm_ext_edge.obj =
                dense_buffers[i].pipelines[pip].packed_edge_props.data();
            hbm_ext_edge.param = 0;

            size_t num_edge_words =
                dense_buffers[i].pipelines[pip].packed_edge_props.size();
            printf("LITTLE SP No.%d, pipe No.%d, edge words %d\n", i, pip,
                   (int)num_edge_words);
            OCL_CHECK(err, edge_props_buf = cl::Buffer(
                               acc.context,
                               CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                   CL_MEM_USE_HOST_PTR,
                               num_edge_words * bytes_per_word, &hbm_ext_edge,
                               &err));

            dense_buffers[i].pipelines[pip].edge_props_buffer = edge_props_buf;
        }

        std::cout << "  Created " << LITTLE_KERNEL_NUM
                  << " little pipeline edge buffers." << std::endl;
    }

    // --- 1.3: Setup buffers for BIG pipelines ---
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        const auto &big_partition = container.SPs[i];

        // Create edge buffers for each big pipeline
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            cl::Buffer edge_props_buf;

            cl_mem_ext_ptr_t hbm_ext_edge;
            hbm_ext_edge.flags =
                XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_edge_id[pip];
            hbm_ext_edge.obj =
                sparse_buffers[i].pipelines[pip].packed_edge_props.data();
            hbm_ext_edge.param = 0;

            size_t num_edge_words =
                sparse_buffers[i].pipelines[pip].packed_edge_props.size();
            OCL_CHECK(err, edge_props_buf = cl::Buffer(
                               acc.context,
                               CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                   CL_MEM_USE_HOST_PTR,
                               num_edge_words * bytes_per_word, &hbm_ext_edge,
                               &err));

            sparse_buffers[i].pipelines[pip].edge_props_buffer = edge_props_buf;
        }

        std::cout << "  Created " << BIG_KERNEL_NUM
                  << " big pipeline edge buffers." << std::endl;
    }

    // --- 1.4: Setup shared node property buffers for hbm_writer (14 total: 11
    // little + 3 big) --- Create 11 little node prop buffers for hbm_writer
    for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
        cl_mem_ext_ptr_t hbm_ext_node;
        hbm_ext_node.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_node_id[pip];
        hbm_ext_node.obj = writer_kernel_node_props[pip].data();
        hbm_ext_node.param = 0;

        size_t num_node_words = writer_kernel_node_props[pip].size();
        cl::Buffer node_buf;
        OCL_CHECK(err, node_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_node_words * bytes_per_word,
                                      &hbm_ext_node, &err));
        writer_kernel_node_prop_buffers.push_back(node_buf);
    }

    // Create 3 big node prop buffers for hbm_writer
    for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
        cl_mem_ext_ptr_t hbm_ext_node;
        hbm_ext_node.flags = XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_node_id[pip];
        hbm_ext_node.obj =
            writer_kernel_node_props[pip + LITTLE_KERNEL_NUM].data();
        hbm_ext_node.param = 0;

        size_t num_node_words =
            writer_kernel_node_props[pip + LITTLE_KERNEL_NUM].size();
        cl::Buffer node_buf;
        OCL_CHECK(err, node_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_node_words * bytes_per_word,
                                      &hbm_ext_node, &err));
        writer_kernel_node_prop_buffers.push_back(node_buf);
    }

    // --- 1.5: Setup unified output buffer ---
    cl_mem_ext_ptr_t hbm_ext_output;
    hbm_ext_output.flags =
        XCL_MEM_TOPOLOGY |
        acc.little_kernel_hbm_node_id[0]; // Use first HBM bank
    hbm_ext_output.obj = nullptr;
    hbm_ext_output.param = 0;

    writer_kernel_host_outputs.resize(total_output_words, 0);
    OCL_CHECK(err,
              writer_kernel_output_buffer = cl::Buffer(
                  acc.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                  total_output_words * bytes_per_word, &hbm_ext_output, &err));

    // --- 1.6: Setup apply_kernel node prop buffer ---
    cl_mem_ext_ptr_t hbm_ext_apply;
    hbm_ext_apply.flags = XCL_MEM_TOPOLOGY | 30;
    hbm_ext_apply.obj = apply_kernel_node_props.data();
    hbm_ext_apply.param = 0;

    size_t apply_node_words = apply_kernel_node_props.size();
    OCL_CHECK(err, apply_kernel_node_prop_buffer =
                       cl::Buffer(acc.context,
                                  CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX |
                                      CL_MEM_USE_HOST_PTR,
                                  apply_node_words * bytes_per_word,
                                  &hbm_ext_apply, &err));

    std::cout << "[SUCCESS] HBM buffers created: " << LITTLE_KERNEL_NUM
              << " little + " << BIG_KERNEL_NUM << " big pipelines, "
              << "total output size: " << total_output_words << " words."
              << std::endl;
}

void AlgorithmHost::update_data(const PartitionContainer &container) {
    std::cout
        << "--- [Host] Phase 2.1: Updating host-side data for new iteration ---"
        << std::endl;

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    const size_t dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;

    // Update BIG partition node distances (shared across all big pipelines)
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        const auto &big_partition = container.SPs[i];
        const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
        std::vector<char> temp_byte_buffer;
        temp_byte_buffer.clear();

        for (int j = 0; j < big_partition.num_vertices; ++j) {
            if ((temp_byte_buffer.size() % bytes_per_word) + bytes_per_dist >
                bytes_per_word) {
                size_t padding_needed =
                    bytes_per_word - (temp_byte_buffer.size() % bytes_per_word);
                temp_byte_buffer.insert(temp_byte_buffer.end(), padding_needed,
                                        0);
            }

            int global_id = big_partition.vtx_map_rev.at(j);
            distance_t dist_val = h_distances[global_id];

            // printf("Putting Data: SP No.%d, local_id %d, global_id %d, dist
            // %.3f\n",
            //        i,  j, global_id, (float)dist_val);

            const char *data_ptr = reinterpret_cast<const char *>(&dist_val);
            temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                    data_ptr + bytes_per_dist);
        }

        // Update all big pipeline buffers with same node data
        size_t num_words =
            (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;

        sparse_buffers[i].packed_node_props.resize(num_words, 0);
        std::memcpy(sparse_buffers[i].packed_node_props.data(),
                    temp_byte_buffer.data(), temp_byte_buffer.size());
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            std::copy(
                sparse_buffers[i].packed_node_props.begin(),
                sparse_buffers[i].packed_node_props.end(),
                writer_kernel_node_props[pip + LITTLE_KERNEL_NUM].begin() +
                    sparse_buffers[i].node_prop_offset);
        }

        // update big_dst_node_props for apply kernel
        {
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t dst_word_number =
                (big_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < big_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = big_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            sparse_buffers[i].packed_dst_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(sparse_buffers[i].packed_dst_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
            std::copy(sparse_buffers[i].packed_dst_props.begin(),
                      sparse_buffers[i].packed_dst_props.end(),
                      apply_kernel_node_props.begin() + big_dst_offset +
                          sparse_buffers[i].dst_prop_offset);
        }
    }

    // Update LITTLE partition node distances (shared across all little
    // pipelines)
    for (int i = 0; i < container.num_dense_partitions; ++i) {
        const auto &little_partition = container.DPs[i];
        const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
        std::vector<char> temp_byte_buffer;
        temp_byte_buffer.clear();

        for (int j = 0; j < little_partition.num_vertices; ++j) {
            if ((temp_byte_buffer.size() % bytes_per_word) + bytes_per_dist >
                bytes_per_word) {
                size_t padding_needed =
                    bytes_per_word - (temp_byte_buffer.size() % bytes_per_word);
                temp_byte_buffer.insert(temp_byte_buffer.end(), padding_needed,
                                        0);
            }
            int global_id = little_partition.vtx_map_rev.at(j);
            distance_t dist_val = h_distances[global_id];
            // printf("Putting Data: DP No.%d, local_id %d, global_id %d, dist
            // %.3f\n",
            //        i,  j, global_id, (float)dist_val);
            const char *data_ptr = reinterpret_cast<const char *>(&dist_val);
            temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                    data_ptr + bytes_per_dist);
        }

        // Update all little pipeline buffers with same node data
        size_t num_words =
            (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;

        dense_buffers[i].packed_node_props.resize(num_words, 0);
        std::memcpy(dense_buffers[i].packed_node_props.data(),
                    temp_byte_buffer.data(), temp_byte_buffer.size());
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            std::copy(dense_buffers[i].packed_node_props.begin(),
                      dense_buffers[i].packed_node_props.end(),
                      writer_kernel_node_props[pip].begin() +
                          dense_buffers[i].node_prop_offset);
        }

        // update little_dst_node_props for apply kernel
        {
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t dst_word_number =
                (little_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < little_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = little_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            dense_buffers[i].packed_dst_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(dense_buffers[i].packed_dst_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
            std::copy(dense_buffers[i].packed_dst_props.begin(),
                      dense_buffers[i].packed_dst_props.end(),
                      apply_kernel_node_props.begin() +
                          dense_buffers[i].dst_prop_offset);
        }
    }

    /*
    // Print writer kernel node prop data
    for (size_t pip = 0; pip < 20; pip += 12) {
        const auto &node_props = writer_kernel_node_props[pip];
        bool is_dense = (pip < LITTLE_KERNEL_NUM);
        std::cout << "Pipeline " << pip << std::endl;
        if (is_dense) {
            std::cout << "  (Dense partition)" << std::endl;
        } else {
            std::cout << "  (Sparse partition)" << std::endl;
        }
        size_t partition_cnt = 0;
        size_t nxt_offset = 0;
        size_t cur_valid_node_num =
            is_dense ? dense_buffers[partition_cnt].node_prop_offset +
                           container.DPs[partition_cnt].num_vertices
                     : sparse_buffers[partition_cnt].node_prop_offset +
                           container.SPs[partition_cnt].num_vertices;

        for (size_t word_idx = 0; word_idx < node_props.size(); ++word_idx) {
            const bus_word_t &word = node_props[word_idx];
            std::cout << "Pipeline " << pip << ", Word " << word_idx << ": ";

            for (size_t data_idx = 0; data_idx < dists_per_word; ++data_idx) {
                distance_t dist =
                    reinterpret_cast<const distance_t *>(&word)[data_idx];
                float dist_float = static_cast<float>(dist);
                std::cout << dist_float;
                size_t cur_idx = word_idx * dists_per_word + data_idx;

                if (cur_idx >= nxt_offset && cur_idx < cur_valid_node_num) {
                    std::cout << " (valid) ";
                } else {
                    if (cur_idx == cur_valid_node_num) {
                        partition_cnt++;
                        if (is_dense) {
                            nxt_offset =
                                dense_buffers[partition_cnt].node_prop_offset *
                                dists_per_word;
                            cur_valid_node_num =
                                nxt_offset +
                                container.DPs[partition_cnt].num_vertices;
                        } else {
                            nxt_offset =
                                sparse_buffers[partition_cnt].node_prop_offset *
                                dists_per_word;
                            cur_valid_node_num =
                                nxt_offset +
                                container.SPs[partition_cnt].num_vertices;
                        }
                    }
                    std::cout << " (invalid) ";
                }
            }
            std::cout << std::endl;
        }
    }
    */

    std::cout << "[SUCCESS] Host-side data updated for new iteration."
              << std::endl;
}

void AlgorithmHost::transfer_data_to_fpga(const PartitionContainer &container) {
    cl_int err;
    std::cout << "--- [Host] Phase 2.2: Transferring data to FPGA HBM ---"
              << std::endl;

    // Transfer edge buffers for all big pipelines across all sparse partitions
    for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
        for (int part = 0; part < container.num_sparse_partitions; ++part) {
            OCL_CHECK(
                err,
                err = acc.big_gs_queue[pip].enqueueMigrateMemObjects(
                    {sparse_buffers[part].pipelines[pip].edge_props_buffer},
                    0 /* 0 means from host*/));
        }
    }

    // Transfer edge buffers for all little pipelines across all dense
    // partitions
    for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
        for (int part = 0; part < container.num_dense_partitions; ++part) {
            OCL_CHECK(
                err, err = acc.little_gs_queue[pip].enqueueMigrateMemObjects(
                         {dense_buffers[part].pipelines[pip].edge_props_buffer},
                         0 /* 0 means from host*/));
        }
    }

    // Transfer apply_kernel node buffer
    OCL_CHECK(err,
              err = acc.apply_queue.enqueueMigrateMemObjects(
                  {apply_kernel_node_prop_buffer}, 0 /* 0 means from host*/));

    // Transfer hbm_writer node prop buffers
    for (size_t i = 0; i < writer_kernel_node_prop_buffers.size(); ++i) {
        OCL_CHECK(err, err = acc.hbm_writer_queue.enqueueMigrateMemObjects(
                           {writer_kernel_node_prop_buffers[i]},
                           0 /* 0 means from host*/));
    }

    // Wait for all transfers to complete
    for (auto &q : acc.big_gs_queue)
        q.finish();
    for (auto &q : acc.little_gs_queue)
        q.finish();
    acc.apply_queue.finish();
    acc.hbm_writer_queue.finish();

    std::cout
        << "[SUCCESS] All data packed and transferred for current iteration."
        << std::endl;
}

// --- PHASE 3: KERNEL EXECUTION ---
// MODIFIED: Kernel arguments are updated to match the new kernel signature.
void AlgorithmHost::execute_kernel_iteration(
    const PartitionContainer &container) {
    cl_int err;
    // std::cout << "--- [Host] Phase 3: Enqueuing kernel tasks ---" <<
    // std::endl;

    auto enqueue_start = std::chrono::high_resolution_clock::now();

    // uint32_t byte_per_word = AXI_BUS_WIDTH / 8;
    // uint32_t dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
    uint32_t little_dst_word_num = 0, big_dst_word_num = 0;

    for (int i = 0; i < container.num_dense_partitions; ++i) {
        little_dst_word_num += dense_buffers[i].packed_dst_props.size();
    }
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        big_dst_word_num += sparse_buffers[i].packed_dst_props.size();
    }

    // 3.1: Enqueue BIG gs kernels (one per pipeline, each with different edges,
    // same nodes)
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            auto &kernel = acc.big_gs_krnls[pip];
            auto &buffer = sparse_buffers[i].pipelines[pip].edge_props_buffer;
            uint32_t pip_num_edges =
                container.SPs[i].pipeline_edges[pip].num_edges;
            uint32_t big_num_vertices = container.SPs[i].num_vertices;
            uint32_t big_num_dsts = container.SPs[i].num_dsts;
            uint32_t memory_offset = sparse_buffers[i].node_prop_offset;

            int arg_idx = 0;
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffer));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, big_num_vertices));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, pip_num_edges));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, big_num_dsts));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, memory_offset));

            printf(
                "Enqueue Big GS Kernel: SP No.%d, big pipe No.%d, num_vertices "
                "%d, num_edges %d, num_dsts %d, memory_offset %d\n",
                i, pip, big_num_vertices, pip_num_edges, big_num_dsts,
                memory_offset);

            OCL_CHECK(err,
                      err = acc.big_gs_queue[pip].enqueueTask(
                          kernel, nullptr, &acc.big_kernel_events[i][pip]));
        }
    }

    // 3.2: Enqueue LITTLE gs kernels (one per pipeline, each with different
    // edges, same nodes)
    for (int i = 0; i < container.num_dense_partitions; ++i) {
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            auto &kernel = acc.little_gs_krnls[pip];
            auto &buffer = dense_buffers[i].pipelines[pip].edge_props_buffer;
            uint32_t pip_num_edges =
                container.DPs[i].pipeline_edges[pip].num_edges;
            uint32_t little_num_vertices = container.DPs[i].num_vertices;
            uint32_t little_num_dsts = container.DPs[i].num_dsts;
            uint32_t memory_offset = dense_buffers[i].src_buf_offset;

            int arg_idx = 0;
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffer));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, little_num_vertices));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, pip_num_edges));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, little_num_dsts));
            OCL_CHECK(err, err = kernel.setArg(arg_idx++, memory_offset));

            printf("Enqueue Little GS Kernel: DP No.%d, little pipe No.%d, "
                   "num_vertices %d, num_edges %d, num_dsts %d, memory_offset "
                   "%d\n",
                   i, pip, little_num_vertices, pip_num_edges, little_num_dsts,
                   memory_offset);

            OCL_CHECK(err,
                      err = acc.little_gs_queue[pip].enqueueTask(
                          kernel, nullptr, &acc.little_kernel_events[i][pip]));
        }
    }

    // 3.3: Enqueue apply_kernel (receives merged streams from little_merger and
    // big_merger)
    {
        auto &apply_kernel = acc.apply_krnl;

        int arg_idx = 0;
        OCL_CHECK(err, err = apply_kernel.setArg(
                           arg_idx++, apply_kernel_node_prop_buffer));
        OCL_CHECK(err,
                  err = apply_kernel.setArg(arg_idx++, little_dst_word_num));
        OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_dst_word_num));
        OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, (uint32_t)0));
        OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_dst_offset));

        OCL_CHECK(err, err = acc.apply_queue.enqueueTask(
                           apply_kernel, nullptr, &acc.apply_kernel_event));
    }

    // Enqueue hbm_writer kernel
    {
        auto &writer_kernel = acc.hbm_writer_krnl;
        uint32_t num_little_partitions = container.num_dense_partitions;
        ;
        uint32_t num_big_partitions = container.num_sparse_partitions;

        int arg_idx = 0;
        // writer_kernel_node_prop_buffers
        for (const auto &buffer : writer_kernel_node_prop_buffers) {
            OCL_CHECK(err, err = writer_kernel.setArg(arg_idx++, buffer));
        }
        OCL_CHECK(err, err = writer_kernel.setArg(arg_idx++,
                                                  writer_kernel_output_buffer));
        OCL_CHECK(err,
                  err = writer_kernel.setArg(arg_idx++, num_little_partitions));
        OCL_CHECK(err,
                  err = writer_kernel.setArg(arg_idx++, num_big_partitions));

        OCL_CHECK(err, err = acc.hbm_writer_queue.enqueueTask(
                           writer_kernel, nullptr, &acc.hbm_writer_event));
    }

    auto enqueue_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> enqueue_time = enqueue_end - enqueue_start;
    std::cout << "[SUCCESS] All kernel tasks enqueued for one iteration (Time: "
              << enqueue_time.count() << " seconds)" << std::endl;
}

// --- PHASE 4: DATA TRANSFER FROM FPGA ---
void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    std::cout << "--- [Host] Phase 4: Transferring results from FPGA ---"
              << std::endl;

    auto transfer_start = std::chrono::high_resolution_clock::now();

    // Read from the single unified output buffer
    OCL_CHECK(err, err = acc.hbm_writer_queue.enqueueReadBuffer(
                       writer_kernel_output_buffer, CL_FALSE, 0,
                       writer_kernel_host_outputs.size() * sizeof(bus_word_t),
                       writer_kernel_host_outputs.data()));

    // Wait for all transfers to complete
    acc.hbm_writer_queue.finish();

    auto transfer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transfer_time = transfer_end - transfer_start;
    std::cout << "[SUCCESS] All results transferred from HBM (Time: "
              << transfer_time.count() << " seconds)" << std::endl;
}

// --- PHASE 5: CONVERGENCE CHECK AND GLOBAL STATE UPDATE ---
// REWRITTEN: Implements unpacking logic to parse results from 512-bit words.
bool AlgorithmHost::check_convergence_and_update(
    const PartitionContainer &container) {
    bool changed = false;
    std::cout << "--- [Host] Phase 5: Unpacking results and checking for "
                 "convergence ---"
              << std::endl;

    std::map<int, distance_t> min_distances;
    const int dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;

    // Output layout: [little_dsts][big_dsts]
    // First little_dst_num entries are little partition destinations
    // Next big_dst_num entries are big partition destinations

    int word_idx = 0;
    int dist_in_word = 0;

    // Process little partition destinations [0:little_dst_num]
    for (int i = 0; i < container.num_dense_partitions; ++i) {
        const auto &little_partition = container.DPs[i];
        for (int local_id = 0; local_id < little_partition.num_dsts;
             ++local_id) {
            int bit_offset = dist_in_word * DISTANCE_BITWIDTH;
            ap_fixed_pod_t dist_pod =
                writer_kernel_host_outputs[word_idx].range(
                    bit_offset + DISTANCE_BITWIDTH - 1, bit_offset);

            if (little_partition.vtx_map_rev.count(local_id)) {
                int global_id = little_partition.vtx_map_rev.at(local_id);
                if (global_id < m_num_vertices) {
                    distance_t new_dist =
                        *reinterpret_cast<distance_t *>(&dist_pod);
                    printf("DP No.%d, little pipe No.%d, local_id %d, "
                           "global_id %d, new_dist %f\n",
                           i, 0, local_id, global_id, (float)new_dist);

                    if (min_distances.find(global_id) == min_distances.end() ||
                        new_dist < min_distances[global_id]) {
                        min_distances[global_id] = new_dist;
                    }
                }
            }

            dist_in_word++;
            if (dist_in_word >= dists_per_word) {
                dist_in_word = 0;
                word_idx++;
            }
        }

        // print other node's local & global id
        for (int local_id = little_partition.num_dsts;
             local_id < little_partition.num_vertices; ++local_id) {
            if (little_partition.vtx_map_rev.count(local_id)) {
                int global_id = little_partition.vtx_map_rev.at(local_id);
                printf("DP No.%d, little pipe No.%d, local_id %d, global_id "
                       "%d, not dst node\n",
                       i, 0, local_id, global_id);
            }
        }
    }

    if (dist_in_word != 0) {
        dist_in_word = 0;
        word_idx++;
    }

    // Process big partition destinations
    // [little_dst_num:little_dst_num+big_dst_num]
    for (int i = 0; i < container.num_sparse_partitions; ++i) {
        const auto &big_partition = container.SPs[i];
        for (int local_id = 0; local_id < big_partition.num_dsts; ++local_id) {
            int bit_offset = dist_in_word * DISTANCE_BITWIDTH;
            ap_fixed_pod_t dist_pod =
                writer_kernel_host_outputs[word_idx].range(
                    bit_offset + DISTANCE_BITWIDTH - 1, bit_offset);

            if (big_partition.vtx_map_rev.count(local_id)) {
                int global_id = big_partition.vtx_map_rev.at(local_id);
                if (global_id < m_num_vertices) {
                    distance_t new_dist =
                        *reinterpret_cast<distance_t *>(&dist_pod);
                    printf("SP No.%d, big pipe No.%d, local_id %d, global_id "
                           "%d, new_dist %f\n",
                           i, 0, local_id, global_id, (float)new_dist);

                    if (min_distances.find(global_id) == min_distances.end() ||
                        new_dist < min_distances[global_id]) {
                        min_distances[global_id] = new_dist;
                    }
                }
            }

            dist_in_word++;
            if (dist_in_word >= dists_per_word) {
                dist_in_word = 0;
                word_idx++;
            }
        }
        // print other node's local & global id
        for (int local_id = big_partition.num_dsts;
             local_id < big_partition.num_vertices; ++local_id) {
            if (big_partition.vtx_map_rev.count(local_id)) {
                int global_id = big_partition.vtx_map_rev.at(local_id);
                printf("SP No.%d, big pipe No.%d, local_id %d, global_id %d, "
                       "not dst node\n",
                       i, 0, local_id, global_id);
            }
        }
    }

    // Update global distance vector and check for changes
    for (auto const &[global_id, new_dist] : min_distances) {
        if (global_id < m_num_vertices && new_dist < h_distances[global_id]) {
            h_distances[global_id] = new_dist;
            changed = true;
        }
    }

    if (changed) {
        std::cout << "[INFO] Distances updated. Preparing for next iteration."
                  << std::endl;
    } else {
        std::cout << "[INFO] No distance updates. Algorithm has converged."
                  << std::endl;
    }

    return !changed;
}

// --- FINALIZATION ---
const std::vector<int> &AlgorithmHost::get_results() const {
    static std::vector<int> final_distances;
    final_distances.clear();
    final_distances.reserve(h_distances.size());

    for (const auto &dist : h_distances) {
        if (dist >= INFINITY_DIST) {
            final_distances.push_back(INFINITY_DIST);
        } else {
            final_distances.push_back(dist.to_int());
        }
    }
    return final_distances;
}
```

`scripts/host/generated_host.h`:

```h
#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "acc_setup/acc_setup.h"
#include "common.h"
#include "graph_preprocess/graph_preprocess.h"
#include <vector>

// Buffer for one big or little pipeline, host + device side
struct PipelineBuffer {
    std::vector<bus_word_t, aligned_allocator<bus_word_t>> packed_edge_props;
    cl::Buffer edge_props_buffer;
};

// One dense / sparse partition buffer including multiple pipelines
struct PartitionBuffer {
    std::vector<PipelineBuffer> pipelines;
    std::vector<bus_word_t, aligned_allocator<bus_word_t>> packed_dst_props;
    std::vector<bus_word_t, aligned_allocator<bus_word_t>> packed_node_props;
    uint32_t node_prop_offset;
    uint32_t dst_prop_offset;
    uint32_t src_buf_offset;
};

class AlgorithmHost {
  public:
    AlgorithmHost(AccDescriptor &acc);

    // --- MODIFICATION: Updated function signatures to use new data structures
    // ---
    void prepare_data(const PartitionContainer &container, int start_node);
    void setup_buffers(const PartitionContainer &container);
    void update_data(const PartitionContainer &container);
    void transfer_data_to_fpga(const PartitionContainer &container);
    void execute_kernel_iteration(const PartitionContainer &container);
    void transfer_data_from_fpga();
    bool check_convergence_and_update(const PartitionContainer &container);
    const std::vector<int> &get_results() const;

  private:
    AccDescriptor &acc;

    // Algorithm state
    int m_num_vertices;

    // Host-side master distance vector using original (global) vertex IDs
    std::vector<distance_t> h_distances;

    // Buffer containers for kernels (big + little)
    std::vector<PartitionBuffer> dense_buffers, sparse_buffers;

    // Buffer containers for HBM writer kernels (one entry per writer kernel
    // instance)
    std::vector<cl::Buffer> writer_kernel_node_prop_buffers;
    cl::Buffer writer_kernel_output_buffer;
    std::vector<bus_word_t, aligned_allocator<bus_word_t>>
        apply_kernel_node_props;
    cl::Buffer apply_kernel_node_prop_buffer;
    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        writer_kernel_node_props;
    std::vector<bus_word_t, aligned_allocator<bus_word_t>>
        writer_kernel_host_outputs;
    uint32_t big_dst_offset = 0;
};

#endif // __GENERATED_HOST_H__
```

`scripts/host/graph_loader.cpp`:

```cpp
#include "graph_loader.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// 边结构体，用于临时存储从文件中读取的边
struct Edge {
    int src, dest, weight;
};

// 主函数，从文件中加载图并转换为 CSR 格式
GraphCSR load_graph_from_file(const std::string &file_path) {
    // ---- 1. 根据文件扩展名判断图的格式 ----
    bool is_one_based = false; // 默认为 0-indexed
    char comment_char = '#';   // 默认注释符

    if (file_path.size() > 4 &&
        file_path.substr(file_path.size() - 4) == ".mtx") {
        is_one_based = true; // .mtx 文件通常是 1-indexed
        comment_char = '%';  // .mtx 文件使用 '%' 作为注释
        std::cout << "Detected .mtx format (1-based indexing)." << std::endl;
    } else {
        std::cout << "Detected .txt format (0-based indexing)." << std::endl;
    }

    // ---- 2. 打开文件并逐行解析 ----
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open graph file: " << file_path
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<Edge> edges;
    int max_vertex_id = -1;
    int min_vertex_id = 1;
    std::string line;
    long line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        // 跳过空行和注释行
        if (line.empty() || line[0] == comment_char) {
            continue;
        }

        std::istringstream iss(line);
        Edge edge;

        // 尝试读取 src, dest, weight
        if (iss >> edge.src >> edge.dest >> edge.weight) {
            // 成功读取三个值
        } else {
            // 如果失败，重置流并尝试只读取 src, dest
            iss.clear();
            iss.seekg(0);
            if (iss >> edge.src >> edge.dest) {
                edge.weight = 1; // 赋予默认权重 1
            } else {
                std::cerr << "Warning: Skipping malformed line " << line_num
                          << ": " << line << std::endl;
                continue;
            }
        }

        // 如果是 1-based 格式，转换为 0-based
        if (is_one_based) {
            edge.src--;
            edge.dest--;
        }

        // 检查顶点ID是否有效
        if (edge.src < 0 || edge.dest < 0) {
            std::cerr
                << "Warning: Skipping edge with negative vertex ID on line "
                << line_num << std::endl;
            continue;
        }

        edges.push_back(edge);
        max_vertex_id = std::max({max_vertex_id, edge.src, edge.dest});
        min_vertex_id = std::min({min_vertex_id, edge.src, edge.dest});
    }
    file.close();

    if (min_vertex_id == 1 && !is_one_based) {
        // modify to 0-based
        std::cout << "Converting graph from 1-based to 0-based indexing."
                  << std::endl;
        for (auto &edge : edges) {
            edge.src--;
            edge.dest--;
        }
    }

    // ---- 3. 将边列表转换为 CSR 格式 ----
    GraphCSR graph;
    if (max_vertex_id == -1) { // 如果文件为空或无效
        graph.num_vertices = 0;
        graph.num_edges = 0;
        std::cout << "Graph is empty." << std::endl;
        return graph;
    }

    graph.num_vertices = max_vertex_id + 1;
    graph.num_edges = edges.size();

    std::cout << "Graph loaded: " << graph.num_vertices << " vertices, "
              << graph.num_edges << " edges." << std::endl;

    // 为了进行 CSR 转换，按源顶点 ID 对边进行排序
    std::sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b) {
        if (a.src != b.src) {
            return a.src < b.src;
        }
        return a.dest < b.dest;
    });

    // 分配 CSR 数组内存
    graph.offsets.resize(graph.num_vertices + 1);
    graph.columns.resize(graph.num_edges);
    graph.weights.resize(graph.num_edges);

    // 填充 columns 和 weights 数组，并计算每个顶点的出度
    std::vector<int> out_degree(graph.num_vertices, 0);
    for (int i = 0; i < graph.num_edges; ++i) {
        graph.columns[i] = edges[i].dest;
        graph.weights[i] = edges[i].weight;
        out_degree[edges[i].src]++;
    }

    // 通过出度的前缀和计算 offsets 数组
    graph.offsets[0] = 0;
    for (int i = 0; i < graph.num_vertices; ++i) {
        graph.offsets[i + 1] = graph.offsets[i] + out_degree[i];
    }

    return graph;
}
```

`scripts/host/graph_loader.h`:

```h
#ifndef __GRAPH_LOADER_H__
#define __GRAPH_LOADER_H__

#include "common.h"

// Loads a graph from a text file (edge list format: src dst weight)
// and converts it into a GraphCSR object.
GraphCSR load_graph_from_file(const std::string &file_path);

#endif // __GRAPH_LOADER_H__
```

`scripts/host/graph_preprocess/graph_preprocess.cpp`:

```cpp
#include "graph_preprocess.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

// A local helper struct to temporarily hold edge information with global vertex
// IDs.
struct Edge {
    int src, dest, weight;
};

/**
 * @brief Partitions a global graph and preprocesses each partition into a local
 * CSR format.
 *
 * This function implements a partitioning strategy based on destination
 * vertices.
 * 1.  It identifies all unique destination vertices in the graph.
 * 2.  It distributes these destination vertices disjointly and as evenly as
 * possible among all available partitions (for both big and little kernels).
 * 3.  It assigns each edge from the global graph to the partition that is
 * responsible for its destination vertex.
 * 4.  For each partition, it collects all unique vertices involved (both
 * sources and destinations).
 * 5.  It performs vertex ID compression for each partition, creating a local ID
 * space. Destination vertices are mapped first to ensure they occupy the lower
 * ID range.
 * 6.  It rewrites the partition's edges using these new local IDs.
 * 7.  Finally, it converts the rewritten edges into a local CSR format.
 *
 * @param graph The input global graph in CSR format.
 * @return A PartitionContainer object containing all processed partitions.
 */
PartitionContainer partitionGraph(const GraphCSR *graph) {
    std::cout << "--- Starting Graph Partitioning and Preprocessing "
                 "(2-Partition Mode) ---"
              << std::endl;
    PartitionContainer container;
    container.num_graph_vertices = graph->num_vertices;
    container.num_graph_edges = graph->num_edges;
    printf("Global graph has %d vertices and %d edges.\n", graph->num_vertices,
           graph->num_edges);

    std::cout << "[INFO] Creating 2 partitions: 1 little (max "
              << LITTLE_MAX_DST << " dsts) and 1 big (max " << BIG_MAX_DST
              << " dsts)" << std::endl;

    // --- PHASE 1: Identify and Collect All Unique Destination Vertices ---
    std::set<int> unique_dst_vertices_set;
    std::unordered_map<int, int> node_indegrees;
    for (int i = 0; i < graph->num_edges; ++i) {
        int dst = graph->columns[i];
        unique_dst_vertices_set.insert(dst);
        if (node_indegrees.find(dst) == node_indegrees.end()) {
            node_indegrees[dst] = 0;
        }
        node_indegrees[dst]++;
    }
    std::vector<int> unique_dst_vertices(unique_dst_vertices_set.begin(),
                                         unique_dst_vertices_set.end());

    // Sort unique_dst_vertices by indegree (descending order) to prioritize
    // high-degree nodes for little partition
    std::sort(unique_dst_vertices.begin(), unique_dst_vertices.end(),
              [&node_indegrees](int a, int b) {
                  return node_indegrees[a] > node_indegrees[b];
              });

    std::cout << "[PHASE 1] Found " << unique_dst_vertices.size()
              << " unique destination vertices (sorted by indegree)."
              << std::endl;

    // --- PHASE 2: Distribute Destination Vertices to 2 Partitions ---
    std::vector<std::set<int>> little_dst_sets, big_dst_sets;
    std::unordered_map<int, int>
        dst_vertex_to_partition_map; // 0 ~ little_partition_sizes.size()-1 =>
                                     // little

    std::vector<size_t> little_partition_sizes;
    std::vector<size_t> big_partition_sizes;

    size_t remaining_dsts = unique_dst_vertices.size();
    while (remaining_dsts > 0) {
        size_t assign_to_little =
            std::min((size_t)LITTLE_MAX_DST, remaining_dsts);
        if (assign_to_little == remaining_dsts) {
            assign_to_little = (size_t)(remaining_dsts * 0.8);
        }
        little_partition_sizes.push_back(assign_to_little);
        little_dst_sets.emplace_back();
        remaining_dsts -= assign_to_little;

        size_t assign_to_big = std::min((size_t)BIG_MAX_DST, remaining_dsts);
        big_partition_sizes.push_back(assign_to_big);
        big_dst_sets.emplace_back();
        remaining_dsts -= assign_to_big;
    }

    size_t vertex_idx = 0;
    for (size_t p = 0; p < little_partition_sizes.size(); ++p) {
        size_t part_size = little_partition_sizes[p];
        for (size_t i = 0; i < part_size; ++i) {
            int vertex_id = unique_dst_vertices[vertex_idx++];
            little_dst_sets[p].insert(vertex_id);
            dst_vertex_to_partition_map[vertex_id] = p; // Little partition
        }
    }

    for (size_t p = 0; p < big_partition_sizes.size(); ++p) {
        size_t part_size = big_partition_sizes[p];
        for (size_t i = 0; i < part_size; ++i) {
            int vertex_id = unique_dst_vertices[vertex_idx++];
            big_dst_sets[p].insert(vertex_id);
            dst_vertex_to_partition_map[vertex_id] =
                p + little_partition_sizes.size(); // Big partition
        }
    }

    std::cout << "[PHASE 2] Little partition assigned "
              << std::accumulate(little_partition_sizes.begin(),
                                 little_partition_sizes.end(), 0)
              << " dst vertices across " << little_partition_sizes.size()
              << " partitions." << std::endl;
    std::cout << "[PHASE 2] Big partition assigned "
              << std::accumulate(big_partition_sizes.begin(),
                                 big_partition_sizes.end(), 0)
              << " dst vertices across " << big_partition_sizes.size()
              << " partitions." << std::endl;

    // --- PHASE 3: Assign Edges to 2 Partitions Based on Destination Vertex ---
    std::vector<std::vector<Edge>> edges_lists;
    edges_lists.resize(little_partition_sizes.size() +
                       big_partition_sizes.size());
    size_t little_edge_num = 0, big_edge_num = 0;
    for (int u = 0; u < graph->num_vertices; ++u) {
        for (int i = graph->offsets[u]; i < graph->offsets[u + 1]; ++i) {
            int v = graph->columns[i];
            int w = graph->weights[i];

            // Find which partition this edge belongs to
            auto it = dst_vertex_to_partition_map.find(v);
            if (it != dst_vertex_to_partition_map.end()) {
                int partition_idx = it->second;
                edges_lists[partition_idx].push_back({u, v, w});
                if (partition_idx < little_partition_sizes.size()) {
                    little_edge_num++;
                } else {
                    big_edge_num++;
                }
            } else {
                std::cerr << "[ERROR] Destination vertex " << v
                          << " not found in any partition!" << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "[PHASE 3] Assigned " << little_edge_num
              << " edges to little partitions." << std::endl;
    std::cout << "[PHASE 3] Assigned " << big_edge_num
              << " edges to big partitions." << std::endl;

    // --- PHASE 4: Process Each Partition (Compress IDs and Distribute Edges)
    // ---
    std::cout << "[PHASE 4] Processing partitions..." << std::endl;

    // Helper lambda to process a partition
    auto process_partition = [&](const std::vector<Edge> &partition_edges,
                                 const std::set<int> &partition_dst_nodes,
                                 bool is_dense,
                                 int num_pipelines) -> PartitionDescriptor {
        PartitionDescriptor pd;
        pd.is_dense = is_dense;
        pd.num_pipelines = num_pipelines;

        if (partition_edges.empty()) {
            std::cout << "  - Partition has no edges. Creating empty partition."
                      << std::endl;
            pd.num_edges = 0;
            pd.num_vertices = 0;
            pd.num_dsts = 0;
            return pd;
        } else {
            // --- 4.1: Collect unique vertices and build ID mappings ---
            std::set<int> local_vertices_set;
            for (const auto &edge : partition_edges) {
                local_vertices_set.insert(edge.src);
                local_vertices_set.insert(edge.dest);
            }

            // --- 4.1.1: Create ordered list of destination vertices (already
            // sorted by indegree from PHASE 1) ---
            std::vector<int> ordered_dst_vertices(partition_dst_nodes.begin(),
                                                  partition_dst_nodes.end());

            int local_id_counter = 0;
            // First, map destination vertices to guarantee they have
            // lower-range IDs
            for (int global_id : ordered_dst_vertices) {
                pd.vtx_map[global_id] = local_id_counter;
                pd.vtx_map_rev[local_id_counter] = global_id;
                local_id_counter++;
            }
            pd.num_dsts = partition_dst_nodes.size();

            // Then, map the remaining source vertices
            for (int global_id : local_vertices_set) {
                if (pd.vtx_map.find(global_id) == pd.vtx_map.end()) {
                    pd.vtx_map[global_id] = local_id_counter;
                    pd.vtx_map_rev[local_id_counter] = global_id;
                    local_id_counter++;
                }
            }
            pd.num_vertices = local_vertices_set.size();

            // --- 4.2: Rewrite edges with local, compressed IDs and sort by src
            // ---
            std::vector<Edge> local_edges;
            local_edges.reserve(partition_edges.size());

            uint32_t last_src_buffer = 0;
            uint32_t last_src_id = 0;

            for (const auto &global_edge : partition_edges) {
                uint32_t src_id = pd.vtx_map[global_edge.src];
                uint32_t dest_id = pd.vtx_map[global_edge.dest];
                uint32_t weight = global_edge.weight;

                uint32_t cur_src_buffer = floor(src_id / SRC_BUFFER_SIZE);
                if (is_dense && cur_src_buffer != last_src_buffer) {
                    uint32_t mod8 = local_edges.size() % 8;
                    if (mod8 != 0) {
                        // Pad with dummy edges to align to 8-edge boundary
                        for (uint32_t pad = 0; pad < (8 - mod8); pad++) {
                            local_edges.push_back({last_src_id, 0x7FFFFFFF, 1});
                        }
                    }
                    last_src_buffer = cur_src_buffer;
                }
                last_src_id = src_id;
                local_edges.push_back({(int)src_id, (int)dest_id, (int)weight});
            }

            // Sort edges by source node ID (ascending)
            std::sort(
                local_edges.begin(), local_edges.end(),
                [](const Edge &a, const Edge &b) { return a.src < b.src; });

            pd.num_edges = local_edges.size();

            // --- 4.3: Distribute edges evenly among pipelines ---
            pd.pipeline_edges.resize(num_pipelines);
            int edges_per_pipeline = pd.num_edges / num_pipelines;

            for (int pip = 0; pip < num_pipelines; ++pip) {
                pd.pipeline_edges[pip].pipeline_id = pip;
                int start_idx = pip * edges_per_pipeline;
                int end_idx =
                    std::min(start_idx + edges_per_pipeline, (int)pd.num_edges);
                if (pip == num_pipelines - 1) {
                    end_idx =
                        pd.num_edges; // Last pipeline takes remaining edges
                }
                pd.pipeline_edges[pip].num_edges = end_idx - start_idx;

                // Build CSR for this pipeline
                pd.pipeline_edges[pip].offsets.resize(pd.num_vertices + 1, 0);
                pd.pipeline_edges[pip].columns.reserve(
                    pd.pipeline_edges[pip].num_edges);
                pd.pipeline_edges[pip].weights.reserve(
                    pd.pipeline_edges[pip].num_edges);

                // Count out-degrees for this pipeline's edges
                std::vector<int> out_degree(pd.num_vertices, 0);
                for (int j = start_idx; j < end_idx; ++j) {
                    out_degree[local_edges[j].src]++;
                }

                // Build offsets
                pd.pipeline_edges[pip].offsets[0] = 0;
                for (int v = 0; v < pd.num_vertices; ++v) {
                    pd.pipeline_edges[pip].offsets[v + 1] =
                        pd.pipeline_edges[pip].offsets[v] + out_degree[v];
                }

                // Fill columns and weights
                std::vector<int> current_offset =
                    pd.pipeline_edges[pip].offsets;
                for (int j = start_idx; j < end_idx; ++j) {
                    int src = local_edges[j].src;
                    int idx = current_offset[src]++;
                    pd.pipeline_edges[pip].columns.push_back(
                        local_edges[j].dest);
                    pd.pipeline_edges[pip].weights.push_back(
                        local_edges[j].weight);
                }
            }
        }

        return pd;
    };

    for (size_t p = 0; p < little_partition_sizes.size(); ++p) {
        PartitionDescriptor little_pd = process_partition(
            edges_lists[p], little_dst_sets[p], true, LITTLE_KERNEL_NUM);
        container.DPs.push_back(little_pd);
        std::cout << "  - Little partition " << p << ": "
                  << little_pd.num_vertices << " vertices, "
                  << little_pd.num_dsts << " dsts, " << little_pd.num_edges
                  << " edges distributed to " << little_pd.num_pipelines
                  << " pipelines." << std::endl;
        // print how much edge for each partition each pipeline
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            std::cout << "    - Pipeline " << pip << ": "
                      << little_pd.pipeline_edges[pip].num_edges << " edges."
                      << std::endl;
        }
    }

    for (size_t p = 0; p < big_partition_sizes.size(); ++p) {
        PartitionDescriptor big_pd =
            process_partition(edges_lists[p + little_partition_sizes.size()],
                              big_dst_sets[p], false, BIG_KERNEL_NUM);
        container.SPs.push_back(big_pd);
        std::cout << "  - Big partition " << p << ": " << big_pd.num_vertices
                  << " vertices, " << big_pd.num_dsts << " dsts, "
                  << big_pd.num_edges << " edges distributed to "
                  << big_pd.num_pipelines << " pipelines." << std::endl;
        // print how much edge for each partition each pipeline
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            std::cout << "    - Pipeline " << pip << ": "
                      << big_pd.pipeline_edges[pip].num_edges << " edges."
                      << std::endl;
        }
    }

    container.num_dense_partitions = container.DPs.size();
    container.num_sparse_partitions = container.SPs.size();

    std::cout << "[SUCCESS] Graph partitioning and preprocessing complete "
                 "(2-partition mode)."
              << std::endl;
    std::cout << "  Total: " << container.num_dense_partitions << " dense + "
              << container.num_sparse_partitions << " sparse partitions."
              << std::endl;
    return container;
}

```

`scripts/host/graph_preprocess/graph_preprocess.h`:

```h
#ifndef GRAPH_PREPROCESS_H
#define GRAPH_PREPROCESS_H

#include "common.h"
#include "host_config.h"

#include <algorithm> // std::swap
#include <iomanip>
#include <numeric> // std::iota
#include <vector>

/**
 * @struct PipelineEdges
 * @brief Holds edge data for a single pipeline instance.
 */
typedef struct PipelineEdges {
    unsigned int pipeline_id;
    unsigned int num_edges;
    std::vector<int> offsets; // Per-vertex offsets for this pipeline's edges
    std::vector<int> columns; // Destination IDs
    std::vector<int> weights; // Edge weights
} PipelineEdges;

/**
 * @struct PartitionDescriptor
 * @brief Describes a single graph partition (either big or little).
 * * This structure holds a self-contained CSR representation of a graph
 * partition, including the mapping between its local, compressed vertex IDs and
 * the original global vertex IDs. Edges are distributed among multiple
 * pipelines.
 */
typedef struct PartitionDescriptor {
    // Metadata about the partition
    unsigned int num_edges;    // Total edges across all pipelines
    unsigned int num_vertices; // Number of vertices *within this partition*
    unsigned int num_dsts;     // Number of destination vertices
    bool is_dense;             // True for little kernel, false for big kernel
    unsigned int
        num_pipelines; // Number of pipeline instances for this partition

    // The core graph data for this partition
    // Vertex mappings are shared across all pipelines
    std::unordered_map<int, int> vtx_map;     // Global ID -> Local ID
    std::unordered_map<int, int> vtx_map_rev; // Local ID -> Global ID

    // Edge data distributed across pipelines
    std::vector<PipelineEdges> pipeline_edges; // One entry per pipeline

} PartitionDescriptor;

/**
 * @struct PartitionContainer
 * @brief A container holding all graph partitions.
 * * This top-level structure contains metadata for the entire graph and holds
 * separate vectors for partitions assigned to sparse (big) and dense (little)
 * kernels.
 */
typedef struct PartitionContainer {
    // Global graph metadata
    unsigned int num_graph_vertices;
    unsigned int num_graph_edges;

    // Partition collections
    unsigned int num_dense_partitions;
    unsigned int num_sparse_partitions;

    std::vector<PartitionDescriptor>
        DPs; // Partitions for Dense (little) kernels
    std::vector<PartitionDescriptor> SPs; // Partitions for Sparse (big) kernels

} PartitionContainer;

/**
 * @brief Partitions the global graph and preprocesses each partition into a CSR
 * format.
 * * This function implements the new partitioning strategy based on destination
 * vertices and creates a container with all the partitioned graph data.
 * @param graph The input global graph in CSR format.
 * @return A PartitionContainer object containing all processed partitions.
 */
PartitionContainer partitionGraph(const GraphCSR *graph);

#endif
```

`scripts/host/host.cpp`:

```cpp
#include "common.h"
#include "fpga_executor.h" // <-- 修改: 包含新的执行器
#include "graph_loader.h"
#include "host_verifier.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <xclbin_file> <graph_data_file>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbin_file = argv[1];
    std::string graph_file = argv[2];
    int start_node = 0;

    // 1. 加载图数据 (不变)
    std::cout << "--- Step 1: Loading Graph Data ---" << std::endl;
    GraphCSR graph = load_graph_from_file(graph_file);
    if (graph.num_vertices == 0) {
        return EXIT_FAILURE;
    }

    // 2. 在 FPGA 上运行 (调用新的通用执行器)
    std::cout << "\n--- Step 2: Running on FPGA ---" << std::endl;
    double total_kernel_time_sec = 0;
    int iter_count = 0;
    std::vector<int> fpga_distances = run_fpga_kernel(
        xclbin_file, graph, start_node, total_kernel_time_sec, iter_count);

    // 3. 在 Host CPU 上验证 (不变, 按你的要求保留)
    std::cout << "\n--- Step 3: Verifying on Host CPU ---" << std::endl;
    std::vector<int> host_distances = verify_on_host(graph, start_node);

    // 4. 比较结果 (不变)
    std::cout << "\n--- Step 4: Comparing Results ---" << std::endl;
    int error_count = 0;
    for (int i = 0; i < graph.num_vertices; ++i) {
        if (fpga_distances[i] != host_distances[i]) {
            if (error_count < 10) {
                std::cout << "Mismatch at vertex " << i << ": "
                          << "FPGA_Result = " << fpga_distances[i] << ", "
                          << "Host_Result = " << host_distances[i] << std::endl;
            }
            error_count++;
        }
    }

    // 5. 最终报告 (不变)
    std::cout << "\n--- Final Report ---" << std::endl;
    if (error_count == 0) {
        std::cout << "SUCCESS: Results match!" << std::endl;
    } else {
        std::cout << "FAILURE: Found " << error_count << " mismatches."
                  << std::endl;
    }

    std::cout << "Total FPGA Kernel Execution Time: "
              << total_kernel_time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "Total MTEPS (Edges / Total Time): "
              << ((double)graph.num_edges * iter_count) /
                     total_kernel_time_sec / 1.0e6
              << " MTEPS" << std::endl;

    return (error_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;

    std::cout << " finish\n";
}

```

`scripts/host/host.mk`:

```mk

# Compiler
CXX := g++

# --- Configuration ---

# Executable name (can be passed from a top-level Makefile)
EXECUTABLE ?= graphyflow_host

# Top-level directory for host source code
HOST_DIR := scripts/host

# --- Automatic File Discovery ---

# Use the shell's 'find' command to recursively find all .cpp files
# This automatically includes files in subdirectories like 'acc_setup'.
HOST_SRCS := $(shell find $(HOST_DIR) -name '*.cpp')

# Generate a list of object files (.o) from the source files list
# e.g., "scripts/host/host.cpp" becomes "scripts/host/host.o"
OBJECTS := $(HOST_SRCS:.cpp=.o)


# --- Compiler and Linker Flags ---

# Include directories
# We now add the top-level host directory. The compiler will handle subdirectories.
CXXFLAGS := -I$(HOST_DIR)
CXXFLAGS += -Iscripts/kernel
CXXFLAGS += -I$(XILINX_XRT)/include
CXXFLAGS += -I$(XILINX_VITIS)/include
CXXFLAGS += -I$(XILINX_HLS)/include

ifeq ($(TARGET),$(filter $(TARGET), sw_emu hw_emu))
CXXFLAGS += -DEMULATION
endif

# Compiler flags
CXXFLAGS += -std=c++17 -O3 -Wall -g # Added -g for easier debugging

# Linker flags (no changes needed here)
LDFLAGS := -L$(XILINX_XRT)/lib
LDFLAGS += -lOpenCL -lxrt_coreutil -lstdc++ -lrt -pthread -Wl,--export-dynamic


# --- Build Rules ---

# The "all" rule is the default target. It depends on the final executable.
all: $(EXECUTABLE)

# Rule to link the final executable from all the object files.
# This rule runs only if any of the object files (.o) have changed.
$(EXECUTABLE): $(OBJECTS)
	@echo "==> Linking executable: $@"
	$(CXX) $(OBJECTS) -o $(EXECUTABLE) $(LDFLAGS)

# Pattern rule to compile any .cpp file into its corresponding .o file.
# This rule runs for each .cpp file that has been modified.
%.o: %.cpp
	@echo "==> Compiling: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to clean up the build artifacts.
# Call with "make clean".
clean:
	@echo "==> Cleaning up generated files"
	rm -f $(EXECUTABLE) $(OBJECTS)

# Phony targets are not files. 'all' and 'clean' are actions.
.PHONY: all clean
```

`scripts/host/host_bellman_ford.cpp`:

```cpp
#include "host_bellman_ford.h"

bool host_bellman_ford_iteration(const GraphCSR &graph,
                                 std::vector<int> &distances) {
    bool changed = false;

    // --- USER MODIFIABLE SECTION: Host Computation Logic ---
    // For each vertex, relax all outgoing edges
    for (int u = 0; u < graph.num_vertices; ++u) {
        if (distances[u] != INFINITY_DIST) {
            for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {
                int v = graph.columns[i];
                if ((v & 0x40000000) != 0) {
                    continue; // Skip dummy edges
                }
                int weight = graph.weights[i];

                // Relaxation step
                if (distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    changed = true;
                }
            }
        }
    }
    // --- END USER MODIFIABLE SECTION ---

    return changed;
}
```

`scripts/host/host_bellman_ford.h`:

```h
#ifndef __HOST_BELLMAN_FORD_H__
#define __HOST_BELLMAN_FORD_H__

#include "common.h"

// This function performs a single iteration of the Bellman-Ford algorithm on
// the host CPU. It returns true if any distance value was updated, false
// otherwise.
bool host_bellman_ford_iteration(const GraphCSR &graph,
                                 std::vector<int> &distances);

#endif // __HOST_BELLMAN_FORD_H__
```

`scripts/host/host_config.h`:

```h
#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 3
#define LITTLE_KERNEL_NUM 11

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define LITTLE_KERNEL_HBM_EDGE_ID {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
#define LITTLE_KERNEL_HBM_NODE_ID {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21}
#define BIG_KERNEL_HBM_EDGE_ID {22, 24, 26}
#define BIG_KERNEL_HBM_NODE_ID {23, 25, 27}

#endif /* __HOST_CONFIG_H__ */

```

`scripts/host/host_verifier.cpp`:

```cpp
#include "host_verifier.h"
#include "host_bellman_ford.h"
#include <iostream>

std::vector<int> verify_on_host(const GraphCSR &graph, int start_node) {
    std::vector<int> distances(graph.num_vertices, INFINITY_DIST);
    distances[start_node] = 0;

    int max_iterations = graph.num_vertices;
    int iter = 0;
    bool changed = true;

    std::cout << "\nStarting Host verification..." << std::endl;

    while (changed && iter < max_iterations) {
        // Run one iteration of the algorithm
        changed = host_bellman_ford_iteration(graph, distances);
        iter++;
    }

    std::cout << "Host computation converged after " << iter << " iterations."
              << std::endl;

    // Check for negative weight cycles (optional but good practice)
    if (iter == max_iterations &&
        host_bellman_ford_iteration(graph, distances)) {
        std::cout << "Warning: Negative weight cycle detected by host verifier."
                  << std::endl;
    }

    return distances;
}
```

`scripts/host/host_verifier.h`:

```h
#ifndef __HOST_VERIFIER_H__
#define __HOST_VERIFIER_H__

#include "common.h"

// Main function to run the Bellman-Ford algorithm on the host CPU for
// verification.
std::vector<int> verify_on_host(const GraphCSR &graph, int start_node);

#endif // __HOST_VERIFIER_H__
```

`scripts/host/xcl2.cpp`:

```cpp
/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "xcl2.h"
#include <climits>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#if defined(_WINDOWS)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace xcl {
std::vector<cl::Device> get_devices(const std::string &vendor_name) {
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++) {
        platform = platforms[i];
        OCL_CHECK(err, std::string platformName =
                           platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (!(platformName.compare(vendor_name))) {
            std::cout << "Found Platform" << std::endl;
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        std::cout << "Found the following platforms : " << std::endl;
        for (size_t j = 0; j < platforms.size(); j++) {
            platform = platforms[j];
            OCL_CHECK(err, std::string platformName =
                               platform.getInfo<CL_PLATFORM_NAME>(&err));
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
        }
        exit(EXIT_FAILURE);
    }
    // Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    OCL_CHECK(err,
              err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}

std::vector<cl::Device> get_xil_devices() { return get_devices("Xilinx"); }

cl::Device find_device_bdf(const std::vector<cl::Device> &devices,
                           const std::string &bdf) {
    char device_bdf[20];
    cl_int err;
    cl::Device device;
    int cnt = 0;
    for (uint32_t i = 0; i < devices.size(); i++) {
        OCL_CHECK(err,
                  err = devices[i].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
        if (bdf == device_bdf) {
            device = devices[i];
            cnt++;
            break;
        }
    }
    if (cnt == 0) {
        std::cout << "Invalid device bdf. Please check and provide valid bdf\n";
        exit(EXIT_FAILURE);
    }
    return device;
}
cl_device_id find_device_bdf_c(cl_device_id *devices, const std::string &bdf,
                               cl_uint device_count) {
    char device_bdf[20];
    cl_int err;
    cl_device_id device;
    int cnt = 0;
    for (uint32_t i = 0; i < device_count; i++) {
        err = clGetDeviceInfo(devices[i], CL_DEVICE_PCIE_BDF,
                              sizeof(device_bdf), device_bdf, 0);
        if (err != CL_SUCCESS) {
            std::cout << "Unable to extract the device BDF details\n";
            exit(EXIT_FAILURE);
        }
        if (bdf == device_bdf) {
            device = devices[i];
            cnt++;
            break;
        }
    }
    if (cnt == 0) {
        std::cout << "Invalid device bdf. Please check and provide valid bdf\n";
        exit(EXIT_FAILURE);
    }
    return device;
}
std::vector<unsigned char>
read_binary_file(const std::string &xclbin_file_name) {
    std::cout << "INFO: Reading " << xclbin_file_name << std::endl;
    FILE *fp;
    if ((fp = fopen(xclbin_file_name.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n",
               xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    // Loading XCL Bin into char buffer
    std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char *>(buf.data()), nb);
    return buf;
}

bool is_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if (xcl_mode != nullptr) {
        ret = true;
    }
    return ret;
}

bool is_hw_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if ((xcl_mode != nullptr) && !strcmp(xcl_mode, "hw_emu")) {
        ret = true;
    }
    return ret;
}
double round_off(double n) {
    double d = n * 100.0;
    int i = d + 0.5;
    d = i / 100.0;
    return d;
}

std::string convert_size(size_t size) {
    static const char *SIZES[] = {"B", "KB", "MB", "GB"};
    uint32_t div = 0;
    size_t rem = 0;

    while (size >= 1024 && div < (sizeof SIZES / sizeof *SIZES)) {
        rem = (size % 1024);
        div++;
        size /= 1024;
    }

    double size_d = (float)size + (float)rem / 1024.0;
    double size_val = round_off(size_d);

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << size_val;
    std::string size_str = stream.str();
    std::string result = size_str + " " + SIZES[div];
    return result;
}

bool is_xpr_device(const char *device_name) {
    const char *output = strstr(device_name, "xpr");

    if (output == nullptr) {
        return false;
    } else {
        return true;
    }
}
}; // namespace xcl

```

`scripts/host/xcl2.h`:

```h
/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error, call)                                                 \
    call;                                                                      \
    if (error != CL_SUCCESS) {                                                 \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, \
               __LINE__, error);                                               \
        exit(EXIT_FAILURE);                                                    \
    }

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include <fstream>
#include <iostream>
// When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
// hood
// User ptr is used if and only if it is properly aligned (page aligned). When
// not
// aligned, runtime has no choice but to create its own host side buffer that
// backs
// user ptr. This in turn implies that all operations that move data to and from
// device incur an extra memcpy to move data to/from runtime's own host buffer
// from/to user pointer. So it is recommended to use this allocator if user wish
// to
// Create Buffer/Memory Object with CL_MEM_USE_HOST_PTR to align user buffer to
// the
// page boundary. It will ensure that user buffer will be used when user create
// Buffer/Mem Object with CL_MEM_USE_HOST_PTR.
template <typename T> struct aligned_allocator {
    using value_type = T;

    aligned_allocator() {}

    aligned_allocator(const aligned_allocator &) {}

    template <typename U> aligned_allocator(const aligned_allocator<U> &) {}

    T *allocate(std::size_t num) {
        void *ptr = nullptr;

#if defined(_WINDOWS)
        {
            ptr = _aligned_malloc(num * sizeof(T), 4096);
            if (ptr == nullptr) {
                std::cout << "Failed to allocate memory" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
#else
        {
            if (posix_memalign(&ptr, 4096, num * sizeof(T)))
                throw std::bad_alloc();
        }
#endif
        return reinterpret_cast<T *>(ptr);
    }
    void deallocate(T *p, std::size_t num) {
#if defined(_WINDOWS)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

namespace xcl {
std::vector<cl::Device> get_xil_devices();
std::vector<cl::Device> get_devices(const std::string &vendor_name);
cl::Device find_device_bdf(const std::vector<cl::Device> &devices,
                           const std::string &bdf);
cl_device_id find_device_bdf_c(cl_device_id *devices, const std::string &bdf,
                               cl_uint dev_count);
std::string convert_size(size_t size);
std::vector<unsigned char>
read_binary_file(const std::string &xclbin_file_name);
bool is_emulation();
bool is_hw_emulation();
bool is_xpr_device(const char *device_name);
class P2P {
  public:
    static decltype(&xclGetMemObjectFd) getMemObjectFd;
    static decltype(&xclGetMemObjectFromFd) getMemObjectFromFd;
    static void init(const cl_platform_id &platform) {
        void *bar = clGetExtensionFunctionAddressForPlatform(
            platform, "xclGetMemObjectFd");
        getMemObjectFd = (decltype(&xclGetMemObjectFd))bar;
        bar = clGetExtensionFunctionAddressForPlatform(platform,
                                                       "xclGetMemObjectFromFd");
        getMemObjectFromFd = (decltype(&xclGetMemObjectFromFd))bar;
    }
};
class Ext {
  public:
    static decltype(&xclGetComputeUnitInfo) getComputeUnitInfo;
    static void init(const cl_platform_id &platform) {
        void *bar = clGetExtensionFunctionAddressForPlatform(
            platform, "xclGetComputeUnitInfo");
        getComputeUnitInfo = (decltype(&xclGetComputeUnitInfo))bar;
    }
};
} // namespace xcl

```

`scripts/kernel/apply_kernel.cpp`:

```cpp
#include "shared_kernel_params.h"

void merge_big_little_writes(
    hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
    hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
    hls::stream<in_write_burst_w_dst_pkt_t> &kernel_out_stream,
    uint32_t little_kernel_length, uint32_t big_kernel_length,
    uint32_t little_kernel_st_offset, uint32_t big_kernel_st_offset) {
    write_burst_pkt_t big_tmp_prop_pkt;
    write_burst_pkt_t little_tmp_prop_pkt;

    uint32_t little_idx = little_kernel_st_offset;
    uint32_t big_idx = big_kernel_st_offset;
    uint32_t total_length = little_kernel_length + big_kernel_length;

LOOP_MERGE_WRITES:
    while (true) {
        if (total_length == 0) {
            in_write_burst_w_dst_pkt_t end_pkt;
            end_pkt.end_flag = true;
            kernel_out_stream.write(end_pkt);
            break;
        }

        if (little_kernel_out_stream.read_nb(little_tmp_prop_pkt)) {
            in_write_burst_w_dst_pkt_t little_write_burst;
            little_write_burst.data = little_tmp_prop_pkt.data;
            little_write_burst.dest_addr = little_idx;
            little_write_burst.end_flag = false;
            kernel_out_stream.write(little_write_burst);
            little_idx++;
            total_length--;
        } else if (big_kernel_out_stream.read_nb(big_tmp_prop_pkt)) {
            in_write_burst_w_dst_pkt_t big_write_burst;
            big_write_burst.data = big_tmp_prop_pkt.data;
            big_write_burst.dest_addr = big_idx;
            big_write_burst.end_flag = false;
            kernel_out_stream.write(big_write_burst);
            big_idx++;
            total_length--;
        }
    }
}

static void
apply_func(bus_word_t *node_props,
           hls::stream<in_write_burst_w_dst_pkt_t> &write_burst_stream,
           hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream) {
APPLY_LOOP:
    while (true) {
        in_write_burst_w_dst_pkt_t in_pkt = write_burst_stream.read();
        if (in_pkt.end_flag) {
            write_burst_w_dst_pkt_t end_pkt;
            end_pkt.last = true;
            kernel_out_stream.write(end_pkt);
            break;
        }

        uint32_t dest_addr = in_pkt.dest_addr;
        bus_word_t ori_props = node_props[dest_addr];
        bus_word_t new_props;

        write_burst_w_dst_pkt_t out_pkt;
        out_pkt.dest = dest_addr;
        out_pkt.last = false;

        for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
            ap_fixed_pod_t update = in_pkt.data.range(31 + (i << 5), (i << 5));
            ap_fixed_pod_t old = ori_props.range(31 + (i << 5), (i << 5));
            ap_fixed_pod_t new_prop = (old < update) ? old : update;
            new_props.range(31 + (i << 5), (i << 5)) = new_prop;
        }

        out_pkt.data = new_props;
        kernel_out_stream.write(out_pkt);
    }
}

extern "C" void
apply_kernel(bus_word_t *node_props, uint32_t little_kernel_length,
             uint32_t big_kernel_length, uint32_t little_kernel_st_offset,
             uint32_t big_kernel_st_offset,
             hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
             hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
             hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = little_kernel_length bundle = control
#pragma HLS INTERFACE s_axilite port = big_kernel_length bundle = control
#pragma HLS INTERFACE s_axilite port = little_kernel_st_offset bundle = control
#pragma HLS INTERFACE s_axilite port = big_kernel_st_offset bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    hls::stream<in_write_burst_w_dst_pkt_t> write_burst_stream;
#pragma HLS STREAM variable = write_burst_stream depth = 16

    merge_big_little_writes(little_kernel_out_stream, big_kernel_out_stream,
                            write_burst_stream, little_kernel_length,
                            big_kernel_length, little_kernel_st_offset,
                            big_kernel_st_offset);
    apply_func(node_props, write_burst_stream, kernel_out_stream);
}
```

`scripts/kernel/big_merger.cpp`:

```cpp
#include "shared_kernel_params.h"

void merge_big_kernels(hls::stream<write_burst_pkt_t> &big_kernel_1_out_stream,
                       hls::stream<write_burst_pkt_t> &big_kernel_2_out_stream,
                       hls::stream<write_burst_pkt_t> &big_kernel_3_out_stream,
                       hls::stream<write_burst_pkt_t> &kernel_out_stream) {
    write_burst_pkt_t tmp_prop_pkt[BIG_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = tmp_prop_pkt dim = 0 complete

    bool process_flag[BIG_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = process_flag dim = 0 complete

    for (int i = 0; i < BIG_MERGER_LENGTH; i++) {
#pragma HLS unroll
        process_flag[i] = 0;
    }

    bus_word_t merged_write_burst;

    write_burst_pkt_t one_write_burst;

    uint32_t outer_idx = 0;

    ap_fixed_pod_t tmp_prop_arrary[16];
#pragma HLS ARRAY_PARTITION variable = tmp_prop_arrary dim = 0 complete

    distance_t max_val = (distance_t)(16384.0);
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);

merge_tmp_prop_big_krnls:
    while (true) {
#pragma HLS pipeline style = flp

        if (!process_flag[0])
            process_flag[0] = big_kernel_1_out_stream.read_nb(tmp_prop_pkt[0]);
        if (!process_flag[1])
            process_flag[1] = big_kernel_2_out_stream.read_nb(tmp_prop_pkt[1]);
        if (!process_flag[2])
            process_flag[2] = big_kernel_3_out_stream.read_nb(tmp_prop_pkt[2]);

        bool merge_flag =
            process_flag[0] & process_flag[1] & process_flag[2] & 1;

        if (merge_flag) {
            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                tmp_prop_arrary[i] = max_pod;
            }

            for (int i = 0; i < BIG_MERGER_LENGTH; i++) {
#pragma HLS UNROLL
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    ap_fixed_pod_t update =
                        tmp_prop_pkt[i].data.range(31 + (j << 5), (j << 5));
                    tmp_prop_arrary[j] =
                        (tmp_prop_arrary[j] < update || update == 0x0)
                            ? tmp_prop_arrary[j]
                            : update;
                }
            }

            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                merged_write_burst.range(31 + (i << 5), (i << 5)) =
                    tmp_prop_arrary[i];
            }

            one_write_burst.data = merged_write_burst;
            kernel_out_stream.write(one_write_burst);

            for (int i = 0; i < BIG_MERGER_LENGTH; i++) {
#pragma HLS unroll
                process_flag[i] = 0;
            }
        }
    }
}

extern "C" void
big_merger(hls::stream<write_burst_pkt_t> &big_kernel_1_out_stream,
           hls::stream<write_burst_pkt_t> &big_kernel_2_out_stream,
           hls::stream<write_burst_pkt_t> &big_kernel_3_out_stream,
           hls::stream<write_burst_pkt_t> &kernel_out_stream) {

#pragma HLS interface ap_ctrl_none port = return

#pragma HLS DATAFLOW

    merge_big_kernels(big_kernel_1_out_stream, big_kernel_2_out_stream,
                      big_kernel_3_out_stream, kernel_out_stream);
}

```

`scripts/kernel/graphyflow_big.cpp`:

```cpp
#include "graphyflow_big.h"

static void src_id_loader(const bus_word_t *node_ids_ddr,
                          hls::stream<node_id_burst_t> &src_id_burst_stream_1,
                          hls::stream<node_id_burst_t> &src_id_burst_stream_2,
                          int32_t num_nodes) {
    const int num_ids_per_word = AXI_BUS_WIDTH / NODE_ID_BITWIDTH;
    const int num_wide_reads =
        (num_nodes + num_ids_per_word - 1) / num_ids_per_word;

    int nodes_read = 0;
    int burst_idx = 0;
    node_id_burst_t burst1, burst2;
#pragma HLS ARRAY_PARTITION variable = burst1.data complete dim = 0
#pragma HLS ARRAY_PARTITION variable = burst2.data complete dim = 0
LOOP_SIL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 2
        bus_word_t wide_word = node_ids_ddr[i];

    LOOP_SIL_UNPACK:
        for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
            if (nodes_read + j < num_nodes) {
                node_id_t cur_id = wide_word.range(
                    (j + 1) * NODE_ID_BITWIDTH - 1, j * NODE_ID_BITWIDTH);
                burst1.data[j] = cur_id;
                // printf("Loaded node ID %d at burst %d, position %d\n",
                // (int)cur_id, burst_idx, j); fflush(NULL);
            }
        }
        bool burst2_valid = false;
        for (int j = 8; j < 16; j++) {
#pragma HLS UNROLL
            if (nodes_read + j < num_nodes) {
                burst2.data[j - 8] = wide_word.range(
                    (j + 1) * NODE_ID_BITWIDTH - 1, j * NODE_ID_BITWIDTH);
                burst2_valid |= true;
                // printf("Loaded node ID %d at burst %d, position %d\n",
                // (int)burst2.data[j - 8], burst_idx + 1, j - 8); fflush(NULL);
            }
        }
        src_id_burst_stream_1.write(burst1);
        src_id_burst_stream_2.write(burst1);
        if (burst2_valid) {
            src_id_burst_stream_1.write(burst2);
            src_id_burst_stream_2.write(burst2);
        }
        nodes_read += num_ids_per_word;
    }
}

static void
edge_descriptor_loader(const bus_word_t *edge_props_ddr,
                       hls::stream<node_id_burst_t> &stream_src_ids,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int32_t num_edges) {
    const int bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
    const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
    const int num_wide_reads =
        (num_edges + edges_per_word - 1) / edges_per_word;

    int edges_read = 0;
    edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
    edge_batch.end_pos = 0;

    node_id_burst_t src_id_burst;
#pragma HLS ARRAY_PARTITION variable = src_id_burst.data complete dim = 0

#if (NODE_ID_BITWIDTH == 32) && (WEIGHT_BITWIDTH == 32)
LOOP_EDL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = edge_props_ddr[i];
    LOOP_EDL_UNPACK:
        for (int j = 0; j < edges_per_word; j++) {
#pragma HLS UNROLL
            if (edges_read + j < num_edges) {
                ap_uint<bits_per_edge> packed_edge = wide_word.range(
                    (j + 1) * bits_per_edge - 1, j * bits_per_edge);
                edge_t edge;
                node_id_t src_id;
                edge.dst_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
                edge.src_id =
                    packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);
                src_id = edge.src_id;

                edge_batch.edges[j] = edge;
                src_id_burst.data[j] = src_id;
            }
        }
        stream_src_ids.write(src_id_burst);
        edges_read += edges_per_word;
        edge_batch.end_pos = (edges_read <= num_edges)
                                 ? edges_per_word
                                 : (num_edges % edges_per_word);
        edge_stream.write(edge_batch);
        edge_batch.end_pos = 0;
    }
#else
// Add support for other bitwidth combinations if needed.
#error                                                                         \
    "edge_descriptor_loader currently only supports 32-bit node_id and 32-bit weight."
#endif
}

ap_uint<4> count_end_ones(ap_uint<PE_NUM> valid_mask) {
#pragma HLS INLINE
    ap_uint<4> count = 0;
    switch (valid_mask) {
    case 0:
        count = 0;
        break;
    case 1:
        count = 1;
        break;
    case 3:
        count = 2;
        break;
    case 7:
        count = 3;
        break;
    case 15:
        count = 4;
        break;
    case 31:
        count = 5;
        break;
    case 63:
        count = 6;
        break;
    case 127:
        count = 7;
        break;
    case 255:
        count = 8;
        break;
    default:
        break;
    }
    return count;
}

static void
dist_req_packer(hls::stream<node_id_burst_t> &src_id_burst_stream,
                hls::stream<distance_req_pack_t> &distance_req_pack_stream,
                int32_t num_nodes) {

    const int max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_idx_max = 0;

LOOP_DRP_SEND_REQ:
    for (int32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         node_burst_idx += 1) {
#pragma HLS PIPELINE II = 1
        ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cache_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cache_idx complete dim = 0
        node_id_burst_t node_id_burst = src_id_burst_stream.read();
#pragma HLS ARRAY_PARTITION variable = node_id_burst.data complete dim = 0
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cache_idx[pe_idx] = node_id_burst.data[pe_idx] >> LOG_DIST_PER_WORD;
            // printf("PE %d requests node ID %d (cache idx %d)\n", pe_idx,
            // (int)node_id_burst.data[pe_idx], (int)cache_idx[pe_idx]);
            // fflush(NULL);
        }

        ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cache_idx_diffs[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cache_idx_diffs complete dim = 0
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cache_idx_diffs[pe_idx] = cache_idx[pe_idx] - last_idx_max;
            // printf("PE %d cache idx diff: %d\n", pe_idx,
            // (int)cache_idx_diffs[pe_idx]); fflush(NULL);
        }

        // if not all diffs are zero, send a req_pack
        if (cache_idx_diffs[PE_NUM - 1]) {
            ap_uint<PE_NUM> valid_mask;
            for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
                if (cache_idx_diffs[pe_idx] == 0) {
                    valid_mask[pe_idx] = 1;
                } else {
                    valid_mask[pe_idx] = 0;
                }
            }

            ap_uint<4> num_unread = count_end_ones(valid_mask);
            // printf("Packing req for %d unread PEs\n", (int)num_unread);
            // fflush(NULL);

            distance_req_pack_t req_pack;
#pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
            req_pack.offset = num_unread;
            req_pack.end_flag = false;

            for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
                req_pack.idx[pe_idx] = cache_idx[pe_idx];
                // printf("Req pack PE %d node ID: %d\n", pe_idx,
                // (int)req_pack.node_ids[pe_idx]); fflush(NULL);
            }

            distance_req_pack_stream.write(req_pack);
        }

        last_idx_max = cache_idx[PE_NUM - 1];
    }

    distance_req_pack_t end_req_pack;
    end_req_pack.end_flag = true;
    end_req_pack.offset = 7;
    distance_req_pack_stream.write(end_req_pack);
}

static void
cacheline_req_sender(hls::stream<distance_req_pack_t> &distance_req_pack_stream,
                     hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
                     int32_t memory_offset) {

    cacheline_request_pkt_t cache_req;
    cache_req.last = false;
    cache_req.data = memory_offset;
    cache_req.dest = 0;
    cacheline_req_stream.write(cache_req);

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cacheline_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cacheline_idx complete dim = 0

LOOP_SEND_CACHE_REQ:
    while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = cacheline_idx inter false
        distance_req_pack_t req_pack = distance_req_pack_stream.read();
#pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cacheline_idx[pe_idx] = req_pack.idx[pe_idx];
        }

        {
        LOOP_SEND_CACHE_REQ_INNER:
            for (ap_uint<4> i = req_pack.offset; i < PE_NUM; i++) {
#pragma HLS PIPELINE II = 1 rewind
#pragma HLS unroll factor = 1
                cache_req.data = cacheline_idx[i] + memory_offset;
                cache_req.dest = i;
                cache_req.last = req_pack.end_flag;
                cacheline_req_stream.write(cache_req);
                // printf("Sent cacheline req for idx %d to PE %d\n",
                // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
            }
        }

        if (req_pack.end_flag) {
            break;
        }
    }
}

// --- 1. Memory Helper Functions ---
// --- MODIFIED: Reads 512-bit words and unpacks 24-bit distance values.
// static void node_property_loader(
//     const bus_word_t *node_distances_ddr,
//     hls::stream<cacheline_req_t> &cacheline_req_stream,
//     hls::stream<cacheline_resp_t> &cacheline_resp_stream,
//     hls::stream<node_distance_burst_t> &node_distance_burst_stream,
//     int32_t num_nodes) {

//     ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx = -1;
//     bus_word_t last_cacheline;
//     cacheline_resp_t cache_resp;
//     bool end_flag_get = false;

//     // Stream 0
// LOOP_NPL_S0_READ:
//     while (true) {
// #pragma HLS PIPELINE II = 1
//         if (!cacheline_req_stream.empty()) {
//             // printf("Waiting for cacheline request...\n");fflush(NULL);
//             cacheline_req_t cache_req = cacheline_req_stream.read();
//             // printf("Received cacheline request for idx %d from PE %d\n",
//             // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
//             if (cache_req.end_flag) {
//                 cache_resp.end_flag = true;
//                 end_flag_get = true;
//             } else {
//                 cache_resp.end_flag = false;
//                 if (cache_req.idx == last_cache_idx) {
//                     cache_resp.data = last_cacheline;
//                 } else {
//                     cache_resp.data = node_distances_ddr[cache_req.idx];
//                 }
//             }

//             last_cacheline = cache_resp.data;
//             last_cache_idx = cache_req.idx;
//             cache_resp.target_pe = cache_req.target_pe;
//             cacheline_resp_stream.write(cache_resp);
//             // printf("Sent cacheline response for idx %d to PE %d\n",
//             // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
//             if (end_flag_get) {
//                 break;
//             }
//         }
//     }

//     // Stream 1
//     int nodes_read_s1 = 0;
//     int burst_idx1 = 0, burst_idx2 = 0;
//     // printf("Loading node distances for %d nodes (%d wide reads)\n",
//     // num_nodes, num_wide_reads); fflush(NULL);
//     const int num_ids_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
//     const int num_wide_reads =
//         (num_nodes + num_ids_per_word - 1) / num_ids_per_word;
// LOOP_NPL_S1_READ:
//     for (int i = 0; i < num_wide_reads; i++) {
// #pragma HLS PIPELINE II = 1
//         bus_word_t wide_word = node_distances_ddr[i];
//         node_distance_burst_t burst;

//     LOOP_NPL_S1_UNPACK:
//         for (int j = 0; j < DBL_PE_NUM; j++) {
// #pragma HLS UNROLL
//             if (nodes_read_s1 + j < num_nodes) {
//                 burst.data[j] = wide_word.range((j + 1) * DISTANCE_BITWIDTH -
//                 1,
//                                                 j * DISTANCE_BITWIDTH);
//             }
//         }
//         nodes_read_s1 += DBL_PE_NUM;
//         node_distance_burst_stream.write(burst);
//     }
// }

static void node_prop_resp_receiver(
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
    hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM]) {

    cacheline_response_pkt_t cache_resp = cacheline_resp_stream.read();
    bus_word_t first_line = cache_resp.data;
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        cacheline_streams[pe_idx].write(first_line);
    }

LOOP_RECEIVE_CACHE_RESP:
    while (true) {
#pragma HLS PIPELINE II = 1
        if (cacheline_resp_stream.read_nb(cache_resp)) {
            if (cache_resp.last) {
                break;
            }
            bus_word_t resp_line = cache_resp.data;
            ap_uint<8> target_pe = cache_resp.dest;
            cacheline_streams[target_pe].write(resp_line);
        }
    }
}

ap_fixed_pod_t get_val_from_bus(const bus_word_t bus, int offset) {
#pragma HLS INLINE
    switch (offset) {
    case 0:
        return bus.range(31, 0);
    case 1:
        return bus.range(63, 32);
    case 2:
        return bus.range(95, 64);
    case 3:
        return bus.range(127, 96);
    case 4:
        return bus.range(159, 128);
    case 5:
        return bus.range(191, 160);
    case 6:
        return bus.range(223, 192);
    case 7:
        return bus.range(255, 224);
    case 8:
        return bus.range(287, 256);
    case 9:
        return bus.range(319, 288);
    case 10:
        return bus.range(351, 320);
    case 11:
        return bus.range(383, 352);
    case 12:
        return bus.range(415, 384);
    case 13:
        return bus.range(447, 416);
    case 14:
        return bus.range(479, 448);
    case 15:
        return bus.range(511, 480);
    default:
        return 0;
    }
}

static void
merge_node_props(hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM],
                 hls::stream<edge_descriptor_batch_t> &edge_stream,
                 //  hls::stream<node_id_burst_t> &src_id_burst_stream,
                 hls::stream<update_tuple_t> &edge_batch_stream,
                 uint32_t edge_num) {
    bus_word_t last_cacheline[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = last_cacheline complete dim = 0
    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = last_cache_idx complete dim = 0

// Init first cacheline for each PE
LOOP_INIT_CACHELINE:
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        last_cacheline[pe_idx] = cacheline_streams[pe_idx].read();
        last_cache_idx[pe_idx] = 0;
    }

    const uint32_t scatter_size = (edge_num + PE_NUM - 1) / PE_NUM;
    distance_t real_edge_weight =
        1.0; // All edge weights are 1.0 in unweighted graph
    const ap_fixed_pod_t edge_weight = (*reinterpret_cast<ap_fixed_pod_t *>(
        &real_edge_weight)); // All edge weights are 1.0 in unweighted graph
// printf("Merging node properties for %d edges (%d scatter batches)\n",
// edge_num, scatter_size); fflush(NULL);
LOOP_SCATTER_EDGES:
    for (int32_t edge_batch_idx = 0; edge_batch_idx < scatter_size;
         edge_batch_idx++) {
#pragma HLS PIPELINE II = 1
        edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
        edge_batch = edge_stream.read();
        //         node_id_burst_t src_id_burst = src_id_burst_stream.read();
        // #pragma HLS ARRAY_PARTITION variable = src_id_burst.data complete dim
        // = 0
        update_tuple_t out_batch;
#pragma HLS ARRAY_PARTITION variable = out_batch.node_id complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_batch.prop complete dim = 0
        out_batch.end_flag = false;
        out_batch.end_pos = edge_batch.end_pos;
        bus_word_t cur_last_cacheline;
        ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cur_last_cache_idx;
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cacheline_idx =
                (edge_batch.edges[pe_idx].src_id >> LOG_DIST_PER_WORD);
            uint32_t offset =
                (edge_batch.edges[pe_idx].src_id & (DIST_PER_WORD - 1));
            if (pe_idx < edge_batch.end_pos) {
                bus_word_t cacheline;
                if (cacheline_idx == last_cache_idx[pe_idx]) {
                    cacheline = last_cacheline[pe_idx];
                } else {
                    cacheline = cacheline_streams[pe_idx].read();
                }

                // ap_fixed_pod_t prop = cacheline.range(
                //     31 + (offset << 5), offset << 5);
                ap_fixed_pod_t prop = get_val_from_bus(cacheline, offset);

                // out_batch.src_distances[pe_idx] = prop;
                // out_batch.weights[pe_idx] = edge_weight;
                // out_batch.dsts[pe_idx] = edge_batch.edges[pe_idx].dst_id;
                out_batch.node_id[pe_idx] = edge_batch.edges[pe_idx].dst_id;
                out_batch.prop[pe_idx] = (prop + edge_weight);

                // distance_t real_prop = *reinterpret_cast<distance_t
                // *>(&prop); printf(
                //     "PE %d edge src_id %d dst_id %d: loaded prop %.3f, "
                //     "updated prop "
                //     "%.3f\n",
                //     (int)pe_idx, (int)edge_batch.edges[pe_idx].src_id,
                //     (int)edge_batch.edges[pe_idx].dst_id, (float)real_prop,
                //     (float)(real_prop + real_edge_weight));

                if (pe_idx == PE_NUM - 1) {
                    cur_last_cacheline = cacheline;
                    cur_last_cache_idx = cacheline_idx;
                }
            }
        }
        edge_batch_stream.write(out_batch);
        // printf("Sent edge batch with %d entries\n", (int)out_batch.end_pos);
        // fflush(NULL);
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            last_cacheline[pe_idx] = cur_last_cacheline;
            last_cache_idx[pe_idx] = cur_last_cache_idx;
        }
    }
    // Send end marker
    update_tuple_t end_batch;
    end_batch.end_flag = true;
    end_batch.end_pos = 0;
    edge_batch_stream.write(end_batch);
}

// static void node_property_responder(
//     hls::stream<node_distance_burst_t> &node_distance_burst_stream,
//     int32_t num_nodes, hls::stream<node_dist_batch_t> &all_distances_stream)
//     { node_dist_batch_t dist_batch;
// #pragma HLS ARRAY_PARTITION variable = dist_batch.data complete dim = 0
//     dist_batch.end_flag = false;
//     int32_t nodes_read = 0;
//     const int num_reads = (num_nodes + DBL_PE_NUM - 1) / DBL_PE_NUM;
//     const int for_compare = num_nodes - DBL_PE_NUM;

// LOOP_FOR_14:
//     for (int32_t read_idx = 0; read_idx < num_reads; read_idx++) {
// #pragma HLS PIPELINE II = 1
//         // Read packet from stream
//         node_distance_burst_t node_dist_burst =
//             node_distance_burst_stream.read();

//     LOOP_FOR_13:
//         for (uint32_t pe_idx = 0; pe_idx < DBL_PE_NUM; pe_idx++) {
// #pragma HLS UNROLL
//             dist_batch.data[pe_idx] = node_dist_burst.data[pe_idx];
//         }
//         uint32_t maybe_remain_num = num_nodes - nodes_read;
//         uint32_t cur_node_read =
//             (nodes_read < for_compare) ? DBL_PE_NUM : maybe_remain_num;
//         dist_batch.end_pos = cur_node_read;
//         nodes_read += cur_node_read;
//         // printf("Writing distance batch with %d entries\n",
//         // (int)cur_node_read); fflush(NULL);
//         all_distances_stream.write(dist_batch);
//     }

//     dist_batch.end_flag = true;
//     dist_batch.end_pos = 0;
//     all_distances_stream.write(dist_batch);
// }

// --- REWRITTEN: New final_writeback function packs only distances (no node
// IDs) into 512-bit words. Node IDs are implicit: they are sequential from 0 to
// num_dsts-1.
// static void
// pack_distances_to_bus_words(hls::stream<internal_end_data_batch_t>
// &in_stream,
//                             hls::stream<write_burst_pkt_t> &output_stream) {
//     bus_word_t word;
//     int pkt_idx = 0;

// LOOP_PACK_TO_BUS:
//     while (true) {
// #pragma HLS PIPELINE II = 1
//         internal_end_data_batch_t in_batch = in_stream.read();
// #pragma HLS ARRAY_PARTITION variable = in_batch.data complete dim = 0

//         if (in_batch.end_pos == 0 && in_batch.end_flag) {
//             break;
//         }

//     LOOP_PACK_BATCH:
//         for (int i = 0; i < DBL_PE_NUM; i++) {
// #pragma HLS UNROLL
//             ap_fixed_pod_t distance = in_batch.data[i];
//             word.range((i + 1) * DISTANCE_BITWIDTH - 1, i *
//             DISTANCE_BITWIDTH) =
//                 distance;
//         }

//         write_burst_pkt_t pkt;
//         pkt.data = word;
//         pkt.dest = pkt_idx;
//         pkt.last = false;
//         pkt_idx++;

//         output_stream.write(pkt);

//         if (in_batch.end_flag) {
//             break;
//         }
//     }

//     write_burst_pkt_t pkt;
//     pkt.last = true;
//     output_stream.write(pkt);
// }

// Write bus words from stream to DDR memory
// Writes exactly the number of words needed to cover dst_num distances
// static void write_bus_words_to_ddr(hls::stream<bus_word_t> &in_bus_stream,
//                                    bus_word_t *out_ddr, int32_t dst_num) {
//     const int dists_per_word =
//         AXI_BUS_WIDTH / DISTANCE_BITWIDTH; // 16 distances per 512-bit word
//     int total_words = (dst_num + dists_per_word - 1) / dists_per_word;
//     int word_idx = 0;

// LOOP_WRITE_TO_DDR:
//     while (true) {
// #pragma HLS PIPELINE II = 1
//         if (!in_bus_stream.empty()) {
//             bus_word_t word = in_bus_stream.read();
//             out_ddr[word_idx] = word;
//             // printf("Wrote bus word %d to DDR.\n", word_idx); fflush(NULL);
//             // printf("Total words to write: %d\n", total_words);
//             fflush(NULL); word_idx++; if (word_idx >= total_words) {
//                 break;
//             }
//         }
//     }
// }

// --- 2. Utility Network Functions ---

static void
demux_1(hls::stream<update_tuple_t> &in_batch_stream,
        hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]) {
    update_tuple_t in_batch;
#pragma HLS ARRAY_PARTITION variable = in_batch.node_id complete dim = 0
#pragma HLS ARRAY_PARTITION variable = in_batch.prop complete dim = 0
LOOP_WHILE_22:
    while (true) {
#pragma HLS PIPELINE
        in_batch = in_batch_stream.read();
        net_wrapper_kt_pair_105_t_t wrapper_data;
#pragma HLS ARRAY_PARTITION variable = wrapper_data.node_id complete dim = 0
#pragma HLS ARRAY_PARTITION variable = wrapper_data.prop complete dim = 0
    LOOP_FOR_20:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if ((i < in_batch.end_pos)) {
                wrapper_data.node_id = in_batch.node_id[i];
                wrapper_data.prop = in_batch.prop[i];
                wrapper_data.end_flag = false;
                out_streams[i].write(wrapper_data);
            }
        }
        if (in_batch.end_flag) {
            break;
        }
    }
    // Propagate end_flag to all output streams
    net_wrapper_kt_pair_105_t_t end_wrapper;
    end_wrapper.end_flag = true;
LOOP_FOR_21:
    for (uint32_t i = 0; i < 8; i++) {
#pragma HLS UNROLL
        out_streams[i].write(end_wrapper);
    }
}

static void sender_2(int32_t i, hls::stream<net_wrapper_kt_pair_105_t_t> &in1,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &in2,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out1,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out2,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out3,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out4) {
#pragma HLS function_instantiate variable = i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
LOOP_WHILE_23:
    while (true) {
#pragma HLS PIPELINE II = 1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_105_t_t data1;
            data1 = in1.read();
            if ((!data1.end_flag)) {
                if (((data1.node_id >> i) & 1)) {
                    out2.write(data1);
                } else {
                    out1.write(data1);
                }
            } else {
                in1_end_flag = true;
            }
        }
        if ((!in2.empty())) {
            net_wrapper_kt_pair_105_t_t data2;
            data2 = in2.read();
            if ((!data2.end_flag)) {
                if (((data2.node_id >> i) & 1)) {
                    out4.write(data2);
                } else {
                    out3.write(data2);
                }
            } else {
                in2_end_flag = true;
            }
        }
        if ((in1_end_flag & in2_end_flag)) {
            net_wrapper_kt_pair_105_t_t data;
            data.end_flag = true;
            out1.write(data);
            out2.write(data);
            out3.write(data);
            out4.write(data);
            in1_end_flag = false;
            in2_end_flag = false;
            break;
        }
    }
}

static void receiver_2(int32_t i,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &out1,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &out2,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in1,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in2,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in3,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in4) {
#pragma HLS function_instantiate variable = i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
    bool in3_end_flag = false;
    bool in4_end_flag = false;
LOOP_WHILE_24:
    while (true) {
#pragma HLS PIPELINE II = 1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in1.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in1_end_flag = true;
            }
        } else if ((!in3.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in3.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in3_end_flag = true;
            }
        }
        if ((!in2.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in2.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in2_end_flag = true;
            }
        } else if ((!in4.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in4.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in4_end_flag = true;
            }
        }
        if ((((in1_end_flag & in2_end_flag) & in3_end_flag) & in4_end_flag)) {
            net_wrapper_kt_pair_105_t_t data;
            data.end_flag = true;
            out1.write(data);
            out2.write(data);
            break;
        }
    }
}

static void switch2x2_2(int32_t i,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &in1,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &in2,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &out1,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &out2) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_1;
#pragma HLS STREAM variable = l1_1 depth = 2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_2;
#pragma HLS STREAM variable = l1_2 depth = 2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_3;
#pragma HLS STREAM variable = l1_3 depth = 2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_4;
#pragma HLS STREAM variable = l1_4 depth = 2
    sender_2(i, in1, in2, l1_1, l1_2, l1_3, l1_4);
    receiver_2(i, out1, out2, l1_1, l1_2, l1_3, l1_4);
}

static void
omega_switch_2(hls::stream<net_wrapper_kt_pair_105_t_t> (&in_streams)[8],
               hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_0[8];
#pragma HLS STREAM variable = stream_stage_0 depth = 2
#pragma HLS ARRAY_PARTITION variable = stream_stage_0 complete dim = 0
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_1[8];
#pragma HLS STREAM variable = stream_stage_1 depth = 2
#pragma HLS ARRAY_PARTITION variable = stream_stage_1 complete dim = 0
    switch2x2_2(2, in_streams[0], in_streams[1], stream_stage_0[0],
                stream_stage_0[1]);
    switch2x2_2(2, in_streams[2], in_streams[3], stream_stage_0[2],
                stream_stage_0[3]);
    switch2x2_2(2, in_streams[4], in_streams[5], stream_stage_0[4],
                stream_stage_0[5]);
    switch2x2_2(2, in_streams[6], in_streams[7], stream_stage_0[6],
                stream_stage_0[7]);
    switch2x2_2(1, stream_stage_0[0], stream_stage_0[4], stream_stage_1[0],
                stream_stage_1[1]);
    switch2x2_2(1, stream_stage_0[1], stream_stage_0[5], stream_stage_1[2],
                stream_stage_1[3]);
    switch2x2_2(1, stream_stage_0[2], stream_stage_0[6], stream_stage_1[4],
                stream_stage_1[5]);
    switch2x2_2(1, stream_stage_0[3], stream_stage_0[7], stream_stage_1[6],
                stream_stage_1[7]);
    switch2x2_2(0, stream_stage_1[0], stream_stage_1[4], out_streams[0],
                out_streams[1]);
    switch2x2_2(0, stream_stage_1[1], stream_stage_1[5], out_streams[2],
                out_streams[3]);
    switch2x2_2(0, stream_stage_1[2], stream_stage_1[6], out_streams[4],
                out_streams[5]);
    switch2x2_2(0, stream_stage_1[3], stream_stage_1[7], out_streams[6],
                out_streams[7]);
}

// --- 3. DFIR Component Functions ---
// static void
// Reduc_105_pre_process(hls::stream<edge_batch_t> &response_to_318,
//                       hls::stream<update_tuple_t> &reduce_105_z2d_pair) {
//     edge_batch_t edge_batch_data;
// #pragma HLS ARRAY_PARTITION variable = edge_batch_data.dsts complete dim = 0
// #pragma HLS ARRAY_PARTITION variable = \
//     edge_batch_data.src_distances complete dim = 0
// #pragma HLS ARRAY_PARTITION variable = edge_batch_data.weights complete dim =
// 0
//     update_tuple_t out_batch_data;
// #pragma HLS ARRAY_PARTITION variable = out_batch_data.node_id complete dim =
// 0 #pragma HLS ARRAY_PARTITION variable = out_batch_data.prop complete dim = 0
//     bool end_flag;
// LOOP_WHILE_26:
//     while (true) {
// #pragma HLS PIPELINE
//         edge_batch_data = response_to_318.read();
//     LOOP_FOR_25:
//         for (uint32_t i = 0; i < PE_NUM; i++) {
// #pragma HLS UNROLL
//             kt_pair_105_t kt_pair;
//             kt_pair.key = edge_batch_data.dsts[i];
//             kt_pair.transform.node_id = edge_batch_data.dsts[i];
//             ap_fixed_pod_t new_dist;
//             // distance_t lhs_68 = *reinterpret_cast<distance_t *>(
//             //     &edge_batch_data.src_distances[i]);
//             // distance_t rhs_68 =
//             //     *reinterpret_cast<distance_t
//             *>(&edge_batch_data.weights[i]);
//             // distance_t temp_BinOp_68_o_0_ap_result;
//             // temp_BinOp_68_o_0_ap_result = (lhs_68 + rhs_68);
//             // ap_fixed_pod_t fused_temp_BinOp_68_o_0 =
//             //     *reinterpret_cast<ap_fixed_pod_t *>(
//             //         &temp_BinOp_68_o_0_ap_result);
//             // Inlining Gathe_179
//             kt_pair.transform.prop =
//                 (edge_batch_data.src_distances[i] +
//                 edge_batch_data.weights[i]);
//             out_batch_data.data[i] = kt_pair;
//         }
//         out_batch_data.end_flag = edge_batch_data.end_flag;
//         out_batch_data.end_pos = edge_batch_data.end_pos;
//         reduce_105_z2d_pair.write(out_batch_data);
//         end_flag = edge_batch_data.end_flag;
//         if (end_flag) {
//             break;
//         }
//     }
// }

inline ap_fixed_pod_t get_raw_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> bits;
    switch (idx) {
    case 0:
        bits = word.range(DISTANCE_BITWIDTH - 1, 0);
        break;
    case 1:
        bits = word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH);
        break;
    case 2:
        bits =
            word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1));
        break;
    default:
        bits = 0;
        break;
    }
    return bits;
}

inline distance_t get_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_fixed_pod_t raw_val = get_raw_val(word, idx);
    distance_t val = *reinterpret_cast<distance_t *>(&raw_val);
    return val;
}

inline void set_val(reduce_word_t &word, int idx, distance_t val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits =
        *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&val);
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1)) =
            val_bits;
        break;
    default:
        break;
    }
}

inline void set_raw_val(reduce_word_t &word, int idx, ap_fixed_pod_t pod_val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits = pod_val;
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1)) =
            val_bits;
        break;
    default:
        break;
    }
}

// Single-PE aggregation function
// Handles initialization and aggregation for one PE
static void Reduc_105_unit_reduce_single_pe(
    hls::stream<net_wrapper_kt_pair_105_t_t> &kt_wrap_item_single,
    hls::stream<reduce_word_t> &pe_mem_out, int32_t pe_id, int32_t dst_num) {

    // --- Phase 1: Memory Declaration ---
    const int MEM_SIZE = (MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[MEM_SIZE];
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM
#pragma HLS dependence variable = prop_mem inter false

    // Latency-hiding cache for recently accessed URAM words
    reduce_word_t cache_data_buffer[L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    int32_t cache_addr_buffer[L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0

    // distance_t MAX_DISTANCE = (distance_t)(16384.0);
    // const ap_fixed_pod_t MAX_DISTANCE_POD =
    //     *reinterpret_cast<ap_fixed_pod_t *>(&MAX_DISTANCE);
    // const reduce_word_t MAX_REDUCE_WORD =
    //     (((reduce_word_t)MAX_DISTANCE_POD << DISTANCE_BITWIDTH) |
    //      ((reduce_word_t)MAX_DISTANCE_POD));

    const int32_t num_words =
        (dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
    const int32_t num_word_per_pe = (num_words + PE_NUM - 1) / PE_NUM;

    // --- Phase 2: Initialization ---
    // LOOP_INIT_MEM:
    //     for (int i = 0; i < MEM_SIZE; i++) {
    // #pragma HLS PIPELINE II = 1
    //         prop_mem[i] = MAX_REDUCE_WORD; // Initialize distances to max
    //     }

#ifdef EMULATION
    memset(prop_mem, 0, sizeof(reduce_word_t) * MEM_SIZE);
#endif

LOOP_INIT_CACHE_ADDR:
    for (int i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
        cache_addr_buffer[i] = -1; // Invalidate cache
    }

    // --- Phase 3: Aggregation Loop ---
    bool end_flag = false;

LOOP_AGGREGATE:
    while (true) {
#pragma HLS PIPELINE II = 1
        net_wrapper_kt_pair_105_t_t kt_elem;
        kt_elem = kt_wrap_item_single.read();
        if (kt_elem.end_flag) {
            break;
        }
        int32_t key = kt_elem.node_id >> LOG_PE_NUM;
        ap_fixed_pod_t incoming_dist_pod = kt_elem.prop;

        int32_t word_addr = (key >> 1);
        int32_t pack_idx = (key & 1);

        reduce_word_t current_word = prop_mem[word_addr];

        // Check cache first
        for (int i = L; i >= 0; --i) {
#pragma HLS UNROLL
            if (cache_addr_buffer[i] == word_addr) {
                current_word = cache_data_buffer[i];
                break;
            }
        }

        // Shift cache
        for (int i = 0; i < L; i++) {
#pragma HLS UNROLL
            cache_addr_buffer[i] = cache_addr_buffer[i + 1];
            cache_data_buffer[i] = cache_data_buffer[i + 1];
        }

        ap_fixed_pod_t old_dist_pod = get_raw_val(current_word, pack_idx);
        ap_fixed_pod_t new_dist_pod =
            (old_dist_pod < incoming_dist_pod && old_dist_pod != 0x0)
                ? old_dist_pod
                : incoming_dist_pod;

        set_raw_val(current_word, pack_idx, new_dist_pod);

        // Write back to URAM and update cache
        prop_mem[word_addr] = current_word;
        cache_addr_buffer[L] = word_addr;
        cache_data_buffer[L] = current_word;
    }

    // --- Phase 4: Stream out aggregated memory ---
LOOP_STREAM_OUT:
    for (int i = 0; i < num_word_per_pe; i++) {
#pragma HLS UNROLL factor = 1
        pe_mem_out.write(prop_mem[i]);
        prop_mem[i] = 0;
    }
}

float ap_fixed_to_float(ap_fixed_pod_t val) {
    return (float)*reinterpret_cast<distance_t *>(&val);
}

// Multi-PE drain function
// Collects aggregated data from all PEs and outputs final results
static void
Reduc_105_drain_multi_pe(hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM],
                         hls::stream<write_burst_pkt_t> &kernel_out_stream,
                         int32_t dst_num) {

    // --- Phase 2: High-Performance Drain Loop ---
    write_burst_pkt_t one_write_burst;
    one_write_burst.last = 0;

LOOP_DRAIN_ADDR:
    for (int32_t base_addr = 0; base_addr < dst_num;
         base_addr += (PE_NUM << 1)) {
#pragma HLS PIPELINE II = 1
    LOOP_FOR_57:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            reduce_word_t word = pe_mem_in[pe_idx].read();

            one_write_burst.data.range(31 + (pe_idx << 5), (pe_idx << 5)) =
                word.range(31, 0);
            one_write_burst.data.range(31 + (pe_idx << 5) + 256,
                                       (pe_idx << 5) + 256) =
                word.range(63, 32);

            // printf("Drained word from PE %d: lower=%f upper=%f\n", pe_idx,
            //        ap_fixed_to_float(word.range(31, 0)),
            //        ap_fixed_to_float(word.range(63, 32)));
        }
        kernel_out_stream.write(one_write_burst);
    }
}

// static void fused_op_294(hls::stream<internal_end_data_batch_t> &i_0,
//                          hls::stream<node_dist_batch_t> &i_1,
//                          hls::stream<internal_end_data_batch_t> &o_0) {
//     internal_end_data_batch_t in_batch_i_0;
//     node_dist_batch_t in_batch_i_1;
//     internal_end_data_batch_t out_batch_o_0;
//     bool end_flag;
//     uint8_t end_pos;
// LOOP_WHILE_59:
//     while (true) {
// #pragma HLS PIPELINE
//         in_batch_i_0 = i_0.read();
//         in_batch_i_1 = i_1.read();
//     LOOP_FOR_58:
//         for (uint32_t i = 0; i < DBL_PE_NUM; i++) {
// #pragma HLS UNROLL
//             // -- Inlining FusedOp fused_op_294 --
//             // Inlining BinOp_128
//             // ap_fixed_pod_t fused_temp_BinOp_128_o_0;
//             // distance_t lhs_128 =
//             //     *reinterpret_cast<distance_t
//             *>(&in_batch_i_0.data[i].prop);
//             // distance_t rhs_128 =
//             //     *reinterpret_cast<distance_t *>(&in_batch_i_1.data[i]);
//             // distance_t temp_BinOp_128_o_0_ap_result;
//             // temp_BinOp_128_o_0_ap_result =
//             //     (((lhs_128) < (rhs_128) ? lhs_128 : rhs_128));
//             // fused_temp_BinOp_128_o_0 = *reinterpret_cast<ap_fixed_pod_t
//             *>(
//             //     &temp_BinOp_128_o_0_ap_result);
//             // Inlining Gathe_288
//             out_batch_o_0.data[i].prop =
//                 ((in_batch_i_0.data[i].prop < in_batch_i_1.data[i])
//                      ? in_batch_i_0.data[i].prop
//                      : in_batch_i_1.data[i]);
//             out_batch_o_0.data[i].node_id = in_batch_i_0.data[i].node_id;
//             // -- End Inlining FusedOp fused_op_294 --
//         }
//         end_flag = in_batch_i_0.end_flag;
//         end_pos = in_batch_i_0.end_pos;
//         out_batch_o_0.end_flag = end_flag;
//         out_batch_o_0.end_pos = end_pos;
//         o_0.write(out_batch_o_0);
//         if (end_flag) {
//             break;
//         }
//     }
// }

static void graphyflow_big_dataflow(
    hls::stream<update_tuple_t> &input_to_demux,
    // hls::stream<node_dist_batch_t> &all_node_distances_to_343,
    hls::stream<write_burst_pkt_t> &kernel_out_stream, int32_t dst_num) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_d2o_pair[8];
#pragma HLS STREAM variable = reduce_105_d2o_pair depth = 16
#pragma HLS ARRAY_PARTITION variable = reduce_105_d2o_pair complete dim = 0
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_o2u_pair[8];
#pragma HLS STREAM variable = reduce_105_o2u_pair depth = 2
#pragma HLS ARRAY_PARTITION variable = reduce_105_o2u_pair complete dim = 0
    //     hls::stream<struct_ibu_14_t> intermediate_key;
    // #pragma HLS STREAM variable = intermediate_key depth = 4
    //     hls::stream<internal_end_data_batch_t> intermediate_transform;
    // #pragma HLS STREAM variable = intermediate_transform depth = 4
    //     hls::stream<struct_sbu_7_t> stream_o_0_273;
    // #pragma HLS STREAM variable = stream_o_0_273 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_236;
    // #pragma HLS STREAM variable = stream_o_0_236 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_1_237;
    // #pragma HLS STREAM variable = stream_o_1_237 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_2_238;
    // #pragma HLS STREAM variable = stream_o_2_238 depth = 4
    //     hls::stream<struct_ibu_14_t> stream_o_0_node_id_232;
    // #pragma HLS STREAM variable = stream_o_0_node_id_232 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_1_250;
    // #pragma HLS STREAM variable = stream_o_1_250 depth = 4
    //     hls::stream<internal_end_data_batch_t> stream_o_0_107;
    // #pragma HLS STREAM variable = stream_o_0_107 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_0_249;
    // #pragma HLS STREAM variable = stream_o_0_249 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_edge_src_distance_275;
    // #pragma HLS STREAM variable = stream_o_0_edge_src_distance_275 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_0_edge_dst_277;
    // #pragma HLS STREAM variable = stream_o_0_edge_dst_277 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_edge_weight_278;
    // #pragma HLS STREAM variable = stream_o_0_edge_weight_278 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_node_distance_300;
    // #pragma HLS STREAM variable = stream_o_0_node_distance_300 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_1_309;
    // #pragma HLS STREAM variable = stream_o_1_309 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_304;
    // #pragma HLS STREAM variable = stream_o_0_304 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_1_305;
    // #pragma HLS STREAM variable = stream_o_1_305 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_0_308;
    // #pragma HLS STREAM variable = stream_o_0_308 depth = 4
    // --- Function Calls (in topological order) ---
    // Memor_274(response_to_318, stream_o_0_edge_src_distance_275,
    //           stream_o_0_edge_dst_277, stream_o_0_edge_weight_278);
    // fused_op_269(stream_o_0_edge_src_distance_275, stream_o_0_edge_dst_277,
    //              stream_o_0_edge_weight_278, stream_o_0_273);
    // Scatt_234(stream_o_0_273, stream_o_0_236, stream_o_1_237,
    // stream_o_2_238); CopyC_247(stream_o_1_237, stream_o_0_249,
    // stream_o_1_250); Memor_231(stream_o_0_node_id_232, stream_o_1_250);
    // --- Start of Reduce Super-Block for Reduc_105 ---
    // Reduc_105_pre_process(response_to_318, reduce_105_z2d_pair);
    // stream_zipper_0(intermediate_key, intermediate_transform,
    //                 reduce_105_z2d_pair);
    demux_1(input_to_demux, reduce_105_d2o_pair);
    omega_switch_2(reduce_105_d2o_pair, reduce_105_o2u_pair);
    // Reduc_105_unit_reduce(reduce_105_o2u_pair, stream_o_0_107, dst_num);
    hls::stream<reduce_word_t> pe_mem_out_streams[PE_NUM];
#pragma HLS STREAM variable = pe_mem_out_streams depth = 4
LOOP_FOR_60:
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        Reduc_105_unit_reduce_single_pe(reduce_105_o2u_pair[pe_idx],
                                        pe_mem_out_streams[pe_idx], pe_idx,
                                        dst_num);
    }
    Reduc_105_drain_multi_pe(pe_mem_out_streams, kernel_out_stream, dst_num);
    // --- End of Reduce Super-Block for Reduc_105 ---
    // Scatt_302(stream_o_0_107, stream_o_0_304, stream_o_1_305);
    // CopyC_306(stream_o_1_305, stream_o_0_308, stream_o_1_309);
    // Memor_299(all_node_distances_to_343, stream_o_0_node_distance_300,
    // dst_num);
    // fused_op_294(stream_o_0_107, all_node_distances_to_343,
    //              internal_end_stream);
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void
graphyflow_big(const bus_word_t *edge_props, int32_t num_nodes,
               int32_t num_edges, int32_t dst_num, int32_t memory_offset,
               hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
               hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
               hls::stream<write_burst_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem0
// #pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem1
// #pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = edge_props
// #pragma HLS INTERFACE s_axilite port = node_props
// #pragma HLS INTERFACE s_axilite port = output
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = dst_num
#pragma HLS INTERFACE s_axilite port = memory_offset
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    // printf("GraphyFlow Big Kernel Started.\n");
    // fflush(NULL);

    // Streams for the new COO-style property loading
    hls::stream<node_id_burst_t> stream_src_ids;
#pragma HLS STREAM variable = stream_src_ids depth = 16
    //     hls::stream<node_id_burst_t> stream_src_ids_2;
    // #pragma HLS STREAM variable = stream_src_ids_2 depth = 16
    hls::stream<distance_req_pack_t> stream_dist_req;
#pragma HLS STREAM variable = stream_dist_req depth = 32
    //     hls::stream<cacheline_req_t> stream_cache_req;
    // #pragma HLS STREAM variable = stream_cache_req depth = 16
    //     hls::stream<cacheline_resp_t> stream_cache_resp;
    // #pragma HLS STREAM variable = stream_cache_resp depth = 16
    hls::stream<bus_word_t> stream_cachelines[PE_NUM];
#pragma HLS STREAM variable = stream_cachelines depth = 32
    // #pragma HLS ARRAY_PARTITION variable = stream_cachelines complete dim = 0

    // Existing streams
    //     hls::stream<node_distance_burst_t> node_distance_burst_stream;
    // #pragma HLS STREAM variable = node_distance_burst_stream depth = 16
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 32
    hls::stream<update_tuple_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 16
    //     hls::stream<node_dist_batch_t> stream_node_dist_data;
    // #pragma HLS STREAM variable = stream_node_dist_data depth = 16
    //     hls::stream<internal_end_data_batch_t> stream_result_data;
    // #pragma HLS STREAM variable = stream_result_data depth = 16

    // --- Data Loading ---
    // src_id_loader(src_ids, stream_src_ids_1, stream_src_ids_2, num_edges);
    edge_descriptor_loader(edge_props, stream_src_ids, edge_stream, num_edges);

    // --- New COO-style Source Property Loading Pipeline ---
    dist_req_packer(stream_src_ids, stream_dist_req, num_edges);
    // printf("Distance Request Packer Completed.\n");
    // fflush(NULL);
    cacheline_req_sender(stream_dist_req, cacheline_req_stream, memory_offset);
    // printf("Cacheline Request Sender Completed.\n");
    // fflush(NULL);
    // node_property_loader(node_props, stream_cache_req, stream_cache_resp,
    //                      node_distance_burst_stream, num_nodes);
    node_prop_resp_receiver(cacheline_resp_stream, stream_cachelines);
    merge_node_props(stream_cachelines, edge_stream, stream_edge_data,
                     num_edges);

    // --- Node Property Responder for Reduce Operation ---
    // node_property_responder(node_distance_burst_stream, num_nodes,
    // stream_node_dist_data);

    // --- Main Dataflow Processing ---
    graphyflow_big_dataflow(stream_edge_data, kernel_out_stream, dst_num);
    // printf("GraphyFlow Big Kernel Completed.\n");
    // fflush(NULL);

    // --- Final Writeback ---
    // final_writeback(stream_result_data, dst_num, output);
    //     hls::stream<bus_word_t> bus_word_stream;
    // #pragma HLS STREAM variable = bus_word_stream depth = 4
    // pack_distances_to_bus_words(stream_result_data, kernel_out_stream);
    // write_bus_words_to_ddr(bus_word_stream, output, dst_num);
}

```

`scripts/kernel/graphyflow_big.h`:

```h
#ifndef __GRAPHYFLOW_GRAPHYFLOW_BIG_H__
#define __GRAPHYFLOW_GRAPHYFLOW_BIG_H__

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PE_NUM 8
#define DBL_PE_NUM 16
#define LOG_PE_NUM 3
#ifdef EMULATION
#define MAX_NUM 512
#else
#define MAX_NUM 524288
#endif
#define L 4

// --- New Bitwidth Definitions for HLS Synthesis ---
#define NODE_ID_BITWIDTH 32
#define DISTANCE_BITWIDTH 32
#define DISTANCE_INTEGER_PART 16
#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH
#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART
#define OUT_END_MARKER_BITWIDTH 4
#define DIST_PER_WORD 16 // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = 512 / 32 = 16
#define LOG_DIST_PER_WORD                                                      \
    4 // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2(512 / 32) = log2(16) = 4

// --- New Memory Word and Bus Definitions ---
#define AXI_BUS_WIDTH 512

#define REDUCE_MEM_WIDTH 64
typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;

const int INFINITY_DIST = 16384;

// --- New Packing-related Constants ---
// Number of distances that can be packed into a single reduce memory word.
#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)

// --- Redefinition of Core Graph Types for HLS ---
// These typedefs override the standard integer types from common.h for
// synthesis.
typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;
typedef ap_uint<32> edge_id_t; // edge_id_t is not customized yet, keep as is.
typedef ap_uint<DISTANCE_BITWIDTH>
    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;
typedef ap_axiu<256, 0, 0, 0> node_dist_pkt_t;
typedef ap_axiu<512, 0, 0, 0> write_burst_pkt_t;
typedef ap_axiu<32, 0, 0, 8> cacheline_request_pkt_t;
typedef ap_axiu<512, 0, 0, 8> cacheline_response_pkt_t;
typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;

// --- Struct Type Definitions (UNCHANGED) ---
// The definitions of these structs remain the same, but the underlying
// types (node_id_t, ap_fixed_pod_t) are now custom-width, not uint32_t.
struct __attribute__((packed)) struct_ana_3_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
    ap_fixed_pod_t ele_2;
};

struct __attribute__((packed)) struct_abu_9_t {
    ap_fixed_pod_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_nbu_11_t {
    node_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ibu_14_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_an_15_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
};

struct __attribute__((packed)) struct_ebu_20_t {
    edge_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) node_with_prop_t {
    ap_fixed_pod_t prop;
    node_id_t node_id;
};

struct __attribute__((packed)) node_distance_cache_burst_t {
    ap_fixed_pod_t data[DIST_PER_WORD];
};

struct __attribute__((packed)) node_distance_burst_t {
    ap_fixed_pod_t data[DBL_PE_NUM];
};

struct __attribute__((packed)) node_id_burst_t {
    node_id_t data[PE_NUM];
};

struct __attribute__((packed)) distance_req_pack_t {
    node_id_t idx[PE_NUM];
    ap_uint<4> offset; // [offset, offset + PE_NUM) are valid
    bool end_flag;
};

struct __attribute__((packed)) cacheline_req_t {
    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> idx;
    ap_uint<4> target_pe;
    bool end_flag;
};

struct __attribute__((packed)) cacheline_resp_t {
    bus_word_t data;
    ap_uint<4> target_pe;
    bool end_flag;
};

struct __attribute__((packed)) edge_batch_t {
    ap_fixed_pod_t weights[PE_NUM];
    ap_fixed_pod_t src_distances[PE_NUM];
    node_id_t dsts[PE_NUM];
    int32_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) node_dist_batch_t {
    ap_fixed_pod_t data[DBL_PE_NUM];
    uint8_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    node_id_t id;
};

struct __attribute__((packed)) struct_sbu_7_t {
    struct_ana_3_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_17_t {
    struct_an_15_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) internal_end_data_batch_t {
    ap_fixed_pod_t data[DBL_PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_105_t {
    int32_t key;
    node_with_prop_t transform;
};

struct __attribute__((packed)) struct_nb_58_t {
    node_with_prop_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) edge_t {
    node_id_t src_id;
    node_id_t dst_id;
};

struct __attribute__((packed)) edge_descriptor_batch_t {
    edge_t edges[PE_NUM];
    int32_t end_pos;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

// struct __attribute__((packed)) struct_kbu_50_t {
//     kt_pair_105_t data[PE_NUM];
//     bool end_flag;
//     uint8_t end_pos;
// };

struct __attribute__((packed)) update_tuple_t {
    node_id_t node_id[PE_NUM];
    ap_fixed_pod_t prop[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_105_t_t {
    node_id_t node_id;
    ap_fixed_pod_t prop;
    bool end_flag;
};

// --- Function Prototypes ---
// static void node_property_loader(const int32_t* node_distances_ddr,
// hls::stream<node_distance_burst_t> &node_distance_burst_stream_0,
// hls::stream<node_distance_burst_t> &node_distance_burst_stream_1, int32_t
// num_nodes); static void edge_descriptor_loader(const edge_des_burst_t*
// edge_des_bursts, hls::stream<edge_descriptor_batch_t> &edge_stream, int32_t
// num_edges); static void src_offset_loader(const int32_t* src_offsets_ddr,
// hls::stream<int32_t> &src_offsets_stream, int32_t num_nodes); static void
// edge_property_loader_and_dispatcher(hls::stream<int32_t>
// &src_offsets_cache_stream, hls::stream<edge_descriptor_batch_t> &edge_stream,
// hls::stream<node_distance_burst_t> &node_distance_burst_stream, int32_t
// num_nodes, hls::stream<edge_batch_t> &response_stream); static void
// node_property_responder(hls::stream<node_distance_burst_t>
// &node_distance_burst_stream, int32_t num_nodes,
// hls::stream<node_dist_batch_t> &all_distances_stream); static void
// final_convert(hls::stream<internal_end_data_batch_t> &in_stream,
// hls::stream<KernelOutputBatch> &converted_stream); static void
// final_write(hls::stream<KernelOutputBatch> &converted_stream,
// KernelOutputBatch* out_o_0_342); static void
// Reduc_105_pre_process(hls::stream<struct_ibu_14_t> &i_global_data_0,
// hls::stream<struct_nbu_11_t> &i_global_data_1, hls::stream<struct_abu_9_t>
// &i_global_data_2, hls::stream<struct_abu_9_t> &i_global_data_3,
// hls::stream<struct_ibu_14_t> &intermediate_key,
// hls::stream<internal_end_data_batch_t> &intermediate_transform); static void
// Reduc_105_unit_reduce(hls::stream<net_wrapper_kt_pair_105_t_t>
// (&kt_wrap_item)[PE_NUM], hls::stream<internal_end_data_batch_t> &o_0); static
// void Scatt_234(hls::stream<struct_sbu_7_t> &i_0, hls::stream<struct_abu_9_t>
// &o_0, hls::stream<struct_nbu_11_t> &o_1, hls::stream<struct_abu_9_t> &o_2);
// static void Memor_231(hls::stream<struct_ibu_14_t> &o_0_node_id,
// hls::stream<struct_nbu_11_t> &i_0_node_id); static void
// CopyC_247(hls::stream<struct_nbu_11_t> &i_0, hls::stream<struct_nbu_11_t>
// &o_0, hls::stream<struct_nbu_11_t> &o_1); static void
// fused_op_269(hls::stream<struct_abu_9_t> &i_0, hls::stream<struct_nbu_11_t>
// &i_1, hls::stream<struct_abu_9_t> &i_2, hls::stream<struct_sbu_7_t> &o_0);
// static void Memor_274(hls::stream<edge_batch_t> &i_0_edge_id,
// hls::stream<struct_abu_9_t> &o_0_edge_src_distance,
// hls::stream<struct_nbu_11_t> &o_0_edge_dst, hls::stream<struct_abu_9_t>
// &o_0_edge_weight); static void Memor_299(hls::stream<node_dist_batch_t>
// &i_all_node_distances, hls::stream<struct_abu_9_t> &o_0_node_distance,
// hls::stream<struct_nbu_11_t> &i_0_node_id); static void
// Scatt_302(hls::stream<internal_end_data_batch_t> &i_0,
// hls::stream<struct_abu_9_t> &o_0, hls::stream<struct_nbu_11_t> &o_1); static
// void CopyC_306(hls::stream<struct_nbu_11_t> &i_0,
// hls::stream<struct_nbu_11_t> &o_0, hls::stream<struct_nbu_11_t> &o_1); static
// void fused_op_294(hls::stream<struct_abu_9_t> &i_0,
// hls::stream<struct_abu_9_t> &i_1, hls::stream<struct_nbu_11_t> &i_2,
// hls::stream<internal_end_data_batch_t> &o_0); static void
// memory_loader(int32_t instantiate_idx, const int32_t* src_offsets, const
// edge_des_burst_t* edge_des_bursts, const int32_t* node_distances, int32_t
// num_nodes, int32_t num_edges, hls::stream<edge_batch_t> &response_to_318,
// hls::stream<node_dist_batch_t> &all_node_distances_to_343); static void
// graphyflow_big_dataflow(hls::stream<edge_batch_t> &response_to_318,
// hls::stream<node_dist_batch_t> &all_node_distances_to_343,
// hls::stream<internal_end_data_batch_t> &internal_end_stream); static void
// final_writeback(int32_t instantiate_idx,
// hls::stream<internal_end_data_batch_t> &internal_end_stream,
// KernelOutputBatch* out_o_0_342);

// --- Top-Level Function Prototype ---
extern "C" void
graphyflow_big(const bus_word_t *edge_props, int32_t num_nodes,
               int32_t num_edges, int32_t dst_num, int32_t memory_offset,
               hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
               hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
               hls::stream<write_burst_pkt_t> &kernel_out_stream);
#endif // __GRAPHYFLOW_GRAPHYFLOW_BIG_H__

```

`scripts/kernel/graphyflow_little.cpp`:

```cpp
#include "graphyflow_little.h"

static void
edge_descriptor_loader(const bus_word_t *edge_props_ddr,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int32_t num_edges) {
    const int bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
    const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
    const int num_wide_reads =
        (num_edges + edges_per_word - 1) / edges_per_word;

    int edges_read = 0;
    edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
    edge_batch.end_pos = 0;

    node_id_burst_t src_id_burst;
#pragma HLS ARRAY_PARTITION variable = src_id_burst.data complete dim = 0

#if (NODE_ID_BITWIDTH == 32) && (WEIGHT_BITWIDTH == 32)
LOOP_EDL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = edge_props_ddr[i];
    LOOP_EDL_UNPACK:
        for (int j = 0; j < edges_per_word; j++) {
#pragma HLS UNROLL
            if (edges_read + j < num_edges) {
                ap_uint<bits_per_edge> packed_edge = wide_word.range(
                    (j + 1) * bits_per_edge - 1, j * bits_per_edge);
                edge_t edge;
                node_id_t src_id;
                edge.dst_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
                edge.src_id =
                    packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);
                src_id = edge.src_id;

                edge_batch.edges[j] = edge;
                src_id_burst.data[j] = src_id;
            }
        }
        edges_read += edges_per_word;
        edge_batch.end_pos = (edges_read <= num_edges)
                                 ? edges_per_word
                                 : (num_edges % edges_per_word);
        edge_stream.write(edge_batch);
        edge_batch.end_pos = 0;
    }
#else
// Add support for other bitwidth combinations if needed.
#error                                                                         \
    "edge_descriptor_loader currently only supports 32-bit node_id and 32-bit weight."
#endif
}

ap_fixed_pod_t get_val_from_bus(bus_word_t bus_data,
                                ap_uint<30> position_in_bus) {
#pragma HLS INLINE
    switch (position_in_bus) {
    case 0:
        return bus_data.range(31, 0);
    case 1:
        return bus_data.range(63, 32);
    case 2:
        return bus_data.range(95, 64);
    case 3:
        return bus_data.range(127, 96);
    case 4:
        return bus_data.range(159, 128);
    case 5:
        return bus_data.range(191, 160);
    case 6:
        return bus_data.range(223, 192);
    case 7:
        return bus_data.range(255, 224);
    case 8:
        return bus_data.range(287, 256);
    case 9:
        return bus_data.range(319, 288);
    case 10:
        return bus_data.range(351, 320);
    case 11:
        return bus_data.range(383, 352);
    case 12:
        return bus_data.range(415, 384);
    case 13:
        return bus_data.range(447, 416);
    case 14:
        return bus_data.range(479, 448);
    case 15:
        return bus_data.range(511, 480);
    default:
        return 0;
    }
}

void request_manager(hls::stream<edge_descriptor_batch_t> &edge_burst_stm,
                     hls::stream<ppb_request_pkt_t> &ppb_request_stm,
                     hls::stream<ppb_response_pkt_t> &ppb_response_stm,
                     hls::stream<update_tuple_t> &update_set_stm,
                     int32_t memory_offset, int32_t part_edge_num) {
    // as we can buffer two vertices in one row with width of 64-bit, we can let
    // the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2.
    bus_word_t src_prop_buffer[PE_NUM][2][SRC_BUFFER_SIZE >> 4];
#pragma HLS ARRAY_PARTITION variable = src_prop_buffer dim = 1 complete
#pragma HLS BIND_STORAGE variable = src_prop_buffer type = RAM_S2P impl = BRAM
#pragma HLS dependence variable = src_prop_buffer inter false

    int32_t pp_read_idx = 0;
    int32_t pp_write_idx = 0;

    int32_t pp_reponse_idx = 0;

    int32_t pp_read_round = 0;
    int32_t pp_write_round = 0;

    int32_t pp_request_round = 0;

    int32_t edge_set_cnt = 0;
    const int32_t total_edge_sets = (part_edge_num + PE_NUM - 1) / PE_NUM;

    bool wait_flag = 0;

    edge_descriptor_batch_t an_edge_burst;
#pragma HLS ARRAY_PARTITION variable = an_edge_burst.edges complete dim = 0

    ppb_request_pkt_t one_ppb_request;

    ppb_response_pkt_t one_ppb_response;

    distance_t real_edge_weight =
        1.0; // All edge weights are 1.0 in unweighted graph
    const ap_fixed_pod_t edge_weight =
        (*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight));

    const uint32_t total_rounds =
        (part_edge_num + SRC_BUFFER_SIZE - 1) / SRC_BUFFER_SIZE;

scatterLoop:
    while (true) {
#pragma HLS PIPELINE II = 1
        // logic to fill the ping-pong buffer.
        if ((pp_request_round - pp_read_round) <= 1) {
            if (pp_request_round < pp_read_round)
                pp_request_round = pp_read_round;
            one_ppb_request.data = pp_request_round + memory_offset;
            one_ppb_request.last = 0;
            ppb_request_stm.write(one_ppb_request);
            pp_request_round++;
        }

        if (ppb_response_stm.read_nb(one_ppb_response)) {
            pp_write_round =
                (one_ppb_response.dest << 4 >> LOG_SRC_BUFFER_SIZE) -
                memory_offset;

            bool write_buffer = pp_write_round & 0x1;

            int32_t write_idx =
                one_ppb_response.dest & ((SRC_BUFFER_SIZE >> 4) - 1);

            bus_word_t one_read_burst =
                one_ppb_response
                    .data; // src_prop[(base_addr >> 4) + pp_write_idx];

            // for (int j = 0; j < 16; j++) {
            //     ap_fixed_pod_t distance =
            //         one_ppb_response.data.range(
            //             (j + 1) * 32 - 1, j * 32);
            //     distance_t real_prop =
            //         *reinterpret_cast<distance_t *>(&distance);
            //     // printf(
            //     //     "memory_offset %d prop[%d] = %.3f, write_idx %d,
            //     write_buffer %d\n",
            //     //     memory_offset, j,
            //     //     (float)real_prop, write_idx, write_buffer);
            //     // fflush(NULL);
            // }

            for (int u = 0; u < PE_NUM; u++) {
#pragma HLS UNROLL
                src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;
            }
        }

        // logic to read the ping-pong buffer and synchronization.
        if (!wait_flag)
            an_edge_burst = edge_burst_stm.read();

        pp_read_round = (an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE);

        wait_flag = (pp_read_round >= pp_write_round) ? 1 : 0;

        bool exit_flag = (wait_flag == 0)
                             ? (edge_set_cnt + 1 >= total_edge_sets)
                             : (edge_set_cnt >= total_edge_sets);

        if (!wait_flag) {

            bool read_buffer = pp_read_round & 0x1;

            update_tuple_t an_update_set;
#pragma HLS ARRAY_PARTITION variable = an_update_set.prop complete dim = 0
#pragma HLS ARRAY_PARTITION variable = an_update_set.node_id complete dim = 0

            for (int u = 0; u < PE_NUM; u++) {
#pragma HLS UNROLL
                ap_uint<31> idx =
                    (an_edge_burst.edges[u].src_id % SRC_BUFFER_SIZE);
                ap_uint<30> uram_row_idx = idx >> 4;
                ap_uint<30> uram_row_offset = (idx & 0xf);

                bus_word_t uram_row =
                    src_prop_buffer[u][read_buffer][uram_row_idx];
                ap_fixed_pod_t src_prop =
                    get_val_from_bus(uram_row, uram_row_offset);

                // distance_t real_src_prop =
                //     *reinterpret_cast<distance_t *>(&src_prop);
                // printf(
                //     "Little PE %d memory_offset %d u %d read_buffer %d
                //     uram_row_idx %d edge src_id %d dst_id %d: loaded src_prop
                //     %.3f\n", (int)u, (int)memory_offset, (int)u,
                //     (int)read_buffer, (int)uram_row_idx,
                //     (int)an_edge_burst.edges[u].src_id,
                //     (int)an_edge_burst.edges[u].dst_id,
                //     (float)real_src_prop);

                an_update_set.prop[u] = (src_prop + edge_weight);
                an_update_set.node_id[u] = an_edge_burst.edges[u].dst_id;
            }
            update_set_stm.write(an_update_set);

            edge_set_cnt++;
        }

        if (exit_flag) {
            one_ppb_request.last = 1;
            ppb_request_stm.write(one_ppb_request);
        exitscatter:
            while (true) {
                ppb_response_stm.read(one_ppb_response);
                if (one_ppb_response.last)
                    break;
            }
            break;
        }
    }
}

ap_fixed_pod_t get_raw_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> bits;
    switch (idx) {
    case 0:
        bits = word.range(DISTANCE_BITWIDTH - 1, 0);
        break;
    case 1:
        bits = word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH);
        break;
    case 2:
        bits =
            word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1));
        break;
    default:
        bits = 0;
        break;
    }
    return bits;
}

void set_raw_val(reduce_word_t &word, int idx, ap_fixed_pod_t pod_val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits = pod_val;
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1)) =
            val_bits;
        break;
    default:
        break;
    }
}

// Single-PE aggregation function
// Handles initialization and aggregation for one PE
static void
Reduc_105_unit_reduce(hls::stream<update_tuple_t> &update_set_stm,
                      hls::stream<reduce_word_t> (&pe_mem_outs)[PE_NUM],
                      int32_t edge_num, int32_t dst_num) {
    // --- Phase 1: Memory Declaration ---
    const int MEM_SIZE = MAX_NUM / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[PE_NUM][MEM_SIZE];
#pragma HLS ARRAY_PARTITION variable = prop_mem complete dim = 1
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_S2P impl = URAM
#pragma HLS dependence variable = prop_mem inter false

    // Latency-hiding cache for recently accessed URAM words
    reduce_word_t cache_data_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    int32_t cache_addr_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0

    const int32_t num_words =
        (dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
    const int32_t rounded_num_words = (num_words + 7) / 8 * 8;

#ifdef EMULATION
    memset(prop_mem, 0, sizeof(reduce_word_t) * PE_NUM * MEM_SIZE);
#endif

LOOP_INIT_CACHE_ADDR:
    for (int i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            cache_addr_buffer[pe][i] = -1; // Invalidate cache
        }
    }

    const int32_t total_updates =
        (edge_num + PE_NUM - 1) / PE_NUM; // Assuming one update per node
    const int32_t last_pack_size =
        (edge_num % PE_NUM == 0) ? PE_NUM : (edge_num % PE_NUM);
    // --- Phase 3: Aggregation Loop ---
LOOP_AGGREGATE:
    for (int update_idx = 0; update_idx < total_updates; update_idx++) {
#pragma HLS PIPELINE II = 1
        update_tuple_t one_update;
#pragma HLS ARRAY_PARTITION variable = one_update.prop complete dim = 0
#pragma HLS ARRAY_PARTITION variable = one_update.node_id complete dim = 0
        one_update = update_set_stm.read();
        int32_t cur_pe_end =
            (update_idx == total_updates - 1) ? last_pack_size : PE_NUM;

        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            int32_t key = one_update.node_id[pe];
            if (pe < cur_pe_end && (key & 0x40000000) == 0) { // Valid key check
                ap_fixed_pod_t incoming_dist_pod = one_update.prop[pe];

                int32_t word_addr = (key >> 1);
                int32_t pack_idx = (key & 1);

                reduce_word_t current_word = prop_mem[pe][word_addr];

                // Check cache first
                for (int i = L; i >= 0; --i) {
#pragma HLS UNROLL
                    if (cache_addr_buffer[pe][i] == word_addr) {
                        current_word = cache_data_buffer[pe][i];
                        break;
                    }
                }

                // Shift cache
                for (int i = 0; i < L; i++) {
#pragma HLS UNROLL
                    cache_addr_buffer[pe][i] = cache_addr_buffer[pe][i + 1];
                    cache_data_buffer[pe][i] = cache_data_buffer[pe][i + 1];
                }

                ap_fixed_pod_t old_dist_pod =
                    get_raw_val(current_word, pack_idx);
                ap_fixed_pod_t new_dist_pod =
                    (old_dist_pod < incoming_dist_pod && old_dist_pod != 0x0)
                        ? old_dist_pod
                        : incoming_dist_pod;

                // distance_t old_dist =
                //     *reinterpret_cast<distance_t *>(&old_dist_pod);
                // distance_t new_dist =
                //     *reinterpret_cast<distance_t *>(&new_dist_pod);
                // printf(
                //     "Little PE %d updating node_id %d: old_dist %.3f,
                //     incoming_dist "
                //     "%.3f, new_dist %.3f\n",
                //     (int)pe, (int)key, (float)old_dist,
                //     (float)*reinterpret_cast<distance_t
                //     *>(&incoming_dist_pod), (float)new_dist);

                set_raw_val(current_word, pack_idx, new_dist_pod);

                // Write back to URAM and update cache
                prop_mem[pe][word_addr] = current_word;
                cache_addr_buffer[pe][L] = word_addr;
                cache_data_buffer[pe][L] = current_word;
            }
        }
    }

    // --- Phase 4: Stream out aggregated memory ---
LOOP_STREAM_OUT:
    for (int i = 0; i < rounded_num_words; i++) {
#pragma HLS PIPELINE II = 1
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            reduce_word_t word = prop_mem[pe][i];
            pe_mem_outs[pe].write(word);
            prop_mem[pe][i] = 0;
        }
    }
}

void set_word_in_bus(bus_word_t &bus_word, int idx, ap_fixed_pod_t pod_low,
                     ap_fixed_pod_t pod_high) {
#pragma HLS INLINE
    switch (idx) {
    case 0:
        bus_word.range(31, 0) = pod_low;
        bus_word.range(63, 32) = pod_high;
        break;
    case 1:
        bus_word.range(95, 64) = pod_low;
        bus_word.range(127, 96) = pod_high;
        ;
        break;
    case 2:
        bus_word.range(159, 128) = pod_low;
        bus_word.range(191, 160) = pod_high;
        break;
    case 3:
        bus_word.range(223, 192) = pod_low;
        bus_word.range(255, 224) = pod_high;
        break;
    case 4:
        bus_word.range(287, 256) = pod_low;
        bus_word.range(319, 288) = pod_high;
        break;
    case 5:
        bus_word.range(351, 320) = pod_low;
        bus_word.range(383, 352) = pod_high;
        break;
    case 6:
        bus_word.range(415, 384) = pod_low;
        bus_word.range(447, 416) = pod_high;
        break;
    case 7:
        bus_word.range(479, 448) = pod_low;
        bus_word.range(511, 480) = pod_high;
        break;
    default:
        break;
    }
}

// Multi-PE drain function
// Collects aggregated data from all PEs and outputs final results
static void
Reduc_105_drain_multi_pe(hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM],
                         hls::stream<little_out_pkt_t> &kernel_out_stream,
                         int32_t dst_num) {

    // --- Phase 2: High-Performance Drain Loop ---
    little_out_pkt_t one_write_burst;
    one_write_burst.last = 0;
    distance_t max_val = (distance_t)(16384.0);
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);

    // round dst_num to be 8 * DISTANCES_PER_REDUCE_WORD
    int32_t rounded_dst_num = ((dst_num + (8 * DISTANCES_PER_REDUCE_WORD) - 1) /
                               (8 * DISTANCES_PER_REDUCE_WORD)) *
                              (8 * DISTANCES_PER_REDUCE_WORD);

LOOP_DRAIN_ADDR:
    for (int32_t base_addr = 0; base_addr < rounded_dst_num;
         base_addr += DISTANCES_PER_REDUCE_WORD) {
#pragma HLS PIPELINE II = 1
        ap_fixed_pod_t uram_res_low = max_pod;
        ap_fixed_pod_t uram_res_high = max_pod;
    LOOP_FOR_57:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            reduce_word_t word = pe_mem_in[pe_idx].read();

            ap_fixed_pod_t incoming_dist_pod_low = word.range(31, 0);
            ap_fixed_pod_t incoming_dist_pod_high = word.range(63, 32);
            uram_res_low = (uram_res_low < incoming_dist_pod_low ||
                            incoming_dist_pod_low == 0x0)
                               ? uram_res_low
                               : incoming_dist_pod_low;
            uram_res_high = (uram_res_high < incoming_dist_pod_high ||
                             incoming_dist_pod_high == 0x0)
                                ? uram_res_high
                                : incoming_dist_pod_high;
        }
        reduce_word_t merged_word;
        merged_word.range(31, 0) = uram_res_low;
        merged_word.range(63, 32) = uram_res_high;
        one_write_burst.data = merged_word;
        kernel_out_stream.write(one_write_burst);
    }
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void
graphyflow_little(const bus_word_t *edge_props, int32_t num_nodes,
                  int32_t num_edges, int32_t dst_num, int32_t memory_offset,
                  hls::stream<ppb_request_pkt_t> &ppb_req_stream,
                  hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
                  hls::stream<little_out_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = edge_props
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = dst_num
#pragma HLS INTERFACE s_axilite port = memory_offset
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    // Existing streams
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 32
    hls::stream<update_tuple_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 8
    hls::stream<reduce_word_t> pe_mem_outs[PE_NUM];
#pragma HLS STREAM variable = pe_mem_outs depth = 8

    // --- Data Loading ---
    edge_descriptor_loader(edge_props, edge_stream, num_edges);
    request_manager(edge_stream, ppb_req_stream, ppb_resp_stream,
                    stream_edge_data, memory_offset, num_edges);

    // --- Reduction ---
    Reduc_105_unit_reduce(stream_edge_data, pe_mem_outs, num_edges, dst_num);
    Reduc_105_drain_multi_pe(pe_mem_outs, kernel_out_stream, dst_num);
}

```

`scripts/kernel/graphyflow_little.h`:

```h
#ifndef __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__
#define __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PE_NUM 8
#define DBL_PE_NUM 16
#define LOG_PE_NUM 3
#ifdef EMULATION
#define MAX_NUM 512
#else
#define MAX_NUM 65536
#endif
#define L 4
#define SRC_BUFFER_SIZE 4096
#define LOG_SRC_BUFFER_SIZE 12

// --- New Bitwidth Definitions for HLS Synthesis ---
#define NODE_ID_BITWIDTH 32
#define DISTANCE_BITWIDTH 32
#define DISTANCE_INTEGER_PART 16
#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH
#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART
#define OUT_END_MARKER_BITWIDTH 4
#define DIST_PER_WORD 16 // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = 512 / 32 = 16
#define LOG_DIST_PER_WORD                                                      \
    4 // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2(512 / 32) = log2(16) = 4

// --- New Memory Word and Bus Definitions ---
#define AXI_BUS_WIDTH 512

#define REDUCE_MEM_WIDTH 64
typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;

const int INFINITY_DIST = 16384;

// --- New Packing-related Constants ---
// Number of distances that can be packed into a single reduce memory word.
#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)

// --- Redefinition of Core Graph Types for HLS ---
// These typedefs override the standard integer types from common.h for
// synthesis.
typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;
typedef ap_uint<32> edge_id_t; // edge_id_t is not customized yet, keep as is.
typedef ap_uint<DISTANCE_BITWIDTH>
    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;
typedef ap_axiu<256, 0, 0, 0> node_dist_pkt_t;
typedef ap_axiu<512, 0, 0, 0> write_burst_pkt_t;
typedef ap_axiu<64, 0, 0, 0> little_out_pkt_t;
typedef ap_axiu<32, 0, 0, 0> ppb_request_pkt_t;
typedef ap_axiu<512, 0, 0, 32> ppb_response_pkt_t;
typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;

// --- Struct Type Definitions (UNCHANGED) ---
// The definitions of these structs remain the same, but the underlying
// types (node_id_t, ap_fixed_pod_t) are now custom-width, not uint32_t.
struct __attribute__((packed)) struct_ana_3_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
    ap_fixed_pod_t ele_2;
};

struct __attribute__((packed)) struct_abu_9_t {
    ap_fixed_pod_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_nbu_11_t {
    node_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ibu_14_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_an_15_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
};

struct __attribute__((packed)) struct_ebu_20_t {
    edge_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) node_with_prop_t {
    ap_fixed_pod_t prop;
    node_id_t node_id;
};

struct __attribute__((packed)) node_distance_cache_burst_t {
    ap_fixed_pod_t data[DIST_PER_WORD];
};

struct __attribute__((packed)) node_distance_burst_t {
    ap_fixed_pod_t data[DBL_PE_NUM];
};

struct __attribute__((packed)) node_id_burst_t {
    node_id_t data[PE_NUM];
};

struct __attribute__((packed)) distance_req_pack_t {
    node_id_t idx[PE_NUM];
    ap_uint<4> offset; // [offset, offset + PE_NUM) are valid
    bool end_flag;
};

struct __attribute__((packed)) cacheline_req_t {
    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> idx;
    ap_uint<4> target_pe;
    bool end_flag;
};

struct __attribute__((packed)) cacheline_resp_t {
    bus_word_t data;
    ap_uint<4> target_pe;
    bool end_flag;
};

struct __attribute__((packed)) edge_batch_t {
    ap_fixed_pod_t weights[PE_NUM];
    ap_fixed_pod_t src_distances[PE_NUM];
    node_id_t dsts[PE_NUM];
    int32_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) node_dist_batch_t {
    ap_fixed_pod_t data[DBL_PE_NUM];
    uint8_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    node_id_t id;
};

struct __attribute__((packed)) struct_sbu_7_t {
    struct_ana_3_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_17_t {
    struct_an_15_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) internal_end_data_batch_t {
    ap_fixed_pod_t data[DBL_PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_105_t {
    int32_t key;
    node_with_prop_t transform;
};

struct __attribute__((packed)) struct_nb_58_t {
    node_with_prop_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) edge_t {
    node_id_t src_id;
    node_id_t dst_id;
};

struct __attribute__((packed)) edge_descriptor_batch_t {
    edge_t edges[PE_NUM];
    int32_t end_pos;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

// struct __attribute__((packed)) struct_kbu_50_t {
//     kt_pair_105_t data[PE_NUM];
//     bool end_flag;
//     uint8_t end_pos;
// };

struct __attribute__((packed)) update_tuple_t {
    node_id_t node_id[PE_NUM];
    ap_fixed_pod_t prop[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_105_t_t {
    node_id_t node_id;
    ap_fixed_pod_t prop;
    bool end_flag;
};

// --- Top-Level Function Prototype ---
extern "C" void
graphyflow_little(const bus_word_t *edge_props, int32_t num_nodes,
                  int32_t num_edges, int32_t dst_num, int32_t memory_offset,
                  hls::stream<ppb_request_pkt_t> &ppb_req_stream,
                  hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
                  hls::stream<little_out_pkt_t> &kernel_out_stream);

#endif // __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__

```

`scripts/kernel/hbm_writer.cpp`:

```cpp
#include "shared_kernel_params.h"

static void little_node_prop_loader(
    int i, const bus_word_t *node_distances_ddr, uint32_t num_partitions,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream
    // hls::stream<cacheline_data_pkt_t> &cacheline_data_stream
) {
#pragma HLS function_instantiate variable = i

    ppb_request_pkt_t one_ppb_request_pkg;
    ppb_response_pkt_t one_ppb_response_pkg;
    uint32_t left_partitions = num_partitions;

littleKernelReadMemory:
    while (true) {
#pragma HLS PIPELINE
        if (ppb_req_stream.read_nb(one_ppb_request_pkg)) {
            uint32_t request_round = one_ppb_request_pkg.data;
            bool end_flag = one_ppb_request_pkg.last;

            uint32_t base_addr = request_round << LOG_SRC_BUFFER_SIZE >> 4;

            if (end_flag) {
                one_ppb_response_pkg.last = end_flag;
                ppb_resp_stream.write(one_ppb_response_pkg);
                left_partitions--;
                if (left_partitions == 0) {
                    break;
                }
            } else {
                for (int i = 0; i < (SRC_BUFFER_SIZE >> 4); i++) {
                    int addr = base_addr + i;

                    one_ppb_response_pkg.data = node_distances_ddr[addr];
                    one_ppb_response_pkg.dest = addr;
                    one_ppb_response_pkg.last = false;
                    ppb_resp_stream.write(one_ppb_response_pkg);
                }
            }
        }
    }
}

static void big_node_prop_loader(
    int i, const bus_word_t *node_distances_ddr, uint32_t num_partitions,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream
    // hls::stream<cacheline_data_pkt_t> &cacheline_data_stream
) {
#pragma HLS function_instantiate variable = i

    cacheline_request_pkt_t cache_req;
    cacheline_response_pkt_t cache_resp;

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx = -1;
    bus_word_t last_cacheline;

    uint32_t left_partitions = num_partitions;

LOOP_BIG_KRL_READ_MEMORY:
    while (true) {
#pragma HLS PIPELINE II = 1
        bool process_flag = cacheline_req_stream.read_nb(cache_req);

        ap_uint<26> idx = cache_req.data;
        ap_uint<8> target_pe = cache_req.dest;
        bool end_flag = cache_req.last;

        ap_uint<8> dst_pe;
        bus_word_t out_data;
        bool out_end_flag;

        if (process_flag) {
            // printf("Waiting for cacheline request...\n");fflush(NULL);
            // printf("Received cacheline request for idx %d from PE %d\n",
            // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
            if (end_flag) {
                out_data = 0;
                last_cache_idx = -1;
                left_partitions--;
            } else {
                if (idx == last_cache_idx) {
                    out_data = last_cacheline;
                } else {
                    out_data = node_distances_ddr[idx];
                    last_cache_idx = idx;
                    last_cacheline = out_data;
                }
            }

            out_end_flag = end_flag;
            dst_pe = target_pe;

            cache_resp.data = out_data;
            cache_resp.dest = dst_pe;
            cache_resp.last = out_end_flag;
            cacheline_resp_stream.write(cache_resp);
            // printf("Sent cacheline response for idx %d to PE %d\n",
            // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
        }
        if (left_partitions == 0) {
            break;
        }
    }
}

void write_out(bus_word_t *output,
               hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream) {
LOOP_WRITE_OUT:
    while (true) {
#pragma HLS PIPELINE II = 1

        write_burst_w_dst_pkt_t one_write_burst;

        if (write_burst_stream.read_nb(one_write_burst)) {
            uint32_t dest_addr = one_write_burst.dest;
            bus_word_t data = one_write_burst.data;
            bool end_flag = one_write_burst.last;

            if (end_flag) {
                break;
            }

            output[dest_addr] = data;
        }
    }
}

extern "C" void hbm_writer(
    bus_word_t *src_prop_1, bus_word_t *src_prop_2, bus_word_t *src_prop_3,
    bus_word_t *src_prop_4, bus_word_t *src_prop_5, bus_word_t *src_prop_6,
    bus_word_t *src_prop_7, bus_word_t *src_prop_8, bus_word_t *src_prop_9,
    bus_word_t *src_prop_10, bus_word_t *src_prop_11, bus_word_t *src_prop_12,
    bus_word_t *src_prop_13, bus_word_t *src_prop_14, bus_word_t *output,
    uint32_t num_partitions_little, uint32_t num_partitions_big,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_2,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_2,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_3,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_3,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_4,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_4,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_5,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_5,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_6,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_6,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_7,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_7,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_8,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_8,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_9,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_9,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_10,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_10,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_11,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_11,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_1,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_1,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_2,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_2,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_3,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_3,
    hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream) {
#pragma HLS INTERFACE m_axi port = src_prop_1 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = src_prop_2 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = src_prop_3 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = src_prop_4 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = src_prop_5 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = src_prop_6 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = src_prop_7 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = src_prop_8 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = src_prop_9 offset = slave bundle = gmem8
#pragma HLS INTERFACE m_axi port = src_prop_10 offset = slave bundle = gmem9
#pragma HLS INTERFACE m_axi port = src_prop_11 offset = slave bundle = gmem10
#pragma HLS INTERFACE m_axi port = src_prop_12 offset = slave bundle = gmem11
#pragma HLS INTERFACE m_axi port = src_prop_13 offset = slave bundle = gmem12
#pragma HLS INTERFACE m_axi port = src_prop_14 offset = slave bundle = gmem13
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = src_prop_1 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_2 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_3 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_4 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_5 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_6 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_7 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_8 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_9 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_10 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_11 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_12 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_13 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_14 bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = num_partitions_little bundle = control
#pragma HLS INTERFACE s_axilite port = num_partitions_big bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    little_node_prop_loader(0, src_prop_1, num_partitions_little,
                            ppb_req_stream_1, ppb_resp_stream_1);
    little_node_prop_loader(1, src_prop_2, num_partitions_little,
                            ppb_req_stream_2, ppb_resp_stream_2);
    little_node_prop_loader(2, src_prop_3, num_partitions_little,
                            ppb_req_stream_3, ppb_resp_stream_3);
    little_node_prop_loader(3, src_prop_4, num_partitions_little,
                            ppb_req_stream_4, ppb_resp_stream_4);
    little_node_prop_loader(4, src_prop_5, num_partitions_little,
                            ppb_req_stream_5, ppb_resp_stream_5);
    little_node_prop_loader(5, src_prop_6, num_partitions_little,
                            ppb_req_stream_6, ppb_resp_stream_6);
    little_node_prop_loader(6, src_prop_7, num_partitions_little,
                            ppb_req_stream_7, ppb_resp_stream_7);
    little_node_prop_loader(7, src_prop_8, num_partitions_little,
                            ppb_req_stream_8, ppb_resp_stream_8);
    little_node_prop_loader(8, src_prop_9, num_partitions_little,
                            ppb_req_stream_9, ppb_resp_stream_9);
    little_node_prop_loader(9, src_prop_10, num_partitions_little,
                            ppb_req_stream_10, ppb_resp_stream_10);
    little_node_prop_loader(10, src_prop_11, num_partitions_little,
                            ppb_req_stream_11, ppb_resp_stream_11);
    big_node_prop_loader(0, src_prop_12, num_partitions_big,
                         cacheline_req_stream_1, cacheline_resp_stream_1);
    big_node_prop_loader(1, src_prop_13, num_partitions_big,
                         cacheline_req_stream_2, cacheline_resp_stream_2);
    big_node_prop_loader(2, src_prop_14, num_partitions_big,
                         cacheline_req_stream_3, cacheline_resp_stream_3);

    write_out(output, write_burst_stream);
}
```

`scripts/kernel/kernel.mk`:

```mk
#
# Vitis 内核的 Makefile (多内核版本)
#

ifeq ($(TARGET),$(filter $(TARGET), sw_emu hw_emu))
CLFLAGS += -DEMULATION
endif

# --- 配置项 ---
VPP := v++
XCLBIN_DIR := ./xclbin
EMCONFIG_FILE := ./emconfig.json

# 1. 在这里定义您所有的内核名称。
#    这是将来您唯一需要修改的变量。
KERNEL_NAMES := graphyflow_little graphyflow_big apply_kernel hbm_writer big_merger little_merger

# 2. 定义最终输出的二进制文件的名称。
XCLBIN_NAME := graphyflow_kernels

# --- 文件名自动生成 ---

# 根据 KERNEL_NAMES 列表，自动生成所有内核对象 (.xo) 文件的列表。
# 例如: "graphyflow_big" -> "./xclbin/graphyflow_big.hw.xo"
KERNEL_XOS := $(patsubst %,$(XCLBIN_DIR)/%.$(TARGET).xo,$(KERNEL_NAMES))

# 定义最终 .xclbin 文件的完整路径。
XCLBIN_FILE := $(XCLBIN_DIR)/$(XCLBIN_NAME).$(TARGET).xclbin

# if defined WAVE then add -g for debug symbols
ifdef WAVE
CLFLAGS += -g
LDFLAGS_VPP += -g
endif

# --- 编译器和链接器参数 ---

# VPP 在编译 .xo 文件时使用的参数。
# 注意：特定的 "--kernel <名称>" 参数现在被移到了编译规则内部。
CLFLAGS += -Iscripts/kernel
CLFLAGS += -Iscripts/host
CLFLAGS += -I$(XILINX_XRT)/include
CLFLAGS += -I$(XILINX_VITIS)/include
CLFLAGS += -O3
CLFLAGS += --kernel_frequency=220

# VPP 在链接 .xclbin 文件时使用的参数。
LDFLAGS_VPP += --config ./system.cfg
LDFLAGS_VPP += -Iscripts/kernel
LDFLAGS_VPP += -Iscripts/host
LDFLAGS_VPP += -I$(XILINX_XRT)/include
LDFLAGS_VPP += -I$(XILINX_VITIS)/include
LDFLAGS_VPP += --xp prop:solution.kernel_compiler_margin=10%


# --- 构建规则 ---

# 默认目标
all: $(XCLBIN_FILE)

# 3. 链接规则：将所有内核对象 (.xo) 文件链接成一个二进制容器 (.xclbin)。
#    此规则依赖于 KERNEL_XOS 变量中定义的所有 .xo 文件。
$(XCLBIN_FILE): $(KERNEL_XOS)
	@echo "==> 正在将所有内核链接到 xclbin 文件: $@"
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) $(LDFLAGS_VPP) -o $@ $^

# 4. 模式规则：将任何内核源文件 (.cpp) 编译成对应的内核对象文件 (.xo)。
#    这一个规则就能同时处理 graphyflow_big.cpp 和 graphyflow_little.cpp。
#    '$*' 是一个特殊变量，代表文件名中的“主干”部分 (例如 "graphyflow_big")。
$(XCLBIN_DIR)/%.$(TARGET).xo: scripts/kernel/%.cpp
	@echo "==> 正在编译内核: $<"
	@mkdir -p $(XCLBIN_DIR)
	$(VPP) -c -t $(TARGET) --platform $(DEVICE) $(CLFLAGS) --kernel $* -o $@ $<

# 用于生成硬件仿真配置的规则。
emconfig:
	emconfigutil --platform $(DEVICE) --od .

# 用于清理所有生成文件的规则。
clean:
	@echo "==> 正在清理构建生成的文件"
	rm -rf $(XCLBIN_DIR) $(EMCONFIG_FILE)

# 声明伪目标 (这些目标不是实际的文件名，而是操作名称)。
.PHONY: all emconfig clean
```

`scripts/kernel/little_merger.cpp`:

```cpp
#include "shared_kernel_params.h"

void merge_little_kernels(
    hls::stream<little_out_pkt_t> &little_kernel_1_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_2_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_3_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_4_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_5_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_6_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_7_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_8_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_9_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_10_out_stream,
    hls::stream<little_out_pkt_t> &little_kernel_11_out_stream,
    hls::stream<write_burst_pkt_t> &kernel_out_stream) {
    little_out_pkt_t tmp_prop_pkt[LITTLE_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = tmp_prop_pkt dim = 0 complete

    bool process_flag[LITTLE_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = process_flag dim = 0 complete

    for (int i = 0; i < LITTLE_MERGER_LENGTH; i++) {
#pragma HLS unroll
        process_flag[i] = 0;
    }

    reduce_word_t merged_write_burst;

    bus_word_t one_write_burst;

    uint32_t inner_idx = 0;

    distance_t max_val = (distance_t)(16384.0);
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);

merge_tmp_prop_big_krnls:
    while (true) {
#pragma HLS pipeline style = flp

        if (!process_flag[0])
            process_flag[0] =
                little_kernel_1_out_stream.read_nb(tmp_prop_pkt[0]);
        if (!process_flag[1])
            process_flag[1] =
                little_kernel_2_out_stream.read_nb(tmp_prop_pkt[1]);
        if (!process_flag[2])
            process_flag[2] =
                little_kernel_3_out_stream.read_nb(tmp_prop_pkt[2]);
        if (!process_flag[3])
            process_flag[3] =
                little_kernel_4_out_stream.read_nb(tmp_prop_pkt[3]);
        if (!process_flag[4])
            process_flag[4] =
                little_kernel_5_out_stream.read_nb(tmp_prop_pkt[4]);
        if (!process_flag[5])
            process_flag[5] =
                little_kernel_6_out_stream.read_nb(tmp_prop_pkt[5]);
        if (!process_flag[6])
            process_flag[6] =
                little_kernel_7_out_stream.read_nb(tmp_prop_pkt[6]);
        if (!process_flag[7])
            process_flag[7] =
                little_kernel_8_out_stream.read_nb(tmp_prop_pkt[7]);
        if (!process_flag[8])
            process_flag[8] =
                little_kernel_9_out_stream.read_nb(tmp_prop_pkt[8]);
        if (!process_flag[9])
            process_flag[9] =
                little_kernel_10_out_stream.read_nb(tmp_prop_pkt[9]);
        if (!process_flag[10])
            process_flag[10] =
                little_kernel_11_out_stream.read_nb(tmp_prop_pkt[10]);

        bool merge_flag = process_flag[0] & process_flag[1] & process_flag[2] &
                          process_flag[3] & process_flag[4] & process_flag[5] &
                          process_flag[6] & process_flag[7] & process_flag[8] &
                          process_flag[9] & process_flag[10] & 1;

        if (merge_flag) {
            ap_fixed_pod_t uram_high = max_pod;;
            ap_fixed_pod_t uram_low = max_pod;

            for (int i = 0; i < LITTLE_MERGER_LENGTH; i++) {
#pragma HLS UNROLL
                ap_fixed_pod_t update_low = tmp_prop_pkt[i].data.range(31, 0);
                ap_fixed_pod_t update_high = tmp_prop_pkt[i].data.range(63, 32);
                uram_low = (uram_low < update_low || update_low == 0x0)
                               ? uram_low
                               : update_low;
                uram_high = (uram_high < update_high || update_high == 0x0)
                                ? uram_high
                                : update_high;
            }

            merged_write_burst.range(31, 0) = uram_low;
            merged_write_burst.range(63, 32) = uram_high;

            one_write_burst.range(63 + (inner_idx << 6), (inner_idx << 6)) =
                merged_write_burst;
            inner_idx++;

            if (inner_idx == 8) {
                write_burst_pkt_t out_pkt;
                out_pkt.data = one_write_burst;
                out_pkt.last = 0;
                kernel_out_stream.write(out_pkt);
                inner_idx = 0;
                one_write_burst = 0;
            }

            for (int i = 0; i < LITTLE_MERGER_LENGTH; i++) {
#pragma HLS unroll
                process_flag[i] = 0;
            }
        }
    }
}

extern "C" void
little_merger(hls::stream<little_out_pkt_t> &little_kernel_1_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_2_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_3_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_4_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_5_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_6_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_7_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_8_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_9_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_10_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_11_out_stream,
              hls::stream<write_burst_pkt_t> &kernel_out_stream) {

#pragma HLS interface ap_ctrl_none port = return

#pragma HLS DATAFLOW

    merge_little_kernels(little_kernel_1_out_stream, little_kernel_2_out_stream,
                         little_kernel_3_out_stream, little_kernel_4_out_stream,
                         little_kernel_5_out_stream, little_kernel_6_out_stream,
                         little_kernel_7_out_stream, little_kernel_8_out_stream,
                         little_kernel_9_out_stream,
                         little_kernel_10_out_stream,
                         little_kernel_11_out_stream, kernel_out_stream);
}

```

`scripts/kernel/shared_kernel_params.h`:

```h
#ifndef __SHARED_KERNEL_PARAMS_H__
#define __SHARED_KERNEL_PARAMS_H__

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PE_NUM 8
#define DBL_PE_NUM 16
#define LOG_PE_NUM 3
#define L 4
#define SRC_BUFFER_SIZE 4096
#define LOG_SRC_BUFFER_SIZE 12

#define NODE_ID_BITWIDTH 32
#define DISTANCE_BITWIDTH 32
#define DISTANCE_INTEGER_PART 16
#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH
#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART
#define OUT_END_MARKER_BITWIDTH 4
#define DIST_PER_WORD 16 // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = 512 / 32 = 16
#define LOG_DIST_PER_WORD                                                      \
    4 // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2(512 / 32) = log2(16) = 4

// --- New Memory Word and Bus Definitions ---
#define AXI_BUS_WIDTH 512

#define BIG_MERGER_LENGTH 3
#define LITTLE_MERGER_LENGTH 11

#define REDUCE_MEM_WIDTH 64
typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;

const int INFINITY_DIST = 16384;

#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)

// --- Redefinition of Core Graph Types for HLS ---
// These typedefs override the standard integer types from common.h for
// synthesis.
typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;
typedef ap_uint<32> edge_id_t; // edge_id_t is not customized yet, keep as is.
typedef ap_uint<DISTANCE_BITWIDTH>
    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;
typedef ap_axiu<256, 0, 0, 0> node_dist_pkt_t;
typedef ap_axiu<512, 0, 0, 0> write_burst_pkt_t;
typedef ap_axiu<64, 0, 0, 0> little_out_pkt_t;
typedef ap_axiu<512, 0, 0, 32> write_burst_w_dst_pkt_t;
typedef ap_axiu<32, 0, 0, 8> cacheline_request_pkt_t;
typedef ap_axiu<512, 0, 0, 8> cacheline_response_pkt_t;
typedef ap_axiu<32, 0, 0, 0> ppb_request_pkt_t;
typedef ap_axiu<512, 0, 0, 32> ppb_response_pkt_t;
typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;

struct __attribute__((packed)) in_write_burst_w_dst_pkt_t {
    bus_word_t data;
    ap_uint<32> dest_addr;
    bool end_flag;
};

extern "C" void
apply_kernel(bus_word_t *node_props, uint32_t little_kernel_length,
             uint32_t big_kernel_length, uint32_t little_kernel_st_offset,
             uint32_t big_kernel_st_offset,
             hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
             hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
             hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream);

extern "C" void
big_merger(hls::stream<write_burst_pkt_t> &big_kernel_1_out_stream,
           hls::stream<write_burst_pkt_t> &big_kernel_2_out_stream,
           hls::stream<write_burst_pkt_t> &big_kernel_3_out_stream,
           hls::stream<write_burst_pkt_t> &kernel_out_stream);

extern "C" void
little_merger(hls::stream<little_out_pkt_t> &little_kernel_1_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_2_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_3_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_4_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_5_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_6_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_7_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_8_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_9_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_10_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_11_out_stream,
              hls::stream<write_burst_pkt_t> &kernel_out_stream);

extern "C" void hbm_writer(
    bus_word_t *src_prop_1, bus_word_t *src_prop_2, bus_word_t *src_prop_3,
    bus_word_t *src_prop_4, bus_word_t *src_prop_5, bus_word_t *src_prop_6,
    bus_word_t *src_prop_7, bus_word_t *src_prop_8, bus_word_t *src_prop_9,
    bus_word_t *src_prop_10, bus_word_t *src_prop_11, bus_word_t *src_prop_12,
    bus_word_t *src_prop_13, bus_word_t *src_prop_14, bus_word_t *output,
    uint32_t num_partitions_little, uint32_t num_partitions_big,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_2,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_2,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_3,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_3,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_4,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_4,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_5,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_5,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_6,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_6,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_7,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_7,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_8,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_8,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_9,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_9,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_10,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_10,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_11,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_11,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_1,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_1,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_2,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_2,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_3,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_3,
    hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream);

#endif // __SHARED_KERNEL_PARAMS_H__
```

`scripts/main.mk`:

```mk
SHELL           := /bin/bash

COMMON_REPO     = ./
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))
SCRIPTS_PATH    = ./scripts

# Remove unused targets from .PHONY
.PHONY: all clean cleanall exe emconfig

include $(SCRIPTS_PATH)/help.mk
include $(SCRIPTS_PATH)/utils.mk

include global_para.mk

# Use SCRIPTS_PATH for consistency

# Remove non-existent makefiles
# include autogen/autogen.mk
# include acc_template/acc.mk

# Include our new kernel makefile
include $(SCRIPTS_PATH)/kernel/kernel.mk

include $(SCRIPTS_PATH)/host/host.mk 

# This include seems to be for Vitis 1.0 examples, not needed here
# include $(SCRIPTS_PATH)/bitstream.mk
include $(SCRIPTS_PATH)/clean.mk

# Update the 'all' rule to depend on the .xclbin file, the host executable, and emconfig
all: $(XCLBIN_FILE) $(EXECUTABLE) emconfig

exe: $(EXECUTABLE)

```

`scripts/utils.mk`:

```mk
#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------

PROFILE := no

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all
endif

DEBUG := no

#Generates debug summary report
ifeq ($(DEBUG), yes)
CLFLAGS += --dk protocol:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
LDCLFLAGS += --dk list_ports
endif

#Checks for XILINX_VITIS
ifndef XILINX_VITIS
$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif

#Checks for XILINX_XRT
check-xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly and rerun)
endif

check-devices:
ifndef DEVICE
	$(error DEVICE not set. Please set the DEVICE properly and rerun. Run "make help" for more details.)
endif

check-aws_repo:
ifndef SDACCEL_DIR
	$(error SDACCEL_DIR not set. Please set it properly and rerun. Run "make help" for more details.)
endif

#   sanitize_dsa - create a filesystem friendly name from dsa name
#   $(1) - name of dsa
COLON=:
PERIOD=.
UNDERSCORE=_
sanitize_dsa = $(strip $(subst $(PERIOD),$(UNDERSCORE),$(subst $(COLON),$(UNDERSCORE),$(1))))

device2dsa = $(if $(filter $(suffix $(1)),.xpfm),$(shell $(COMMON_REPO)/utility/parsexpmf.py $(1) dsa 2>/dev/null),$(1))
device2sandsa = $(call sanitize_dsa,$(call device2dsa,$(1)))
device2dep = $(if $(filter $(suffix $(1)),.xpfm),$(dir $(1))/$(shell $(COMMON_REPO)/utility/parsexpmf.py $(1) hw 2>/dev/null) $(1),)

# Cleaning stuff
RM = rm -f
RMDIR = rm -rf

ECHO:= @echo

```