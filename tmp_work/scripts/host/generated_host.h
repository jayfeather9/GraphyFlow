#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "acc_setup/acc_setup.h"
#include "common.h"
#include "graph_preprocess/graph_preprocess.h"
#include <vector>

#include <vector>

// Define a structure to hold all OpenCL buffers for a single kernel instance.
// This improves code organization and simplifies buffer management.
struct KernelBuffers {
    cl::Buffer edge_props_buf; // Buffer for edge properties (destination ID and
                               // source ID)
};

// Structure to hold buffers for HBM writer kernel
struct WriterKernelBuffers {
    cl::Buffer node_props_buf; // Buffer for node properties (distances)
    cl::Buffer output_buf;     // Buffer for final output
};

struct HostInputBuffers {
    std::vector<bus_word_t, aligned_allocator<bus_word_t>> packed_edge_props;
    std::vector<bus_word_t, aligned_allocator<bus_word_t>> packed_node_props;
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
    void execute_kernel_iteration(const PartitionContainer &container,
                                  std::vector<cl::Event> &big_kernel_events,
                                  std::vector<cl::Event> &little_kernel_events,
                                  std::vector<cl::Event> &apply_kernel_events,
                                  cl::Event &hbm_writer_event);
    void transfer_data_from_fpga();
    bool check_convergence_and_update(const PartitionContainer &container);
    const std::vector<int> &get_results() const;

  private:
    AccDescriptor &acc;

    // Algorithm state
    int m_num_vertices;

    // Host-side master distance vector using original (global) vertex IDs
    std::vector<distance_t> h_distances;

    // Buffer containers for big kernels (one entry per kernel instance)
    std::vector<HostInputBuffers> big_kernel_input_buffers,
        little_kernel_input_buffers;
    std::vector<KernelBuffers> big_kernel_buffers, little_kernel_buffers;

    // Buffer containers for HBM writer kernels (one entry per writer kernel
    // instance)
    std::vector<WriterKernelBuffers> writer_kernel_buffers;
    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        writer_kernel_host_outputs;
};

#endif // __GENERATED_HOST_H__