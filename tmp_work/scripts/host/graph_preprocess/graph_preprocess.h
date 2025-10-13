#ifndef GRAPH_PREPROCESS_H
#define GRAPH_PREPROCESS_H

#include "common.h"
#include "host_config.h"

#include <algorithm> // std::swap
#include <iomanip>
#include <numeric> // std::iota
#include <vector>

/**
 * @struct PartitionDescriptor
 * @brief Describes a single graph partition for one kernel.
 * * This structure holds a self-contained CSR representation of a graph
 * partition, including the mapping between its local, compressed vertex IDs and
 * the original global vertex IDs.
 */
typedef struct PartitionDescriptor {
    // Metadata about the partition
    unsigned int num_edges;
    unsigned int num_vertices; // Number of vertices *within this partition*
    bool is_dense;             // True for little kernel, false for big kernel
    unsigned int kernel_id; // The kernel instance this partition is assigned to

    // The core graph data for this partition in CSR format
    // It includes the compressed graph topology and vertex ID mappings.
    GraphCSR partitioned_graph;

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