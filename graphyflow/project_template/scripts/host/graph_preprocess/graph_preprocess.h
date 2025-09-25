#ifndef GRAPH_PREPROCESS_H
#define GRAPH_PREPROCESS_H

#include "graph_preprocess.h"
#include "host_config.h"
#include "common.h"

#include <numeric>      // std::iota
#include <algorithm>    // std::swap
#include <iomanip>



using BatchVec = std::vector<BATCH_TYPE>;





typedef 
struct suPartitionDescriptor{
    unsigned int            num_edges;
    unsigned int            num_vertices;

    bool                    kernel_type; // 0 stands big kernel; 1 for little kernel.
    unsigned int            kernel_id;
    
    unsigned int            dst_offset;
    unsigned int            dst_len;

    std::vector<uint, aligned_allocator<uint>> edge_array_host;

    cl::Buffer              edge_array_dev;
    cl_mem_ext_ptr_t        edge_array_ext_ptr;  

    cl::Event               event;    
} subpartition_descriptor_dt;


typedef 
struct PartitionDescriptor{
    unsigned int            num_edges;
    unsigned int            num_vertices;

    bool                    is_dense; // 0 stands big kernel; 1 for little kernel.
    unsigned int            kernel_id;
    
    unsigned int            dst_offset;//似乎用不到
    unsigned int            dst_len;    //似乎用不到

    std::vector<BATCH_TYPE, aligned_allocator<BATCH_TYPE> > batch_array_host;//这是一个数组，就存储这个分区的所有边
    cl::Buffer              edge_array_dev; // 似乎没用到，句柄在别的地方存着
    cl_mem_ext_ptr_t        edge_array_ext_ptr;  //没用到

    cl::Event               event;    //没用到
    uint                    est_cycles; //没用到
    
    unsigned int            num_subpartitions = LITTLE_KERNEL_NUM;//没用到
    std::vector<subpartition_descriptor_dt> subP;//没用到

} partition_descriptor_dt;


typedef 
struct PartitionContainer{// 包含了若干个 partition_descriptor_dt
    unsigned int            num_graph_vertices;
    unsigned int            num_graph_edges;

	//std::vector<uint,  aligned_allocator<uint>>     vertex_property;//最初的数据，没用到
	//std::vector<prop_t, aligned_allocator<prop_t>>  edge_property;//最初的数据，没用到

    unsigned int            num_partitions;//分成几份，没用到
    std::vector<partition_descriptor_dt> P;//最初的分区，没用到

    unsigned int            num_dense_partitions;//小核数量
    unsigned int            num_sparse_partitions;//大核数量

    std::vector<partition_descriptor_dt> DP; //分好区，分到了小核
    std::vector<partition_descriptor_dt> SP; //分好区，分到了大核


    //std::vector<cl::Buffer> src_prop_dev;//用不到，
    //std::vector<cl_mem_ext_ptr_t> src_prop_ext_ptr;//用不到

    //std::vector<uint,  aligned_allocator<uint>>  dst_tmp_prop_host;//用不到
    //std::vector<cl::Buffer> dst_tmp_prop_dev;//用不到
    //std::vector<cl_mem_ext_ptr_t> dst_tmp_prop_ext_ptr;//用不到

	//std::vector<uint,  aligned_allocator<uint>>     outdegree_host;
    //cl::Buffer outdegree_dev;
    //cl_mem_ext_ptr_t outdegree_ext_ptr;

    //cl::Buffer apply_src_prop_dev;
    //cl_mem_ext_ptr_t apply_src_prop_ptr;
    
} partition_container_dt;


partition_container_dt partitionGraph (const GraphCSR* graph);



#endif