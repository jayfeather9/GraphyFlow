#ifndef __GRAPH_PARTITION_H__
#define __GRAPH_PARTITION_H__

#include "common.h"
#include <vector>

std::vector<GraphCSR> partition_graph(const GraphCSR &graph, int num_partitions);
GraphCSR merge_partitions(const std::vector<GraphCSR> &partitions);

#endif // __GRAPH_PARTITION_H__