#ifndef __GRAPH_PARTITION_H__
#define __GRAPH_PARTITION_H__

#include "common.h"
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

std::vector<GraphCSR> partition_graph(const GraphCSR &graph, int num_partitions, float *partition_weights);
GraphCSR merge_partitions(const std::vector<GraphCSR> &partitions);

#endif // __GRAPH_PARTITION_H__