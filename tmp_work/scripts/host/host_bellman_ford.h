#ifndef __HOST_BELLMAN_FORD_H__
#define __HOST_BELLMAN_FORD_H__

#include "common.h"

// This function performs a single iteration of the Bellman-Ford algorithm on
// the host CPU. It returns true if any distance value was updated, false
// otherwise.
bool host_bellman_ford_iteration(const GraphCSR &graph,
                                 std::vector<int> &distances);

#endif // __HOST_BELLMAN_FORD_H__