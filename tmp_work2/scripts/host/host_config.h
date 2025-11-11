#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 0
#define LITTLE_KERNEL_NUM 10

#define NUM_LITTLE_MERGERS 2
#define NUM_BIG_MERGERS 0

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

static constexpr uint32_t LITTLE_MERGER_PIPELINE_LENGTHS[] = {5, 5};
static constexpr uint32_t LITTLE_MERGER_KERNEL_OFFSETS[] = {0, 5};
static constexpr uint32_t BIG_MERGER_PIPELINE_LENGTHS[] = {};
static constexpr uint32_t BIG_MERGER_KERNEL_OFFSETS[] = {};
static constexpr uint32_t LITTLE_KERNEL_GROUP_ID[] = {0, 0, 0, 0, 0,
                                                      1, 1, 1, 1, 1};
static constexpr uint32_t BIG_KERNEL_GROUP_ID[] = {};

#define LITTLE_KERNEL_HBM_EDGE_ID {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
#define LITTLE_KERNEL_HBM_NODE_ID {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
#define BIG_KERNEL_HBM_EDGE_ID                                                 \
    {                                                                          \
    }
#define BIG_KERNEL_HBM_NODE_ID                                                 \
    {                                                                          \
    }

#endif /* __HOST_CONFIG_H__ */
