#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 0
#define LITTLE_KERNEL_NUM 1

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

// #define LITTLE_KERNEL_HBM_EDGE_ID {0, 2, 4}
// #define LITTLE_KERNEL_HBM_NODE_ID {1, 3, 5}
// #define BIG_KERNEL_HBM_EDGE_ID {6, 8}
// #define BIG_KERNEL_HBM_NODE_ID {7, 9}

#define LITTLE_KERNEL_HBM_EDGE_ID {0}
#define LITTLE_KERNEL_HBM_NODE_ID {1}
#define BIG_KERNEL_HBM_EDGE_ID                                                 \
    {                                                                          \
    }
#define BIG_KERNEL_HBM_NODE_ID                                                 \
    {                                                                          \
    }

#endif /* __HOST_CONFIG_H__ */
