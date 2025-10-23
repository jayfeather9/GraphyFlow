#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 3
#define LITTLE_KERNEL_NUM 0

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define BIG_KERNEL_HBM_EDGE_ID {0, 1, 2}
#define BIG_KERNEL_HBM_NODE_ID {20, 21, 22}
// #define LITTLE_KERNEL_HBM_EDGE_ID                                             \
//     {                                                                          \
//     }
// #define LITTLE_KERNEL_HBM_NODE_ID                                            \
//     {                                                                          \
//     }

#endif /* __HOST_CONFIG_H__ */
