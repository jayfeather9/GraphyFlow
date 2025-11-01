#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 2
#define LITTLE_KERNEL_NUM 2

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define LITTLE_KERNEL_HBM_EDGE_ID {0, 1}
#define LITTLE_KERNEL_HBM_NODE_ID {9, 10}
#define BIG_KERNEL_HBM_EDGE_ID {28, 30}
#define BIG_KERNEL_HBM_NODE_ID {11, 12}

#endif /* __HOST_CONFIG_H__ */
