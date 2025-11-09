#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 1
#define LITTLE_KERNEL_NUM 1

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define LITTLE_KERNEL_HBM_EDGE_ID {0}
#define LITTLE_KERNEL_HBM_NODE_ID {1}
#define BIG_KERNEL_HBM_EDGE_ID {22}
#define BIG_KERNEL_HBM_NODE_ID {23}

#endif /* __HOST_CONFIG_H__ */
