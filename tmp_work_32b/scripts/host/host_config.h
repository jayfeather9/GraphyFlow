#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 3
#define LITTLE_KERNEL_NUM 11

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define USE_DDR 0

#define LITTLE_KERNEL_MEM_EDGE_ID {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
#define LITTLE_KERNEL_MEM_NODE_ID {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21}
#define BIG_KERNEL_MEM_EDGE_ID {22, 24, 26}
#define BIG_KERNEL_MEM_NODE_ID {23, 25, 27}

#define LITTLE_KERNEL_HBM_EDGE_ID {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
#define LITTLE_KERNEL_HBM_NODE_ID {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21}
#define BIG_KERNEL_HBM_EDGE_ID {22, 24, 26}
#define BIG_KERNEL_HBM_NODE_ID {23, 25, 27}

#define LITTLE_KERNEL_DDR_EDGE_ID {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define LITTLE_KERNEL_DDR_NODE_ID {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define BIG_KERNEL_DDR_EDGE_ID {0, 0, 0}
#define BIG_KERNEL_DDR_NODE_ID {0, 0, 0}

#endif /* __HOST_CONFIG_H__ */
