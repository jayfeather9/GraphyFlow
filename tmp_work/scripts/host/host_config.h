#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 3
#define LITTLE_KERNEL_NUM 3

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define BIG_KERNEL_HBM_INPUT_ID {0, 2, 4}
#define BIG_KERNEL_HBM_OUTPUT_ID {1, 3, 5}
#define LITTLE_KERNEL_HBM_INPUT_ID {6, 8, 10}
#define LITTLE_KERNEL_HBM_OUTPUT_ID {7, 9, 11}

#endif /* __HOST_CONFIG_H__ */
