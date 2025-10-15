#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

#define BIG_KERNEL_NUM 1
#define LITTLE_KERNEL_NUM 0

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define BIG_KERNEL_HBM_INPUT_ID {0}
#define BIG_KERNEL_HBM_OUTPUT_ID {1}
#define LITTLE_KERNEL_HBM_INPUT_ID                                             \
    {                                                                          \
    }
#define LITTLE_KERNEL_HBM_OUTPUT_ID                                            \
    {                                                                          \
    }

#endif /* __HOST_CONFIG_H__ */
