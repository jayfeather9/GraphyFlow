#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>


#define BIG_KERNEL_NUM {{BIG_KERNEL_NUM}}
#define LITTLE_KERNEL_NUM {{LITTLE_KERNEL_NUM}}

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)

#define BIG_KERNEL_HBM_INPUT_ID {{BIG_KERNEL_HBM_INPUT_ID}};
#define BIG_KERNEL_HBM_OUTPUT_ID {{BIG_KERNEL_HBM_OUTPUT_ID}};
#define LITTLE_KERNEL_HBM_INPUT_ID {{LITTLE_KERNEL_HBM_INPUT_ID}};
#define LITTLE_KERNEL_HBM_OUTPUT_ID {{LITTLE_KERNEL_HBM_OUTPUT_ID}};

#if 1

#define DEBUG_PRINTF(fmt,...)   printf(fmt,##__VA_ARGS__); fflush(stdout);

#else

#define DEBUG_PRINTF(fmt,...)   ;

#endif

#endif /* __HOST_CONFIG_H__ */



