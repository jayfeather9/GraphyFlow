#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>


#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)
#define BIG_KERNEL_NUM 1
#define LITTLE_KERNEL_NUM 2

#if 1

#define DEBUG_PRINTF(fmt,...)   printf(fmt,##__VA_ARGS__); fflush(stdout);

#else

#define DEBUG_PRINTF(fmt,...)   ;

#endif

#endif /* __HOST_CONFIG_H__ */



