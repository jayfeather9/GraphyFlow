# GraphyFlow

## 算子定义

- n.map_(map_func) 传入一个lambda函数，对数组n的每个元素执行map_func
- n.filter(filter_func) 传入一个lambda函数，对数组n每个元素算filter_func，如果结果为False就将其从n中丢掉
- n.reduce_by(reduce_key, reduce_transform, reduce_method)，传入三个lambda函数，计算时，先把数组n的每个元素计算reduce_key然后按计算结果相同的来分组，然后对每个组内的每个元素做reduce_transform之后，逐个使用reduce_method两两计算，要求reduce_method满足交换律结合律，输出一个reduce_method输出类型的数组（个数等于分组个数）
