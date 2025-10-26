#include "acc_setup.h"

AccDescriptor initAccelerator(const std::string xclbin_path) {
    cl_int err;
    AccDescriptor acc;

    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    // 设置使用的HBM BANK：
    // acc.big_kernel_hbm_input_id = {0};
    // acc.big_kernel_hbm_output_id = {1};
    // acc. big_kernel_hbm_stop_flag_id = {2};

    // acc.little_kernel_hbm_input_id = {4,8};
    // acc.little_kernel_hbm_output_id = {5,9};
    // acc.little_kernel_hbm_stop_flag_id = {6,10};
    //  为选定的设备创建上下文和主命令队列
    OCL_CHECK(err, acc.context =
                       cl::Context(device, nullptr, nullptr, nullptr, &err));
    OCL_CHECK(err, acc.q = cl::CommandQueue(acc.context, device,
                                            CL_QUEUE_PROFILING_ENABLE, &err));

    // 为每个 "big" 内核实例创建一个专用的命令队列 ---
    acc.big_gs_queue.resize(acc.num_big_krnl);
    for (int k = 0; k < acc.num_big_krnl; k++) {
        cl::CommandQueue tmp_q;
        OCL_CHECK(err,
                  tmp_q = cl::CommandQueue(acc.context, device,
                                           CL_QUEUE_PROFILING_ENABLE, &err));
        acc.big_gs_queue[k] = tmp_q;
    }

    // 为每个 "little" 内核实例创建一个专用的命令队列 ---
    acc.little_gs_queue.resize(acc.num_little_krnl);
    for (int k = 0; k < acc.num_little_krnl; k++) {
        cl::CommandQueue tmp_q;
        OCL_CHECK(err,
                  tmp_q = cl::CommandQueue(acc.context, device,
                                           CL_QUEUE_PROFILING_ENABLE, &err));
        acc.little_gs_queue[k] = tmp_q;
    }

    // 为每个 "writer" 内核实例创建一个专用的命令队列 ---
    acc.writer_queue.resize(acc.num_writer_krnl);
    for (int k = 0; k < acc.num_writer_krnl; k++) {
        cl::CommandQueue tmp_q;
        OCL_CHECK(err,
                  tmp_q = cl::CommandQueue(acc.context, device,
                                           CL_QUEUE_PROFILING_ENABLE, &err));
        acc.writer_queue[k] = tmp_q;
    }

    // 为 apply 内核创建一个专用的命令队列 ---
    OCL_CHECK(err, acc.apply_queue = cl::CommandQueue(
                       acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    std::cout << "Attempting to program device: "
              << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(acc.context, {device}, bins, nullptr, &err);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file!\n";
        exit(EXIT_FAILURE);
    } else {
        std::cout << "Device program successful!\n";

        // 创建 acc.num_big_krnl 个 "graphyflow_big" 内核实例 ---
        for (int i = 0; i < acc.num_big_krnl; i++) {
            std::string cu_id = std::to_string(i + 1);
            // 构造内核名称，格式为 "kernel_name:{instance_name_ID}"
            std::string krnl_name_full = std::string("graphyflow_big:{") +
                                         "graphyflow_big_" + cu_id + "}";

            cl::Kernel tmp_gs_krnl;
            printf("Creating a big kernel [%s] for CU(%d)\n",
                   krnl_name_full.c_str(), i + 1);
            OCL_CHECK(err, tmp_gs_krnl = cl::Kernel(
                               program, krnl_name_full.c_str(), &err));
            acc.big_gs_krnls.push_back(tmp_gs_krnl);
        }

        // 创建 acc.num_little_krnl 个 "graphyflow_little" 内核实例 ---
        // for (int i = 0; i < acc.num_little_krnl; i++) {
        //     std::string cu_id = std::to_string(i + 1);
        //     std::string krnl_name_full = std::string("graphyflow_little:{") +
        //                                  "graphyflow_little_" + cu_id + "}";

        //     cl::Kernel tmp_gs_krnl;
        //     printf("Creating a little kernel [%s] for CU(%d)\n",
        //            krnl_name_full.c_str(), i + 1);
        //     OCL_CHECK(err, tmp_gs_krnl = cl::Kernel(
        //                        program, krnl_name_full.c_str(), &err));
        //     acc.little_gs_krnls.push_back(tmp_gs_krnl);
        // }

        // 创建 acc.num_writer_krnl 个 "hbm_writer" 内核实例 ---
        // for (int i = 0; i < acc.num_writer_krnl; i++) {
        //     std::string cu_id = std::to_string(i + 1);
        //     std::string krnl_name_full =
        //         std::string("hbm_writer:{") + "hbm_writer_" + cu_id + "}";

        //     cl::Kernel tmp_writer_krnl;
        //     printf("Creating a writer kernel [%s] for CU(%d)\n",
        //            krnl_name_full.c_str(), i + 1);
        //     OCL_CHECK(err, tmp_writer_krnl = cl::Kernel(
        //                        program, krnl_name_full.c_str(), &err));
        //     acc.writer_krnls.push_back(tmp_writer_krnl);
        // }

        // 创建 apply 内核实例 ---
        // {
        //     std::string krnl_name_full =
        //         std::string("apply_kernel:{") + "apply_kernel_1}";

        //     cl::Kernel tmp_apply_krnl;
        //     printf("Creating an apply kernel [%s]\n",
        //     krnl_name_full.c_str()); OCL_CHECK(err, tmp_apply_krnl =
        //     cl::Kernel(
        //                        program, krnl_name_full.c_str(), &err));
        //     acc.apply_krnl = tmp_apply_krnl;
        // }
    }

    return acc;
}