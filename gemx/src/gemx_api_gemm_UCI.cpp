/**********
 * Copyright (c) 2017, Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * **********/
/**
 *  @brief Simple GEMM example of C++ API client interaction with GEMMX linear algebra accelerator on Xilinx FPGA
 *
 *  $DateTime: 2017/08/18 08:31:34 $
 *  $Author: jzejda $
 */

// Prerequisites:
//  - Boost installation (edit the Makefile with your boost path)
//  - Compiled C++ to bitstream accelerator kernel
//     - use "make run_hw"
//     - or get a pre-compiled copy of the out_hw/gemx.xclbin)
// Compile and run this API example:
//   make out_host/gemx_api_gemm.exe
//   out_host/gemx_api_gemm.exe
//     # No argumens will show help message
// You can also test it with a cpu emulation accelerator kernel (faster to combile, make run_cpu_em)
//   ( setenv XCL_EMULATION_MODE true ; out_host/gemx_api_gemm.exe out_cpu_emu/gemx.xclbin )

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdio.h>  // fgets for popen

#include "gemx_kernel.h"
#include "gemx_fpga.h"
#include "gemx_gen_bin.h"
#include "gemx_gemm.h"
#include "xcl2.hpp"


//Dimension of square input array

#define DATA_SIZE 8192
#define BSIZE 32

//Each matrix A, B, X, and C represent a quarter of the input matrix
size_t matrix_Xsize_bytes = sizeof(GEMX_XdataType) * DATA_SIZE * DATA_SIZE;
size_t matrix_ABsize_bytes = sizeof(GEMX_dataType) * DATA_SIZE * DATA_SIZE;
size_t matrix_Csize_bytes = sizeof(GEMX_dataType) * DATA_SIZE * DATA_SIZE;
size_t matrix_size_bytes = 2 * matrix_ABsize_bytes + matrix_Xsize_bytes + matrix_Csize_bytes;

typedef struct Offsets{
    int A;
    int B;
    int X;
    int C;
}offsets;

//Matrix multiply, out = in + in1 x in2
void MatMul(GEMX_dataType *in1, GEMX_dataType *in2, GEMX_dataType *in, GEMX_dataType *out, int outRow, int outCol, int midSize){
  for(int i = 0; i < outRow; i++) {
    for(int j = 0; j < outCol; j++) {
        out[i * outCol + j] = in[i * outCol + j];//e_a;
        for(int k = 0; k < midSize; k++) {
	    //out[i * outCol + j] += in1[i * midSize + k] * in2[k * outCol + j];
            out[i * outCol + j] = add(out[i * outCol + j], mul(in1[i * midSize + k], in2[k * outCol + j]));
        }
    	out[i * outCol + j] = add(in[i * outCol + j], out[i * outCol + j]);
    }
  }
}

//CPU implementation of Floyd-Warshall
//The inputs are of the size (DATA_SIZE x DATA_SIZE)
void FW_cpu (
    GEMX_dataType *in,   //Input Matrix
    GEMX_dataType *out,   //Output Matrix
    int dim     //One dimension of matrix
)
{
    //Initialize output matrix to input matrix
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
	    out[i * dim + j] = in[i * dim + j];

	}
    }
    //Perform Floyd-Warshall
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            for(int k = 0; k < dim; k++) {
		            if(out[j * dim + i] + out[i * dim + k] < out[j * dim + k])
		              out[j * dim + k] = out[j * dim + i] + out[i * dim + k];
            }
        }
    }
}

//Functionality to setup OpenCL context and trigger the Kernel
uint64_t GEMM_fpga (
    std::string l_xclbinFile,  //xclbinFile
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>>& source_in1,   //Input/Otput Matrix 1
    int l_M,
    int l_K,
    int l_N,
    int l_LdA,
    int l_LdB,
    int l_LdC,
    int l_LdX,
    int l_postScale
)
{
    cl::Event l_event_1, l_event_2;
    uint64_t kernel_duration = 0;

    //The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    //Creating Context and Command Queue for selected Device
    cl::Context context(device);
    cl::CommandQueue q1(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    //import_binary() command will find the OpenCL binary file created using the
    //xocc compiler load into OpenCL Binary and return as Binaries
    //OpenCL and it can contain many functions which can be executed on the
    //device.;
    cl::Program::Binaries bins = xcl::import_binary_file(l_xclbinFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    //This call will extract a kernel out of the program we loaded in the
    //previous line. A kernel is an OpenCL function that is executed on the
    //FPGA. This function is defined in the src/mmult.cl file.
    cl::Kernel kernel1(program,"GemmCall");
    std::vector<cl::Memory> m_Buffers;

	//These commands will allocate memory on the FPGA. The cl::Buffer
	//objects can be used to reference the memory locations on the device.
	//The cl::Buffer object cannot be referenced directly and must be passed
	//to other OpenCL functions.
	cl::Buffer buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_in1.data());
	m_Buffers.push_back(buffer);
	std::cout<<"[INFO] cl::Buffer size_in_bytes: "<<matrix_size_bytes<<std::endl;

    	//These commands will load the source_in1 and source_in2 vectors from the host
    	//application into the buffer_in1 and buffer_in2 cl::Buffer objects. The data
    	//will be be transferred from system memory over PCIe to the FPGA on-board
    	//DDR memory.

	std::vector<cl::Event> m_Mem2FpgaEvents;
	std::vector<cl::Event> m_ExeKernelEvents;

//    	q1.enqueueMigrateMemObjects({buffer},0/* 0 means from host*/);
    	q1.enqueueMigrateMemObjects(m_Buffers,0/* 0 means from host*/,NULL,&l_event_1);
	m_Mem2FpgaEvents.push_back(l_event_1);

        int narg = 0;
	kernel1.setArg(narg++, buffer);
	kernel1.setArg(narg++, buffer);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        kernel1.setArg(narg++, l_LdA);
        kernel1.setArg(narg++, l_LdB);
        kernel1.setArg(narg++, l_LdC);
        kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);

	//Launch the kernel
//    	q1.enqueueTask(kernel1, NULL, &event);
	q1.enqueueTask(kernel1, &m_Mem2FpgaEvents, &l_event_2);
	m_ExeKernelEvents.push_back(l_event_2);
	m_Mem2FpgaEvents.clear();

//    	q1.enqueueMigrateMemObjects({buffer},CL_MIGRATE_MEM_OBJECT_HOST);
    	q1.enqueueMigrateMemObjects(m_Buffers,CL_MIGRATE_MEM_OBJECT_HOST, &m_ExeKernelEvents);
	m_ExeKernelEvents.clear();

    	q1.finish();

    return kernel_duration;
}

void RKleene_fpga_helper (
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>>& source_in1,   //Input Matrix 1
    cl::Buffer& buf,   	// Matrix buffer
    cl::Buffer& buf_out,   	// Matrix buffer
    int x, int y,      	// Origin coordinates
    int dim,            // One dimension of computation
    cl::Kernel kernel1, // MatMul kernel
    cl::Kernel kernel2, // FW kernel
    cl::CommandQueue& q1,// Command Queue
    uint64_t& kernel_duration,
    int l_M,
    int l_K,
    int l_N,
    int l_LdA,
    int l_LdB,
    int l_LdC,
    int l_LdX,
    int l_postScale,
    int offset,
    offsets Offset
)
{
    int mid = dim/2;
    cl::Event event;

    if(dim<=BSIZE){
    	printf("For DIM = %d:\n", dim);
	printf("For MID = %d:\n", mid);
    	//Set the kernel arguments
    	int narg = 0;
    	kernel2.setArg(narg++, buf);
	kernel2.setArg(narg++, buf_out);
    	//kernel2.setArg(narg++, x);
    	//kernel2.setArg(narg++, y);
    	kernel2.setArg(narg++, dim);
	kernel2.setArg(narg++, offset);
	//printf("\tPerforming FW...\n");
    	//Launch the kernel
    	q1.enqueueTask(kernel2, NULL, &event);
	//wait();
	//printf("\tFinished.\n");
	
	//q1.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST);
        // Display the current matrix:
        /*std::cout << "The matrix is: ";
        for (int ct = 0; ct < DATA_SIZE*DATA_SIZE; ct++){
	    if(ct % DATA_SIZE == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/
    	//q1.enqueueMigrateMemObjects({buf},0/* 0 means from host*/);
	
	//kernel_duration += get_duration_ns(event);
    }else{
	RKleene_fpga_helper(source_in1, buf, buf_out, x, y, mid, kernel1, kernel2, q1, kernel_duration, mid, mid, mid, mid, mid, mid, mid, l_postScale, offset, Offset);
    	printf("For DIM = %d:\n", dim);
	printf("For MID = %d:\n", mid);
	//printf("\tPerforming B += A*B...\n");
	//B += A*B
    	int narg = 0;

	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf_out);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        kernel1.setArg(narg++, l_LdA);
        kernel1.setArg(narg++, l_LdB);
        kernel1.setArg(narg++, l_LdC);
        kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);
        kernel1.setArg(narg++, Offset);

	//Launch the kernel
    	q1.enqueueTask(kernel1, NULL, &event);
	//wait();
	//printf("\tFinished.\n");
	
	//q1.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST);
        // Display the current matrix:
        /*std::cout << "The matrix is: ";
        for (int ct = 0; ct < DATA_SIZE*DATA_SIZE; ct++){
	    if(ct % DATA_SIZE == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/
    	//q1.enqueueMigrateMemObjects({buf},0/* 0 means from host*/);

	//kernel_duration += get_duration_ns(event);

	//printf("\tPerforming C += C*A...\n");
	//C += C*A
    	narg = 0;
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf_out);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        kernel1.setArg(narg++, l_LdA);
        kernel1.setArg(narg++, l_LdB);
        kernel1.setArg(narg++, l_LdC);
        kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);
        kernel1.setArg(narg++, Offset);
	//Launch the kernel
    	q1.enqueueTask(kernel1, NULL, &event);
	//wait();
	//printf("\tFinished.\n");
	
	//q1.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST);
        // Display the current matrix:
        /*std::cout << "The matrix is: ";
        for (int ct = 0; ct < DATA_SIZE*DATA_SIZE; ct++){
	    if(ct % DATA_SIZE == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/
    	//q1.enqueueMigrateMemObjects({buf},0/* 0 means from host*/);

	//kernel_duration += get_duration_ns(event);

	//printf("\tPerforming D += C*B...\n");
	//D += C*B
    	narg = 0;
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf_out);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        kernel1.setArg(narg++, l_LdA);
        kernel1.setArg(narg++, l_LdB);
        kernel1.setArg(narg++, l_LdC);
        kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);
        kernel1.setArg(narg++, Offset);
	//Launch the kernel
    	q1.enqueueTask(kernel1, NULL, &event);
	//wait();
	//printf("\tFinished.\n");
	
	//q1.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST);
        // Display the current matrix:
        /*std::cout << "The matrix is: ";
        for (int ct = 0; ct < DATA_SIZE*DATA_SIZE; ct++){
	    if(ct % DATA_SIZE == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/
    	//q1.enqueueMigrateMemObjects({buf},0/* 0 means from host*/);

	//kernel_duration += get_duration_ns(event);

	RKleene_fpga_helper(source_in1, buf, buf_out, x+mid, y+mid, dim-mid, kernel1, kernel2, q1, kernel_duration, mid, mid, mid, mid, mid, mid, mid, l_postScale, offset, Offset);
    	//printf("For DIM = %d:\n", dim);
	//printf("For MID = %d:\n", mid);
	//printf("\tPerforming B += B*D...\n");
	//B += B*D
    	narg = 0;
/*	kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf_out);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        kernel1.setArg(narg++, l_LdA);
        kernel1.setArg(narg++, l_LdB);
        kernel1.setArg(narg++, l_LdC);
        kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);
        kernel1.setArg(narg++, Offset);
	//Launch the kernel
    	q1.enqueueTask(kernel1, NULL, &event);
	//wait();
	//printf("\tFinished.\n");
	
	//q1.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST);
        // Display the current matrix:
        /*std::cout << "The matrix is: ";
        for (int ct = 0; ct < DATA_SIZE*DATA_SIZE; ct++){
	    if(ct % DATA_SIZE == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/
    	//q1.enqueueMigrateMemObjects({buf},0/* 0 means from host*/);

	//kernel_duration += get_duration_ns(event);

	//printf("\tPerforming C += D*C...\n");
	//C += D*C
    	narg = 0;
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf_out);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        kernel1.setArg(narg++, l_LdA);
        kernel1.setArg(narg++, l_LdB);
        kernel1.setArg(narg++, l_LdC);
        kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);
        kernel1.setArg(narg++, Offset);
	//Launch the kernel
    	q1.enqueueTask(kernel1, NULL, &event);
	//wait();
	//printf("\tFinished.\n");
	
	//q1.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST);
        // Display the current matrix:
        /*std::cout << "The matrix is: ";
        for (int ct = 0; ct < DATA_SIZE*DATA_SIZE; ct++){
	    if(ct % DATA_SIZE == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/
    	//q1.enqueueMigrateMemObjects({buf},0/* 0 means from host*/);

	//kernel_duration += get_duration_ns(event);

	//printf("\tPerforming A += B*C...\n");
	//A += B*C
    	narg = 0;
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, x+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, dim-mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	/*kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf_out);
	kernel1.setArg(narg++, x);
	kernel1.setArg(narg++, y+mid);
	kernel1.setArg(narg++, y);
	kernel1.setArg(narg++, mid);
	kernel1.setArg(narg++, DATA_SIZE);*/
	kernel1.setArg(narg++, buf);
	kernel1.setArg(narg++, buf);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        kernel1.setArg(narg++, l_LdA);
        kernel1.setArg(narg++, l_LdB);
        kernel1.setArg(narg++, l_LdC);
        kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);
        kernel1.setArg(narg++, Offset);
	//Launch the kernel
    	q1.enqueueTask(kernel1, NULL, &event);
	//wait();
	//printf("\tFinished.\n");
	
	//q1.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST);
        // Display the current matrix:
        /*std::cout << "The matrix is: ";
        for (int ct = 0; ct < DATA_SIZE*DATA_SIZE; ct++){
	    if(ct % DATA_SIZE == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/
    	//q1.enqueueMigrateMemObjects({buf},0/* 0 means from host*/);

	//kernel_duration += get_duration_ns(event);
    }
}

uint64_t Both_fpga (
    std::string l_xclbinFile,  //xclbinFile
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>>& source_in1,   //Input/Otput Matrix 1
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>>& source_fpga_results,
    int l_M,
    int l_K,
    int l_N,
    int l_LdA,
    int l_LdB,
    int l_LdC,
    int l_LdX,
    int l_postScale,
    int dim,                                         //One dimension of matrix
    int offset,
    offsets Offset
)
{
    cl::Event l_event_1, l_event_2, l_event_3;
    uint64_t kernel_duration = 0;

    //The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    //Creating Context and Command Queue for selected Device
    cl::Context context(device);
    cl::CommandQueue q1(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    //import_binary() command will find the OpenCL binary file created using the
    //xocc compiler load into OpenCL Binary and return as Binaries
    //OpenCL and it can contain many functions which can be executed on the
    //device.;
    cl::Program::Binaries bins = xcl::import_binary_file(l_xclbinFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    //This call will extract a kernel out of the program we loaded in the
    //previous line. A kernel is an OpenCL function that is executed on the
    //FPGA. This function is defined in the src/mmult.cl file.
    cl::Kernel kernel1(program,"GemmCall");
    cl::Kernel kernel2(program,"FW");
    std::vector<cl::Memory> m_Buffers;

	//These commands will allocate memory on the FPGA. The cl::Buffer
	//objects can be used to reference the memory locations on the device.
	//The cl::Buffer object cannot be referenced directly and must be passed
	//to other OpenCL functions.
	cl::Buffer buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_in1.data());
	cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_fpga_results.data());
	m_Buffers.push_back(buffer);
	m_Buffers.push_back(buffer_output);
	std::cout<<"[INFO] cl::Buffer size_in_bytes: "<<matrix_size_bytes<<std::endl;

    	//These commands will load the source_in1 and source_in2 vectors from the host
    	//application into the buffer_in1 and buffer_in2 cl::Buffer objects. The data
    	//will be be transferred from system memory over PCIe to the FPGA on-board
    	//DDR memory.

	std::vector<cl::Event> m_Mem2FpgaEvents;
	std::vector<cl::Event> m_ExeKernelEvents;

//    	q1.enqueueMigrateMemObjects({buffer},0/* 0 means from host*/);
//    	q1.enqueueMigrateMemObjects(m_Buffers,0/* 0 means from host*/,NULL,&l_event_1);
/*	m_Mem2FpgaEvents.push_back(l_event_1);
	
        int narg = 0;
	kernel1.setArg(narg++, buffer);
	kernel1.setArg(narg++, buffer);
    	kernel1.setArg(narg++, l_M);
        kernel1.setArg(narg++, l_K);
        kernel1.setArg(narg++, l_N);
        //kernel1.setArg(narg++, l_LdA);
        //kernel1.setArg(narg++, l_LdB);
        //kernel1.setArg(narg++, l_LdC);
        //kernel1.setArg(narg++, l_LdX);
        kernel1.setArg(narg++, l_postScale);
*/

	//Launch the kernel
//    	q1.enqueueTask(kernel1, NULL, &event);
/*	q1.enqueueTask(kernel1, &m_Mem2FpgaEvents, &l_event_2);
	m_ExeKernelEvents.push_back(l_event_2);
    	narg = 0;
	kernel2.setArg(narg++, buffer);
	//kernel1.setArg(narg++, buffer_gemm);
	kernel2.setArg(narg++, buffer_output);
	kernel2.setArg(narg++, dim/4);
	kernel2.setArg(narg++, offset);
	//kernel1.setArg(narg++, DATA_SIZE);
	//Launch the kernel
    	q1.enqueueTask(kernel2, &m_Mem2FpgaEvents, &l_event_3);
	m_ExeKernelEvents.push_back(l_event_3);
	m_Mem2FpgaEvents.clear();
*/

	RKleene_fpga_helper(source_in1, buffer, buffer_output, 0, 0, dim, kernel1, kernel2, q1, kernel_duration, l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX, l_postScale, offset, Offset);
//    	q1.enqueueMigrateMemObjects({buffer},CL_MIGRATE_MEM_OBJECT_HOST);
    	q1.enqueueMigrateMemObjects(m_Buffers,CL_MIGRATE_MEM_OBJECT_HOST, &m_ExeKernelEvents);
	m_ExeKernelEvents.clear();

    	q1.finish();

    return kernel_duration;
}

uint64_t callFW (
    std::string l_xclbinFile,  //xclbinFile
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>>& source_in1,   //Input Matrix 1
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>>& source_fpga_results,    //Output Matrix
    //cl::CommandQueue& q1,
    //cl::Context context,
    // std::vector<cl::Device> devices,
    // cl::Program::Binaries bins,
    //cl::Program program,
    int dim,                                         //One dimension of matrix
    int offset
)
{
    int size = dim;
    //size_t matrix_size_bytes = sizeof(int) * size * size;
    printf("Size(B): %d\n", matrix_size_bytes);
    printf("Dim: %d\n", dim);
    cl::Event event;//, event1, event2;
    uint64_t kernel_duration = 0;

    // //The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    //
    // //Creating Context and Command Queue for selected Device
    cl::Context context(device);
    cl::CommandQueue q1(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
    //
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    //
    // //import_binary() command will find the OpenCL binary file created using the
    // //xocc compiler load into OpenCL Binary and return as Binaries
    // //OpenCL and it can contain many functions which can be executed on the
    // //device.
    //std::string binaryFile = xcl::find_binary_file(device_name,"Kleene");
    cl::Program::Binaries bins = xcl::import_binary_file(l_xclbinFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    //This call will extract a kernel out of the program we loaded in the
    //previous line. A kernel is an OpenCL function that is executed on the
    //FPGA. This function is defined in the src/mmult.cl file.
    cl::Kernel kernel1(program,"FW");

  	//These commands will allocate memory on the FPGA. The cl::Buffer
  	//objects can be used to reference the memory locations on the device.
  	//The cl::Buffer object cannot be referenced directly and must be passed
  	//to other OpenCL functions.
  	cl::Buffer buffer_in1(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_in1.data());
  	cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_fpga_results.data());

  	//These commands will load the source_in1 and source_in2 vectors from the host
  	//application into the buffer_in1 and buffer_in2 cl::Buffer objects. The data
  	//will be be transferred from system memory over PCIe to the FPGA on-board
  	//DDR memory.
    // std::cout << "The FPGA in numbers are!!: ";
    // for (int ct = 0; ct < size*size; ct++){
    //   if(ct % size == 0)    std::cout << std::endl << ct / size << " | ";
    //     std::cout << source_in1[ct] << " ";
    // }
    // std::cout << std::endl;
  	q1.enqueueMigrateMemObjects({buffer_in1},0/* 0 means from host*/);
  	//q1.enqueueMigrateMemObjects({buffer_gemm},0/* 0 means from host*/);
    q1.enqueueMigrateMemObjects({buffer_output},0/* 0 means from host*/);

    int narg = 0;
    kernel1.setArg(narg++, buffer_in1);
    //kernel1.setArg(narg++, buffer_gemm);
    kernel1.setArg(narg++, buffer_output);
  	kernel1.setArg(narg++, dim/2);
  	kernel1.setArg(narg++, offset);
    //kernel1.setArg(narg++, DATA_SIZE);
    //Launch the kernel
    q1.enqueueTask(kernel1, NULL, &event);
    //wait();
    //kernel_duration += get_duration_ns(event);

  	q1.enqueueMigrateMemObjects({buffer_in1},CL_MIGRATE_MEM_OBJECT_HOST);
  	//q1.enqueueMigrateMemObjects({buffer_gemm},CL_MIGRATE_MEM_OBJECT_HOST);
  	q1.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
  	// q1.finish();

    return kernel_duration;
}

//Functionality to setup OpenCL context and trigger the Kernel
uint64_t RKleene_fpga (
    std::string l_xclbinFile,  //xclbinFile
    std::vector<short,aligned_allocator<short>>& source_in1,   //Input Matrix 1
    //std::vector<short,aligned_allocator<short>>& source_gemm,   //GEMM buffer
    std::vector<short,aligned_allocator<short>>& source_fpga_results,    //Output Matrix
    std::vector<short,aligned_allocator<short>>& source_cpu_results,  //CPU results
    int dim,                                         //One dimension of matrix
    int offset
)
{
    int size = dim;
    size_t matrix_size_bytes = sizeof(short) * size * size;
    printf("Size(B): %d\n", matrix_size_bytes);
    printf("Dim: %d\n", dim);
    cl::Event event;//, event1, event2;
    uint64_t kernel_duration = 0;

    //The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    //Creating Context and Command Queue for selected Device
    cl::Context context(device);
    cl::CommandQueue q1(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    //import_binary() command will find the OpenCL binary file created using the
    //xocc compiler load into OpenCL Binary and return as Binaries
    //OpenCL and it can contain many functions which can be executed on the
    //device.
    //std::string binaryFile = xcl::find_binary_file(device_name,"Kleene");
    cl::Program::Binaries bins = xcl::import_binary_file(l_xclbinFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    //This call will extract a kernel out of the program we loaded in the
    //previous line. A kernel is an OpenCL function that is executed on the
    //FPGA. This function is defined in the src/mmult.cl file.
    cl::Kernel kernel1(program,"FW");

	//These commands will allocate memory on the FPGA. The cl::Buffer
	//objects can be used to reference the memory locations on the device.
	//The cl::Buffer object cannot be referenced directly and must be passed
	//to other OpenCL functions.
	cl::Buffer buffer_in1(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_in1.data());
	//cl::Buffer buffer_gemm(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_gemm.data());
	cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_fpga_results.data());

/*
        std::cout << "The numbers are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;
*/

    	//These commands will load the source_in1 and source_in2 vectors from the host
    	//application into the buffer_in1 and buffer_in2 cl::Buffer objects. The data
    	//will be be transferred from system memory over PCIe to the FPGA on-board
    	//DDR memory.

    	q1.enqueueMigrateMemObjects({buffer_in1},0/* 0 means from host*/);
    	//q1.enqueueMigrateMemObjects({buffer_gemm},0/* 0 means from host*/);
	q1.enqueueMigrateMemObjects({buffer_output},0/* 0 means from host*/);

    		    int narg = 0;
		    kernel1.setArg(narg++, buffer_in1);
		    //kernel1.setArg(narg++, buffer_gemm);
		    kernel1.setArg(narg++, buffer_output);
	    	    kernel1.setArg(narg++, dim/2);
		    kernel1.setArg(narg++, offset);
		    //kernel1.setArg(narg++, DATA_SIZE);
		    //Launch the kernel
    		    q1.enqueueTask(kernel1, NULL, &event);
		    //wait();
		    //kernel_duration += get_duration_ns(event);

    	q1.enqueueMigrateMemObjects({buffer_in1},CL_MIGRATE_MEM_OBJECT_HOST);
    	//q1.enqueueMigrateMemObjects({buffer_gemm},CL_MIGRATE_MEM_OBJECT_HOST);
    	q1.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
    	q1.finish();

    return kernel_duration;
}

int main(int argc, char **argv)
{
  //############  UI and GEMM problem size  ############
/*  if (argc < 2) {
    std::cerr << "Usage:\n"
              <<  "  gemx_api_gemm.exe <path/gemx.xclbin> [M K N  [LdA LdB LdC LdX postScaleVal postScaleShift] ]\n"
              <<  "  Examples:\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin  256 256 256\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin  256 256 256  256 256 256\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin  256 256 256  256 256 256  256 1 0\n";
    exit(2);
  }
*/
  unsigned int l_argIdx = 1;
  std::string l_xclbinFile(argv[l_argIdx]);
  unsigned int l_ddrW = GEMX_ddrWidth;
  // the smallest matrices for flow testing
  unsigned int l_M = DATA_SIZE/2,  l_K = DATA_SIZE/2,  l_N = DATA_SIZE/2;

  unsigned int l_LdA = l_K,  l_LdB = l_N,  l_LdC = l_N, l_LdX = l_N;
  int32_t l_postScaleVal = 1, l_postScaleShift = 0;

  int32_t l_postScale = (l_postScaleVal << 8) | (l_postScaleShift & 0x000000ff);

  offsets Offset;
  Offset.A = 0; Offset.B = 32*32; Offset.X = 2*32*32; Offset.C = 3*32*32;

    //Allocate Memory in Host Memory
    int size = DATA_SIZE;

    //size_t matrix_size_bytes = sizeof(DATA_TYPE) * size * size;// + sizeof(DATA_TYPE) * size * size / 4;

    printf("Main - %dx%dx%d, matrix_size_bytes: %d\n", l_M, l_K, l_N, matrix_size_bytes);

    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>> source_in1(matrix_size_bytes);// /sizeof(GEMX_dataType));
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>> source_cpu_results(matrix_size_bytes);// /sizeof(GEMX_dataType));

    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>> source_fpga(matrix_size_bytes);// /sizeof(GEMX_dataType));
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>> source_fpga_results(matrix_size_bytes);// /sizeof(GEMX_dataType));

    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>> source_inA(matrix_ABsize_bytes);// /sizeof(GEMX_dataType));
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>> source_inB(matrix_ABsize_bytes);// /sizeof(GEMX_dataType));
    std::vector<GEMX_XdataType,aligned_allocator<GEMX_XdataType>> source_inX(matrix_Xsize_bytes);// /sizeof(GEMX_dataType));
    std::vector<GEMX_dataType,aligned_allocator<GEMX_dataType>> source_outC(matrix_Csize_bytes);// /sizeof(GEMX_dataType));

    //Create the test data and Software Result
    int loc = 0;
    loop_readA:for(int i = 0; i < matrix_ABsize_bytes/sizeof(GEMX_dataType); i++, loc++) {
        source_in1[loc] = source_inA[i] = rand() % 10 +1;
	source_fpga[loc] = source_in1[loc];
    }
    loop_readBX:for(int i = 0; i < matrix_ABsize_bytes/sizeof(GEMX_dataType); i++, loc++) {
        source_in1[loc] = source_inB[i] = rand() % 10 +1;
	source_fpga[loc] = source_in1[loc];
    }
    loop_readX:for(int i = 0; i < matrix_Xsize_bytes/sizeof(GEMX_XdataType); i++, loc++) {
	source_in1[loc] = source_inX[i] = rand() % 10 +1;
	source_fpga[loc] = source_in1[loc];
    }
    //Provided to initialize C to various values
    int C_start = loc;
    loop_readC:for(int i = 0; i < matrix_ABsize_bytes/sizeof(GEMX_dataType); i++, loc++) {
	//source_in1[loc] = source_outC[i] = 0;
	source_in1[loc] = source_outC[i] = rand()%10 +1;
	source_fpga[loc] = source_in1[loc];
    }

        // Display the numbers read:
        std::cout << "The numbers are: ";
        /*std::cout << std::endl << "A:";
        for (int ct = 0; ct < matrix_size_bytes/sizeof(GEMX_dataType); ct++){
          if(ct == matrix_ABsize_bytes/sizeof(GEMX_dataType)) std::cout << std::endl << "B:";
          if(ct == 2*matrix_ABsize_bytes/sizeof(GEMX_dataType)) std::cout << std::endl << "X:";
          if(ct == 2*matrix_ABsize_bytes/sizeof(GEMX_dataType)+matrix_Xsize_bytes/sizeof(GEMX_XdataType)) std::cout << std::endl << "C:";
          if(ct%size == 0)    std::cout << std::endl << ct/size << " | ";
          std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;*/

    uint64_t kernel_duration = 0;
    int offset = 4*3*size*size/4;

    std::cout << "Computing MM on CPU...\n";
    //MatMul(source_inA.data(), source_inB.data(), source_inX.data(), source_outC.data(), size, size, size);

    //FW_cpu(source_in1.data(), source_cpu_results.data(), DATA_SIZE*2);

       // Display the numbers produced:

        std::cout << "The MM results are: ";
        /*std::cout << std::endl << "C:";
        for (int ct = 0; ct < matrix_Csize_bytes/sizeof(GEMX_dataType); ct++){
          if(ct%size == 0)    std::cout << std::endl << ct/size << " | ";
          std::cout << source_outC[ct] << " ";
        }
        std::cout << std::endl;
	
       for (int ct = 0; ct < matrix_size_bytes/sizeof(GEMX_dataType); ct++){
         if(ct%size == 0)    std::cout << std::endl << ct/size << " | ";
         std::cout << source_cpu_results[ct] << " ";
       }
        std::cout << std::endl;*/
    std::cout << "Computing MM on FPGA... \n";

    //Compute FPGA Results
    //kernel_duration = GEMM_fpga(l_xclbinFile, source_in1, l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX, l_postScale);
    //kernel_duration = callFW(l_xclbinFile, source_fpga, source_fpga_results, DATA_SIZE*2, offset);
    //kernel_duration = RKleene_fpga(l_xclbinFile, source_fpga, /*source_gemm,*/ source_fpga_results, source_cpu_results, size*2, offset);

    kernel_duration = Both_fpga(l_xclbinFile, source_in1, source_fpga_results, l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX, l_postScale, size*2, offset, Offset);

       std::cout << "The FPGA results are: ";
    /*   std::cout << std::endl << "C:";
       for (int ct = C_start; ct < matrix_size_bytes/sizeof(GEMX_dataType); ct++){
         if(ct%size == 0)    std::cout << std::endl << ct/size << " | ";
         std::cout << source_in1[ct] << " ";
       }
        std::cout << std::endl;
	*/
       /*for (int ct = 0; ct < matrix_size_bytes/sizeof(GEMX_dataType); ct++){
         if(ct%size == 0)    std::cout << std::endl << ct/size << " | ";
         std::cout << source_in1[ct] << " ";
       }
        std::cout << std::endl;*/
    std::cout << "Finished. \n";
    //Compare the results of the FPGA to CPU
    bool match = true;
    for (int i = 0 ; i < 4*matrix_Csize_bytes/sizeof(GEMX_dataType); i++, C_start++){
        if (/*source_outC[i] != source_in1[C_start]*/source_cpu_results[i] != source_in1[i]){
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << /*source_outC[i]*/source_cpu_results[i]
                << " FPGA result = " << /*source_in1[C_start]*/source_in1[i] << std::endl;
            match = false;
            break;
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    std::cout << "Wall Clock Time (Kernel execution): " << kernel_duration << std::endl;
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution only,"
            << "not for emulation." << std::endl;

    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);

}
