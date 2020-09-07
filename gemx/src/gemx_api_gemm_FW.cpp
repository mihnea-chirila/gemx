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

//Max Array Size
#define MAX_SIZE 8192

//Array Size to access
#define DATA_SIZE 16

//Block Size
#define BSIZE 16

#define add(a,b) (((a)<(b))?(a):(b))
#define e_a 10000
#define mul(a,b) ((a)+(b))
#define e_m 0

//#define VERBOSE 0

uint64_t get_duration_ns (const cl::Event &event) {
    uint64_t nstimestart, nstimeend;
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend);
    return(nstimeend-nstimestart);
}

//Matrix multiply, out = in + in1 x in2
void MatMul(int *in1, int *in2, int *in, int *out, int outRow, int outCol, int midSize){
  for(int i = 0; i < outRow; i++) {
    for(int j = 0; j < outCol; j++) {
        out[i * outCol + j] = in[i * outCol + j];//e_a;
        for(int k = 0; k < midSize; k++) {
            //out[i * outCol + j] = add(out[i * outCol + j], mul(in1[i * midSize + k], in2[k * outCol + j]));
	    out[i * outCol + j] += in1[i * midSize + k] * in2[k * outCol + j];
        }
    	//out[i * outCol + j] = add(in[i * outCol + j], out[i * outCol + j]);
    }
  }
}

//CPU implementation of Floyd-Warshall
//The inputs are of the size (DATA_SIZE x DATA_SIZE)
void FW_cpu (
    short *in,   //Input Matrix
    short *out,   //Output Matrix
    int dim     //One dimension of matrix
)
{
    //Initialize output matrix to input matrix
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
	    out[i * dim + j] = in[i * dim + j];
	    printf("%d ", out[i * dim + j]);
	}
	printf("\n");
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
uint64_t RKleene_fpga (
    std::string l_xclbinFile,  //xclbinFile
    std::vector<short,aligned_allocator<short>>& source_in1,   //Input Matrix 1
    std::vector<short,aligned_allocator<short>>& source_gemm,   //GEMM buffer
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
	cl::Buffer buffer_gemm(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes, source_gemm.data());
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
		    kernel_duration += get_duration_ns(event);

    	q1.enqueueMigrateMemObjects({buffer_in1},CL_MIGRATE_MEM_OBJECT_HOST);
    	//q1.enqueueMigrateMemObjects({buffer_gemm},CL_MIGRATE_MEM_OBJECT_HOST);
    	q1.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
    	q1.finish();

    return kernel_duration;
}

bool checkDim(unsigned int p_Val, unsigned int p_Mod, unsigned int p_Min) {
  bool l_ok = true;
  if (p_Val % p_Mod != 0) {
    std::cerr << "ERROR: value " << p_Val << " must be multiple of " << p_Mod << "\n";
    l_ok = false;
  }
  if (p_Val < p_Min) {
    std::cerr << "ERROR: value " << p_Val << " must be at least " << p_Min << "\n";
    l_ok = false;
  }
  return(l_ok);
}

float getBoardFreqMHz(unsigned int p_BoardId) {
  std::string l_freqCmd = "$XILINX_XRT/bin/xbsak query -d" + std::to_string(p_BoardId);;
  float l_freq = -1;
  char l_lineBuf[256];
  std::shared_ptr<FILE> l_pipe(popen(l_freqCmd.c_str(), "r"), pclose);
  //if (!l_pipe) throw std::runtime_error("ERROR: popen(" + l_freqCmd + ") failed");
  if (!l_pipe) std::cout << ("ERROR: popen(" + l_freqCmd + ") failed");
  bool l_nextLine_isFreq = false;
  while (l_pipe && fgets(l_lineBuf, 256, l_pipe.get()) ) {
    std::string l_line(l_lineBuf);
    //std::cout << "DEBUG: read line " << l_line << std::endl;
    if (l_nextLine_isFreq) {
      std::string l_prefix, l_val, l_mhz;
      std::stringstream l_ss(l_line);
      l_ss >> l_prefix >> l_val >> l_mhz;
      l_freq = std::stof(l_val);
      assert(l_mhz == "MHz");
      break;
    } else if (l_line.find("OCL Frequency:") != std::string::npos) {
      l_nextLine_isFreq = true;
    }
  }
  if (l_freq == -1) {
	//if xbutil does not work, user could put the XOCC achieved kernel frequcy here
	l_freq = 250;
        std::cout << "INFO: Failed to get board frequency by xbutil. This is normal for cpu and hw emulation, using 250 MHz for reporting.\n";
  }
  return(l_freq);
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
/*
  // Row major  C  M rows N cols  =  A  M rows K cols  *  B  K rows N cols
  //   MatType - tensor like type to allocate/store/align memory; you can use your own type instead
  //   Min size is the array edge (e.g., 32 on ku115), see GenGemm::check() for example of arg checking functions
  unsigned int l_ddrW = GEMX_ddrWidth;
  // the smallest matrices for flow testing
  unsigned int l_M = l_ddrW * GEMX_gemmMBlocks,  l_K = l_ddrW * GEMX_gemmKBlocks,  l_N = l_ddrW * GEMX_gemmNBlocks;
  if (argc > ++l_argIdx) {l_M = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_K = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_N = atoi(argv[l_argIdx]);}
  unsigned int l_LdA = l_K,  l_LdB = l_N,  l_LdC = l_N, l_LdX = l_N;
  int32_t l_postScaleVal = 1, l_postScaleShift = 0;
  if (argc > ++l_argIdx) {l_LdA = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_LdB = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_LdC = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_LdX = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_postScaleVal = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_postScaleShift= atoi(argv[l_argIdx]);}
  int32_t l_postScale = (l_postScaleVal << 8) | (l_postScaleShift & 0x000000ff);

  if (! (
      checkDim(l_M, l_ddrW * GEMX_gemmMBlocks, l_ddrW * GEMX_gemmMBlocks) &&
      checkDim(l_K, l_ddrW * GEMX_gemmKBlocks, l_ddrW * GEMX_gemmKBlocks) &&
      checkDim(l_N, l_ddrW * GEMX_gemmNBlocks, l_ddrW * GEMX_gemmNBlocks) &&
      checkDim(l_LdA, l_ddrW, l_K) &&
      checkDim(l_LdB, l_ddrW, l_N) &&
      checkDim(l_LdC, l_ddrW, l_N) &&
      checkDim(l_LdX, l_ddrW, l_N)
    )) {
    return EXIT_FAILURE;
  }

  printf("GEMX-gemm C++ API example using accelerator image \n",
         l_xclbinFile.c_str());

  //############  Client code - prepare the gemm problem input  ############
  GenGemm l_gemm;
  ProgramType l_program[GEMX_numKernels];  // Holds instructions and controls memory allocation

  std::string l_handleA[GEMX_numKernels];
  std::string l_handleB[GEMX_numKernels];
  std::string l_handleC[GEMX_numKernels];
  std::string l_handleX[GEMX_numKernels];

  for (int i=0; i<GEMX_numKernels; ++i) {
	l_handleA[i] = "A"+std::to_string(i);
	l_handleB[i] = "B"+std::to_string(i);
	l_handleC[i] = "C"+std::to_string(i);
	l_handleX[i] = "X"+std::to_string(i);

    l_gemm.addInstr(l_program[i], l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX, l_postScale, l_handleA[i], l_handleB[i], l_handleC[i], l_handleX[i], false);
    std::cout << "In kernel " << i << " ";
    std::cout << "Added instruction GEMM (" << l_M << "x" << l_K <<" * "<< l_K << "x" << l_N << " + " << l_M << "x" << l_N << ") * " << l_postScaleVal <<" >> " << l_postScaleShift <<"\n";
  }

  std::string kernelNames[GEMX_numKernels];
  gemx::MemDesc l_memDesc[GEMX_numKernels];

  for (int i=0; i<GEMX_numKernels; ++i) {
  	l_memDesc[i] = l_program[i].getMemDesc();
  }

  //############  Runtime reporting Infra  ############
  TimePointType l_tp[10];
  unsigned int l_tpIdx = 0;
  l_tp[l_tpIdx] = std::chrono::high_resolution_clock::now();

  //############  Run FPGA accelerator  ############
  // Init FPGA
  gemx::Fpga l_fpga;

  for (int i=0; i<GEMX_numKernels; ++i){
	kernelNames[i] = "gemxKernel_" + std::to_string(i);
  }
  if (l_fpga.loadXclbin(l_xclbinFile) && l_fpga.createKernel(0, kernelNames)) {
      std::cout << "INFO: created kernels" << std::endl;
  } else {
      std::cerr << "ERROR: failed to load " + l_xclbinFile + "\n";
      return EXIT_FAILURE;
  }
  showTimeData("loadXclbin", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  //create buffers for transferring data to FPGA
  if (!l_fpga.createBufferForKernel(0, l_memDesc)) {
      std::cerr << "ERROR: failed to create buffers for transffering data to FPGA DDR\n";
      return EXIT_FAILURE;
  }
  showTimeData("created buffers", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
  //mchirila--------

  printf("\n!!!Writing to bin\n");
  std::string l_binFile = "outBuff.txt";
  l_program[0].writeToBinFile(l_binFile);
  //----------------
  // Transfer data to FPGA
  if (l_fpga.copyToKernel(0)) {
      (VERBOSE > 0) && std::cout << "INFO: transferred data to FPGA" << std::endl;
  } else {
      std::cerr << "ERROR: failed to copy data to FPGA DDR\n";
      return EXIT_FAILURE;
  }
  showTimeData("copyToFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  // Gemx kernel ops
  if (l_fpga.callKernel(0)) {
      (VERBOSE > 0) && std::cout << "INFO: Executed kernel" << std::endl;
  } else {
      std::cerr << "ERROR: failed to call kernels ";
      for (int i=0; i<GEMX_numKernels; ++i) {
	std::cerr << kernelNames[i] << " ";
      }
      std::cerr << "\n";
      return EXIT_FAILURE;
  }
  showTimeData("callKernel", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  // Transfer data back to host - due to lazy evaluation this is generally wheer the accelerator performs the work
  if (l_fpga.copyFromKernel(0)) {
      (VERBOSE > 0) && std::cout << "INFO: Transferred data from FPGA" << std::endl;
  } else {
      std::cerr << "ERROR: failed to copy data from FPGA DDR\n";
      return EXIT_FAILURE;
  }
  showTimeData("copyFromFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
  showTimeData("total", l_tp[0], l_tp[l_tpIdx]); l_tpIdx++;
  double l_timeApiInMs = -1;
  showTimeData("subtotalFpga", l_tp[2], l_tp[l_tpIdx], &l_timeApiInMs); l_tpIdx++; // Host->DDR, kernel, DDR->host
  l_fpga.finish();

  std::cout<<"\n[INFO][hting1] POST show begin"<<std::endl; //Note: hting1: hardwire to show program[0] only
  KargsType l_kargs;
  l_kargs.load(l_program[0].getBaseInstrAddr(), 0);
  GemmArgsType l_gemmArgs = l_kargs.getGemmArgs();
  l_gemm.show(l_program[0], l_gemmArgs);
  std::cout<<"\n[INFO][hting1] POST show end"<<std::endl;


  //############  Get the exact kernel time from HW cycle counters on the accelerator  ############
  float l_boardFreqMHz = getBoardFreqMHz(0);
  unsigned long int l_Ops = 2ull * l_M * l_N * l_K + l_M * l_N * 3;
  unsigned long int l_Parallel_Ops = 2ull * l_M * l_N * l_K;
  KargsType l_kargsRes[GEMX_numKernels];
  KargsOpType l_op[GEMX_numKernels];
  gemx::InstrResArgs l_instrRes[GEMX_numKernels];
  unsigned long int l_cycleCount[GEMX_numKernels];
  unsigned long int l_maxCycleCount=0;
  double l_timeKernelInMs[GEMX_numKernels];
  double l_maxTimeKernelInMs=0;
  double l_perfKernelInTops[GEMX_numKernels];
  double l_totalPerfKernelInTops=0;
  double l_perfApiInTops;
  double l_timeMsAt100pctEff;
  double l_effKernelPct;
  double l_effApiPct;

  for (int i=0; i<GEMX_numKernels; ++i) {
  	l_op[i] = l_kargsRes[i].load(l_program[i].getBaseResAddr(), 0);
  	assert(l_op[i] == KargsType::OpResult);
  	l_instrRes[i] = l_kargsRes[i].getInstrResArgs();
  	l_cycleCount[i] = l_instrRes[i].getDuration();
    l_maxCycleCount = (l_cycleCount[i] > l_maxCycleCount)? l_cycleCount[i]: l_maxCycleCount;
  	l_timeKernelInMs[i] = l_cycleCount[i] / (l_boardFreqMHz * 1e6) * 1e3;
    l_maxTimeKernelInMs = (l_timeKernelInMs[i] > l_maxTimeKernelInMs)? l_timeKernelInMs[i]: l_maxTimeKernelInMs;
	l_perfKernelInTops[i] = l_Ops / (l_timeKernelInMs[i] * 1e-3) / 1e12;
    l_totalPerfKernelInTops += l_perfKernelInTops[i];
  }
  l_perfApiInTops = (l_Ops*GEMX_numKernels) / (l_timeApiInMs * 1e-3) / 1e12;
  l_timeMsAt100pctEff = l_Parallel_Ops / 2 / GEMX_ddrWidth / GEMX_ddrWidth / (l_boardFreqMHz * 1e6) * 1e3;
  l_effKernelPct = (100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs < 100)?(100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs):100;
  l_effApiPct = 100 * l_timeMsAt100pctEff / l_timeApiInMs;
  // Show time, Tops in csv format
  std::cout << std::string("DATA_CSV:,DdrWidth,Freq,M,K,N,")
             + "Ops,KernelCycles,"
             + "TimeKernelMs,TimeApiMs,"
             + "EffKernelPct,EffApiPct,"
             + "PerfKernelTops,PerfApiTops\n"
            << "DATA_CSV:," <<  GEMX_ddrWidth << "," << l_boardFreqMHz << ","
            << l_M << "," << l_K << "," << l_N << ","
            << l_Ops*GEMX_numKernels << "," << l_maxCycleCount << ","
            << l_maxTimeKernelInMs << "," << l_timeApiInMs << ","
            << l_effKernelPct << "," << l_effApiPct << ","
            << l_totalPerfKernelInTops << "," << l_perfApiInTops
            << std::endl;

  return EXIT_SUCCESS;
*/
///*
    if (DATA_SIZE > MAX_SIZE) {
        std::cout << "Size is bigger than internal buffer size,"
        << " please use a size smaller than " << MAX_SIZE << "!" << std::endl;
        return EXIT_FAILURE;
    }

    //Allocate Memory in Host Memory
    int size = DATA_SIZE;
    int dim = BSIZE;
    size_t matrix_size_bytes = sizeof(short) * size * size;
    size_t matrix_dim_bytes = sizeof(short) * dim * dim;

    //When creating a buffer with user pointer, under the hood user ptr is
    //used if and only if it is properly aligned (page aligned). When not
    //aligned, runtime has no choice but to create its own host side buffer
    //that backs user ptr. This in turn implies that all operations that move
    //data to/from device incur an extra memcpy to move data to/from runtime's
    //own host buffer from/to user pointer. So it is recommended to use this
    //allocator if user wish to Create Buffer/Memory Object to align user buffer
    //to the page boundary. It will ensure that user buffer will be used when
    //user create Buffer/Mem Object.
    std::vector<short,aligned_allocator<short>> source_in1(matrix_size_bytes);

    std::vector<short,aligned_allocator<short>> source_inA(matrix_dim_bytes);
    std::vector<short,aligned_allocator<short>> source_inB(matrix_dim_bytes);
    std::vector<short,aligned_allocator<short>> source_inX(matrix_dim_bytes);
    std::vector<short,aligned_allocator<short>> source_outC(matrix_dim_bytes);

    std::vector<short,aligned_allocator<short>> source_gemm(matrix_size_bytes);
    std::vector<short,aligned_allocator<short>> source_fpga_results(matrix_size_bytes);
    std::vector<short,aligned_allocator<short>> source_cpu_results(matrix_size_bytes);

    //Create the test data and Software Result
    for(int i = 0 ; i < DATA_SIZE * DATA_SIZE ; i++){
        //if(i % DATA_SIZE == i / DATA_SIZE) source_in1[i] = 1; else source_in1[i] = 0;
	source_in1[i] = rand() % DATA_SIZE;
	//source_in1[i] = 1;
        //source_cpu_results[i] = e_a;
        source_cpu_results[i] = source_in1[i];
        //source_fpga_results[i] = e_a;
    }
    loop_readA:for(int loc = 0, i = 0, j = 0; loc < /*BSIZE * DATA_SIZE*/ size*size/4; i++, j++, loc++) {
        //if(j == dim) { loc = loc + DATA_SIZE -j; j = 0;}
        source_inA[i] = source_in1[loc];
    }
    loop_readBX:for(int loc = /*BSIZE*/ size*size/4, i = 0, j = 0; loc < /*BSIZE * DATA_SIZE*/ 2*size*size/4; i++, j++, loc++) {
        //if(j == dim) { loc = loc + DATA_SIZE -j; j = 0;}
        source_inB[i] = source_in1[loc];
	//source_inX[i] = source_inB[i];
    }
    loop_readX:for(int loc = /*BSIZE*/ 2*size*size/4, i = 0, j = 0; loc < /*BSIZE * DATA_SIZE*/ 3*size*size/4; i++, j++, loc++) {
        //if(j == dim) { loc = loc + DATA_SIZE -j; j = 0;}
        //source_inB[i] = source_in1[loc];
	source_inX[i] = source_in1[loc];
    }
//*/
        // Display the numbers read:
///*
        std::cout << "The numbers are!!: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl << ct / size << " | ";
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;
//*/
///*
    uint64_t kernel_duration = 0;
    int offset = 3*size*size/4;
    FW_cpu(source_in1.data()+offset, source_cpu_results.data()+offset, size/2);
    std::cout << "Computing FW on CPU...\n";
    //MatMul(source_inA.data(), source_inB.data(), source_inX.data(), source_outC.data(), BSIZE, BSIZE, BSIZE);

    loop_writeC:for(int loc = /*BSIZE*/ 3*size*size/4, i = 0, j = 0; loc < /*BSIZE * DATA_SIZE*/ 4*size*size/4; i++, j++, loc++) {
        //if(j == dim) { loc = loc + DATA_SIZE -j; j = 0;}
//        source_cpu_results[loc] = source_outC[i];
    }
//*/        // Display the numbers produced:
///*
        std::cout << "The FW results are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl;
            std::cout <<  source_cpu_results[ct] << " ";
        }

        std::cout << std::endl;
//*/
///*
    std::cout << "Finished. \n";
    //Compute CPU Results
    //int c = 0;
    //RKleene_cpu(source_in1.data(), source_cpu_results.data(), size, c);

    std::cout << "Computing R-Kleene on FPGA...\n";
/*
        std::cout << "The numbers are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;
*/
    //Test();
    //Compute FPGA Results
    kernel_duration = RKleene_fpga(l_xclbinFile, source_in1, source_gemm, source_fpga_results, source_cpu_results, size, offset);
//*/
        // Display the numbers produced:
///*
        std::cout << "The CPU results are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)   std::cout << std::endl << ct / size << " | ";
            std::cout <<  source_cpu_results[ct] << " ";
        }

        std::cout << std::endl;


       std::cout << "The FPGA results are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl << ct / size << " | ";
            std::cout <<  source_fpga_results[ct] << " ";
        }

        std::cout << std::endl;
//*/
///*
    std::cout << "Finished. \n";
    //Compare the results of the FPGA to CPU
    bool match = true;
    for (int i = 0 ; i < size * size; i++){
        if (source_cpu_results[i] != source_fpga_results[i]){
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_cpu_results[i]
                << " FPGA result = " << source_fpga_results[i] << std::endl;
            match = false;
            break;
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    std::cout << "Wall Clock Time (Kernel execution): " << kernel_duration << std::endl;
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution only,"
            << "not for emulation." << std::endl;

    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
//*/
}
