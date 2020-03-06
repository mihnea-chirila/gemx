/**********
 * Copyright (c) 2017-2019, Xilinx, Inc.
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
 *  @brief FPGA utilities
 *
 *  $DateTime: 2018/01/30 15:02:37 $
 */

#ifndef GEMX_FPGA_H
#define GEMX_FPGA_H

#include "assert.h"
#include "gemx_gen_bin.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iterator>
#include "xcl2.hpp"

//#define GEMX_numKernels GEMX_layers
namespace gemx {

class Fpga
{
  private:
    std::string  m_XclbinFile;

    cl::Context       m_Context;
    cl::CommandQueue  m_CommandQueue;
    cl::Program       m_Program;
    cl::Kernel        m_Kernels[GEMX_numKernels];
    std::vector<cl::Memory>				m_Buffers[GEMX_numKernels];
    std::vector<cl::Event>	   		m_Mem2FpgaEvents[GEMX_numKernels];
    std::vector<cl::Event>	   		m_ExeKernelEvents[GEMX_numKernels];

  public:
    Fpga()
    {}

    bool
    loadXclbin(std::string p_xclbinFile) {
    	bool ok = false;
			std::vector<cl::Device> l_devices = xcl::get_xil_devices();
			cl::Device l_device = l_devices[0];
			std::string l_deviceName = l_device.getInfo<CL_DEVICE_NAME>();
			std::cout << "INFO: device name is: " << l_deviceName << std::endl;
      // Create the OpenCL context, cmmandQueue and program 
      cl::Context l_context(l_device);
      m_Context = l_context;
			cl::CommandQueue l_cmdQueue(m_Context, l_device,  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
			m_CommandQueue = l_cmdQueue;
			cl::Program::Binaries l_bins = xcl::import_binary_file(p_xclbinFile);
			l_devices.resize(1);
			cl::Program l_program(m_Context, l_devices, l_bins);
      m_Program = l_program;
      ok = true;
      return(ok);
    }

		bool
		createKernel(unsigned int p_kernelId, std::string p_kernelName[GEMX_numKernels]) {
			bool ok = false;
			assert(p_kernelId < GEMX_numKernels);
			cl::Kernel l_kernel(m_Program, p_kernelName[p_kernelId].c_str());
			m_Kernels[p_kernelId] = l_kernel;
			ok = true;
			return(ok);
		}
		
		bool 
		createBufferForKernel(unsigned int p_kernelId, MemDesc p_memDesc[GEMX_numKernels]) {
			bool ok = false;
			
			assert(p_kernelId < GEMX_numKernels);
			cl::Buffer l_buffer(m_Context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, p_memDesc[p_kernelId].sizeBytes(), p_memDesc[p_kernelId].data());
			m_Buffers[p_kernelId].push_back(l_buffer);	
			m_Kernels[p_kernelId].setArg(0, l_buffer);
			m_Kernels[p_kernelId].setArg(1, l_buffer);
//			m_Kernels[p_kernelId].setArg(0, m_Buffers); // hting1 modified
//			m_Kernels[p_kernelId].setArg(1, m_Buffers);
			ok = true;
			return(ok);
		}
	
    bool
    copyToKernel(unsigned int p_kernelId) {
      bool ok = false;
      assert(p_kernelId < GEMX_numKernels);  
      cl::Event l_event;
      // Send the input data to the accelerator
			m_CommandQueue.enqueueMigrateMemObjects(m_Buffers[p_kernelId], 0/* 0 means from host*/, NULL, &l_event);
			m_Mem2FpgaEvents[p_kernelId].push_back(l_event);
			ok = true;
      return(ok);
    }

    bool
    callKernel(unsigned int p_kernelId) {
    	bool ok = false;
      assert(p_kernelId < GEMX_numKernels);

      cl::Event l_event;
//	m_Kernels[p_kernelId].setArg(0, m_Buffers); // not here
//	m_Kernels[p_kernelId].setArg(1, m_Buffers);
      m_CommandQueue.enqueueTask(m_Kernels[p_kernelId], &(m_Mem2FpgaEvents[p_kernelId]), &l_event);
			m_ExeKernelEvents[p_kernelId].push_back(l_event);
			m_Mem2FpgaEvents[p_kernelId].clear();
      ok = true;
      return(ok);
    }

    bool
    copyFromKernel(unsigned int p_kernelId) {
    	bool ok = false;
			assert(p_kernelId < GEMX_numKernels);
			cl::Event l_event;
      m_CommandQueue.enqueueMigrateMemObjects(m_Buffers[p_kernelId], CL_MIGRATE_MEM_OBJECT_HOST, &(m_ExeKernelEvents[p_kernelId]));
			m_ExeKernelEvents[p_kernelId].clear();
			ok=true;
      return(ok);
    }
		bool
		finish() {
			bool ok = false;
			m_CommandQueue.finish();
			ok = true;
			return(ok);
		}

};

} // namespace

#endif
