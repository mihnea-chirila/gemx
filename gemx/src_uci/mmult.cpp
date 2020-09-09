/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

//This function represents an OpenCL kernel. The kernel will be call from
//host application. The pointers in kernel parameters with the global
//keyword represents cl_mem objects on the FPGA DDR memory. Array partitioning
//and loop unrolling is done to achieve better performance.

//#include "mmult.h"
#include "gemx_gemm.h"
#include "gemx_kernel.h"
#include <bitset>

typedef gemx::WideType<GEMX_dataType, GEMX_ddrWidth> DdrWideType;
typedef gemx::WideType<GEMX_XdataType, GEMX_XddrWidth> XDdrWideType;

typedef hls::stream<DdrWideType> DdrStream;
typedef hls::stream<XDdrWideType> XDdrStream;

typedef struct Offsets{
    int A;
    int B;
    int X;
    int C;
}offsets;

// Creating and passing buffer to GEMX
extern "C" {
void GemmCall( DdrWideType* p_DdrRd,  //input/output matrix
		DdrWideType* p_DdrWr,
    	     int l_M,  
	     int l_K,  
	     int l_N,
    	     int l_LdA,  
	     int l_LdB,  
	     int l_LdC, 
	     int l_LdX,
    	     int l_postScaleVal,
	     offsets Offset
           )
{
    #pragma HLS INLINE self off
    #pragma HLS INTERFACE m_axi port=p_DdrRd offset=slave bundle=gmemm num_write_outstanding=16 num_read_outstanding=16 max_write_burst_length=16 max_read_burst_length=16 depth=16 latency=125
    #pragma HLS INTERFACE m_axi port=p_DdrWr offset=slave bundle=gmemm num_write_outstanding=16 num_read_outstanding=16 max_write_burst_length=16 max_read_burst_length=16 depth=16 latency=125
    #pragma HLS INTERFACE s_axilite port=p_DdrRd bundle=control
    #pragma HLS INTERFACE s_axilite port=p_DdrWr bundle=control
    #pragma HLS INTERFACE s_axilite port=l_M bundle=control
    #pragma HLS INTERFACE s_axilite port=l_K bundle=control
    #pragma HLS INTERFACE s_axilite port=l_N bundle=control
    //#pragma HLS INTERFACE s_axilite port=l_LdA bundle=control
    //#pragma HLS INTERFACE s_axilite port=l_LdB bundle=control
    //#pragma HLS INTERFACE s_axilite port=l_LdC bundle=control
    //#pragma HLS INTERFACE s_axilite port=l_LdX bundle=control
    #pragma HLS INTERFACE s_axilite port=l_postScaleVal bundle=control
    #pragma HLS INTERFACE s_axilite port=aOffset bundle=control
    #pragma HLS INTERFACE s_axilite port=bOffset bundle=control
    #pragma HLS INTERFACE s_axilite port=cOffset bundle=control
    #pragma HLS INTERFACE s_axilite port=xOffset bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS DATA_PACK variable=p_DdrRd
    #pragma HLS DATA_PACK variable=p_DdrWr

    GemmType l_gemm;

    DdrWideType *l_aAddr = p_DdrRd + Offset.A;
    DdrWideType *l_bAddr = p_DdrRd + Offset.B;//l_M*l_K/GEMX_ddrWidth;
    DdrWideType *l_xAddr = p_DdrRd + Offset.X;//l_M*l_K/GEMX_ddrWidth + l_K*l_N/GEMX_ddrWidth;
    DdrWideType *l_cAddr = p_DdrWr + Offset.C;//l_M*l_K/GEMX_ddrWidth + l_K*l_N/GEMX_ddrWidth + l_M*l_N/GEMX_XddrWidth;

    int t_aColMemWords = GEMX_gemmKBlocks, t_aRowMemWords = GEMX_gemmMBlocks, t_bColMemWords = GEMX_gemmNBlocks;
        	const unsigned int l_aColBlocks = l_K / (GEMX_ddrWidth * t_aColMemWords);
        	const unsigned int l_aRowBlocks = l_M / (GEMX_ddrWidth * t_aRowMemWords);
        	const unsigned int l_bColBlocks = l_N / (GEMX_ddrWidth * t_bColMemWords);
        	const unsigned int l_aLd  = l_K / GEMX_ddrWidth;
        	const unsigned int l_bLd  = l_N / GEMX_ddrWidth;
        	const unsigned int l_cLd  = l_N / GEMX_ddrWidth;
		const unsigned int l_xLd = l_N / GEMX_XddrWidth;
    unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks *t_aRowMemWords;

      #ifndef __SYNTHESIS__
    	std::cout<<"aOffset = "<<aOffset<<std::endl;
    	std::cout<<"bOffset = "<<bOffset<<std::endl;
    	std::cout<<"cOffset = "<<cOffset<<std::endl;
    	std::cout<<"xOffset = "<<xOffset<<std::endl;
    	std::cout<<"p_DdrRd = "<<p_DdrRd<<std::endl;
//    	std::cout<<"l_aAddr = "<<l_aAddr<<std::endl;
//	std::cout<<"l_bAddr = "<<l_bAddr<<std::endl;
//	std::cout<<"l_cAddr = "<<l_cAddr<<std::endl;
//	std::cout<<"l_xAddr = "<<l_xAddr<<std::endl;
//        std::cout<<"l_M = "<<l_M<<std::endl;
//        std::cout<<"l_K = "<<l_K<<std::endl;
//        std::cout<<"l_N = "<<l_N<<std::endl;
//        std::cout<<"l_LdA = "<<l_LdA<<std::endl;
//        std::cout<<"l_LdB = "<<l_LdB<<std::endl;
//        std::cout<<"l_LdC = "<<l_LdC<<std::endl;
//        std::cout<<"l_LdX = "<<l_LdX<<std::endl;
//        std::cout<<"l_postScaleVal = "<<l_postScaleVal<<std::endl;
//        std::cout<<"l_transpBlocks = "<<l_transpBlocks<<std::endl;
					std::cout<<"l_aAddr = "<<l_aAddr<<std::endl;
					std::cout<<"l_bAddr = "<<l_bAddr<<std::endl;
					std::cout<<"l_cAddr = "<<l_cAddr<<std::endl;
					std::cout<<"l_xAddr = "<<l_xAddr<<std::endl;
					std::cout<<"l_aColBlocks = "<<l_aColBlocks<<std::endl;
					std::cout<<"l_aRowBlocks = "<<l_aRowBlocks<<std::endl;
					std::cout<<"l_bColBlocks = "<<l_bColBlocks<<std::endl;
					std::cout<<"l_aLd = "<<l_aLd<<std::endl;
					std::cout<<"l_bLd = "<<l_bLd<<std::endl;
					std::cout<<"l_cLd = "<<l_cLd<<std::endl;
					std::cout<<"l_xLd = "<<l_xLd<<std::endl;
					std::cout<<"l_transpBlocks = "<<l_transpBlocks<<std::endl;
					std::cout<<"l_postScaleVal = "<<l_postScaleVal<<std::endl;
					std::cout<<"l_M = "<<l_M<<std::endl;
					std::cout<<"l_N = "<<l_N<<std::endl;
					std::cout<<"l_K = "<<l_K<<std::endl;
      #endif

    l_gemm.GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_xAddr,l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd, l_xLd, l_transpBlocks, l_postScaleVal);
    #ifndef __SYNTHESIS__
      std::cout<<"Finished!"<<std::endl;
    #endif
}
}
