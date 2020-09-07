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
#define MAX_SIZE 32

#define add(a,b) (((a)<(b))?(a):(b))
#define e_a INT_MAX
#define mul(a,b) ((a)+(b))
#define e_m 0


//TRIPCOUNT identifier
const unsigned int c_size = MAX_SIZE;

typedef struct Func{
    int type;
    int i;
    int ja;
    int jb;
    int k;
    int dim;
}func;

typedef gemx::WideType<GEMX_dataType, GEMX_ddrWidth> DdrWideType;
typedef gemx::WideType<GEMX_XdataType, GEMX_XddrWidth> XDdrWideType;

typedef hls::stream<DdrWideType> DdrStream;
typedef hls::stream<XDdrWideType> XDdrStream;


/*void GemmCall(
      long *p_aAddr,
      long *p_bAddr,
      long *p_cAddr,
			long *p_xAddr,
      unsigned int p_aColBlocks,
      unsigned int p_aRowBlocks,
      unsigned int p_bColBlocks,
      unsigned int p_aLd,
	  	unsigned int p_bLd,
	  	unsigned int p_cLd,
			unsigned int p_xLd,
	  	unsigned int p_transpBlocks,
			int p_postScale
    	){
        GemmBlocks(p_aAddr, p_bAddr, p_cAddr, p_xAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, p_cLd, p_xLd, p_transpBlocks, p_postScale);
}*/

extern "C" {
void FW( int* in,
	 int* out,
	 int dim
       )
{
    #pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmemm
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmemm
    #pragma HLS INTERFACE s_axilite port=in bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=dim bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int A[MAX_SIZE][MAX_SIZE];
    int B[MAX_SIZE][MAX_SIZE];
    int total = dim*dim;
    //#pragma HLS ARRAY_PARTITION variable = A dim = 2 factor = 4 complete
    //#pragma HLS ARRAY_PARTITION variable = B dim = 1 factor = 4 complete
    /*A = in*/
    for (int iter = 0, i = 0, j = 0; iter < total; iter++, j++) 
    {
       #pragma HLS PIPELINE II=1
        if (j == dim) 
	{
            j = 0;
            i++;
        }
        A[i][j] = in[iter];
        B[i][j] = A[i][j];
    }    

    //PHASE 1
    for (int i=1; i<dim; i++)//PIPELINE |
    {
        #pragma HLS PIPELINE II=1
	#pragma HLS LOOP_TRIPCOUNT min=31 max=31 avg=31
	for(int k=0; k<i; k++)//PIPELINE |
	{
            #pragma HLS PIPELINE II=1
	    #pragma HLS LOOP_TRIPCOUNT min=1 max=31 avg=16 
	    for(int j=0; j<MAX_SIZE; j++)//UNROLL | row or column
	    {
		#pragma HLS unroll
	    	if(A[i][j] > A[i][k] + A[k][j]) A[i][j] = A[i][k] + A[k][j];
	    }
	    for(int j=0; j<MAX_SIZE; j++)//UNROLL | row or column
	    {
		#pragma HLS unroll
	    	if(B[j][i] > B[j][k] + B[k][i]) B[j][i] = B[j][k] + B[k][i];
	    }
	}
    }

    //PHASE 2
    for (int i=0; i<dim-1; i++)//PIPELINE
    {
        #pragma HLS PIPELINE II=1
	#pragma HLS LOOP_TRIPCOUNT min=31 max=31 avg=31
	for(int k=i; k<dim; k++)//PIPELINE |
	{  
            #pragma HLS PIPELINE II=1
	    #pragma HLS LOOP_TRIPCOUNT min=1 max=31 avg=16  
	    for(int j=0; j<MAX_SIZE; j++)//UNROLL | row or column
	    {
		#pragma HLS unroll
	    	if(A[i][j] > A[i][k] + A[k][j]) A[i][j] = A[i][k] + A[k][j];
	    }
	    for(int j=0; j<MAX_SIZE; j++)//UNROLL | row or column
	    {
		#pragma HLS unroll
	    	if(B[j][i] > B[j][k] + B[k][i]) B[j][i] = B[j][k] + B[k][i];
	    }
	}
    }

    /*out = A*/
    for (int iter = 0, i = 0, j = 0; iter < total; iter++, j++) 
    {
       #pragma HLS PIPELINE II=1
        if (j == dim) 
	{
            j = 0;
            i++;
        }
	A[i][j] = B[i][j];
        out[iter] = A[i][j];
    }  
}
}
