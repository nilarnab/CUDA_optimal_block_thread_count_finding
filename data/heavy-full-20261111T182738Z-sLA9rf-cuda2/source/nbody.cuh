/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// Authored force-only adaptation of NVIDIA's tiled all-pairs algorithm (see provenance.json).
// Grid-stride target tiles keep every thread in each block at the same barriers.
#ifndef HEAVY_NBODY_CUH
#define HEAVY_NBODY_CUH
__host__ __device__ inline void heavy_interaction(float4 p,float4 q,float& ax,float& ay,float& az) {
    const float dx=q.x-p.x,dy=q.y-p.y,dz=q.z-p.z;
    const float inv=rsqrtf(dx*dx+dy*dy+dz*dz+0.01f);
    const float weight=q.w*inv*inv*inv;
    ax+=dx*weight;ay+=dy*weight;az+=dz*weight;
}
#ifndef HEAVY_CPU_TEST
__global__ void nbody_force(const float4* positions,float* force,int count) {
    extern __shared__ float4 tile[];
    for(int base=blockIdx.x*blockDim.x;base<count;base+=gridDim.x*blockDim.x) {
        const int i=base+threadIdx.x;
        const float4 p=i<count?positions[i]:make_float4(0,0,0,0);
        float ax=0,ay=0,az=0;
        for(int start=0;start<count;start+=blockDim.x) {
            const int source=start+threadIdx.x;
            tile[threadIdx.x]=source<count?positions[source]:make_float4(0,0,0,0);
            __syncthreads();
            for(int j=0;j<blockDim.x && start+j<count;j++) {
                heavy_interaction(p,tile[j],ax,ay,az);
            }
            __syncthreads();
        }
        if(i<count){force[3*i]=ax;force[3*i+1]=ay;force[3*i+2]=az;}
    }
}
#endif
#endif
