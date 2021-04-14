 #include "hardSwishPlugin.h"
 #include <cuda_fp16.h>


 template <typename T_DATA>
     __global__ void kernelCopy(
         int N,
         T_DATA* inputs,
         T_DATA* outputs
         )
 {
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     if (index < N){
         outputs[index] = inputs[index];
     }
     __syncthreads();
 }

 __global__ void hswishkernel(
    const int n,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        float temp = 0;
        if (x[i] < -3.0) {
            temp = 0.0;
        } else if (x[i] > 3.0) {
            temp = x[i];
        } else {
            temp = x[i] * (x[i] + 3.0) / 6.0;
        }
        y[i] = temp;
    }
}

 template <typename T>
 int hardSwishInference(
     int batchSize,
     int iC,
     int iH,
     int iW,
     T* inputs,
     T* outputs,
     cudaStream_t stream){
         // NCHW
         const int nThreads = 512;
         int lenCopy = iC * iH * iW;

         int nBlocksCopy = (int)((float)lenCopy / nThreads) + 1;

         for(int i=0; i < batchSize; ++i){
             // NOTE: kernelCopy kernel can be replaced with cudaMemcpy function
             kernelCopy<<<nBlocksCopy, nThreads, 0, stream>>>(lenCopy, inputs, outputs);
             outputs += lenCopy;

             hswishkernel<<<nBlocksCopy, nThreads, 0, stream>>>(lenCopy,  inputs, outputs);
             outputs += lenCopy;
             inputs += lenCopy;
         }

     cudaError_t err = cudaGetLastError();
     if ( cudaSuccess != err )
     {
         fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
         return 1;
     }
     return 0;
 }

 int HardSwishPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
 {

     return hardSwishInference(batchSize, iC, iH, iW, (float*)inputs[0], (float*)outputs[0], stream);
 }
