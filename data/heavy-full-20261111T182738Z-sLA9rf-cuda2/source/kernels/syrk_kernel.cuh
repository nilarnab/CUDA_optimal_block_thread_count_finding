__global__ void syrk_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *c)
{
	/*  C := alpha*A*A' + beta*C */
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NI))
	{
		c[i * NI + j] *= beta;
		int k;		
		for(k=0; k < _PB_NJ; k++)
		{
			c[i * NI + j] += alpha * a[i * NJ + k] * a[j * NJ + k];
		}
	}
}

