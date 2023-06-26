#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <cmath>
#define INTERVAL 1000
using namespace std;
typedef long long ll;

const int dim = 128;
const int trainNum = 1024;
const int testNum = 128;

void init(float (*train)[dim], float (*test)[dim])
{
  for (int i = 0;i < testNum;i++)
		for (int k = 0;k < dim;k++)
			test[i][k] = rand() / double(RAND_MAX) * 1000;
	for (int i = 0;i < trainNum;i++)
		for (int k = 0;k < dim;k++)
			train[i][k] = rand() / double(RAND_MAX) * 1000;
}

void plain(float (*train)[dim], float (*test)[dim], float (*dist)[trainNum])
{
	for (int i = 0;i < testNum;i++)
	{
		for (int j = 0;j < trainNum;j++)
		{
			float sum = 0.0;
			for (int k = 0;k < dim;k++)
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sum += temp;
			}
			dist[i][j] = sqrtf(sum);
		}
	}
}

void timing(void (*func)(float (*train)[dim], float (*test)[dim], float (*dist)[trainNum]), float (*train)[dim], float (*test)[dim], float (*dist)[trainNum])
{
    timeval tv_begin, tv_end;
    int counter(0);
    double time = 0;
    gettimeofday(&tv_begin, 0);
    while(INTERVAL>time)
    {
        func(train, test, dist);
        gettimeofday(&tv_end, 0);
        counter++;
        time = ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec)*1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec)/1000.0;
    }
    cout<<time/counter<<","<<counter<<'\t';
}

__global__
void calcDist(float (*train)[dim], float (*test)[dim], float (*dist)[trainNum])
{
  //int row = threadIdx.x + blockIdx.x * blockDim.x;
  //int col = threadIdx.y + blockIdx.y * blockDim.y;
  int row = blockIdx.x;
  int col = threadIdx.x;
  if(row>=128||col>=1024) return;
  float sum = 0.0;
	for (int k = 0;k < 128;k++)
	{
		float temp = test[row][k] - train[col][k];
		temp *= temp;
		sum += temp;
	}
	dist[row][col] = sqrtf(sum);
}

void checkElementsAre(float (*dist)[trainNum], float (*distComp)[trainNum])
{
  float error = 0;
  for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
  printf("误差%f\n", error);
}

int main()
{
  float (*train)[dim];
  float (*test)[dim];
  float (*dist)[trainNum];
  float (*check)[trainNum];
  size_t size1 = trainNum * dim * sizeof(float);
  size_t size2 = testNum * dim * sizeof(float);
  size_t size3 = testNum * trainNum * sizeof(float);
  cudaMallocManaged(&train, size1);
  cudaMallocManaged(&test, size2);
  cudaMallocManaged(&dist, size3);
  cudaMallocManaged(&check, size3);
  init(train, test);
  
  timing(plain, train, test, check);
  
  cudaError_t addVectorsErr;
  cudaError_t asyncErr;
  //dim3 grid(128,1,1);
  //dim3 block(1,1024,1);
  calcDist<<<128, 1024>>>(train, test, dist);
  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));
  
  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
  
  checkElementsAre(dist, check);

  cudaFree(train);
  cudaFree(test);
  cudaFree(dist);
  cudaFree(check);
}