/******************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

#define trainNum 100000 // number of train data
#define testNum 1000 // number of test data

#define inLayout 10 // number of input layer's neurons
#define hideLayout 8 // number of hidden layer's neurons
#define outLayout 1 // number of output layer's neurons

#define initWeightMax 0.5 // max value of initial weight

#define eta (0.1f) // learn rate

#define iterMax 10000 // max iteration times

#define batchNum 32 // number of batches

#define BLOCKSIZE 16  
#define BLOCKSIZE_32 32 

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
	
	float *inputTrain, *inputTest, *outputTrain, *outputTest;
	
    inputTrain = (float*)malloc(100000 * 10 * sizeof(float));
	inputTest = (float*)malloc(1000 * 10 * sizeof(float));
	outputTrain = (float*)malloc(100000 * 1 * sizeof(float));
	outputTest = (float*)malloc(1000 * 1 * sizeof(float));
    dim3 dim_grid, dim_block;

	int sumTrain = 0, sumTest = 0;
	
    for (unsigned int i=0; i < 1000000; i++) { 
		inputTrain[i] = rand()%2; 
		sumTrain += inputTrain[i]; 
		if((i % 10 == 9){
			outputTrain[i / 10] = sumTrain % 2;
			sumTrain = 0;
		}
	}

    for (unsigned int i=0; i < 10000; i++) { 
		inputTest[i] = rand()%2;
		sumTest += inputTest[i]; 
		if((i % 10 == 9){
			outputTest[i / 10] = sumTest % 2;
			sumTest = 0;
		}
	}

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

	float *inputTrain_D, *inputTest_D, *outputTrain_D, *outputTest_D;
    cudaMalloc((float**) &inputTrain_D, sizeof(float) * 1000000);
    cudaMalloc((float**) &inputTest_D, sizeof(float) * 10000);
    cudaMalloc((float**) &outputTrain_D, sizeof(float) * 100000);
	cudaMalloc((float**) &outputTest_D, sizeof(float) * 1000);
		
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(inputTrain_D, inputTrain, trainNum * inLayout * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(inputTest_D, inputTest, testNum * inLayout * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(outputTrain_D, outputTrain, trainNum * outLayout * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(outputTest_D, outputTest, testNum * outLayout * sizeof(float), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
	
    BpMain(inputTrain, inputTest, outputTrain, outputTest);

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	

    // Free memory ------------------------------------------------------------

    free(inputTrain);
	free(inputTest);
	free(outputTrain);
	free(outputTest);

    cudaFree(inputTrain_D);
	cudaFree(inputTest_D);
	cudaFree(outputTrain_D);
	cudaFree(outputTest_D);

    return 0;

}

