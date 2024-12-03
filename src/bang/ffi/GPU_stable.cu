/** In the cases that a PBN is too large, both shared memory and texture memory
 * are used to store it*/
#include "./shared/German.h"
#include "./shared/ltqnorm.h"

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <pybind11/pybind11.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include <bitset>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

namespace py = pybind11;

#define epsilon 1e-9;   // to adjust the precision of float number
#define RegPerThread 63 // this value is obtained via compiling command
#define maxAllowedSharedMemory 40960 // the maximum allowed shared memory 40KB

// declare texture reference
texture<int, 1, cudaReadModeElementType> texExtraF;
texture<unsigned short, 1, cudaReadModeElementType> texExtraFIndex;
texture<unsigned short, 1, cudaReadModeElementType> texCumExtraF;

/** store PBN directly*/
int n;
unsigned short *nf;
unsigned short *nv;
int *F;
unsigned short *varF;
float *cij;
float p;
int *g_positiveIndex;
int *g_negativeIndex;
int *g_npNode; // no perturbation node
int g_npLength;
int stateSize;
string outputName;
float precision;
float confidence;
float epsilon_twostate;
int blockInfor[2];

/**in case parent nodes are more than 5*/
int *extraF; // store the extra F, the total elements should equal to
             // extraFCount
unsigned short *extraFIndex; // store the function index which has extraF, the
                             // total elements should equal to extraFIndexCount
unsigned short *cumExtraF;   // cum number of extra F, the total elements should
                             // equal to extraFIndexCount+1
int extraFInitialIndex;      // used for initialising extraF
int extraFCount;             // total number of extra spaces
int extraFIndexCount;        // how many function needs extra space

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__constant__ int powNum[2][32]; // used for adding shifNums, powNum[0][*]=0
__constant__ int nodeNum[1];
__constant__ float constantP[1];
__constant__ unsigned short
    constantCumNf[2002]; // reserve 2002 elements, i.e., the maximum node number
                         // can be 2001
__constant__ int constantPositiveIndex[63];
__constant__ int constantNegativeIndex[63];
__constant__ float constantCij[15000]; // this is the maximum size of constant
                                       // cache we can use;

extern __shared__ int arrays[];

/* this GPU kernel function is used to initialize the random states */
__global__ void init(int seed, curandState_t *states) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // printf("idx %d", idx);

  /* we have to initialize the state */
  /* the seed can be the same for each core, here we pass the time in from the
   * CPU */
  /* the sequence number should be different for each core (unless you want all
   cores to get the same sequence of numbers for some reason - use thread id! */
  /* the offset is how much extra we advance in the sequence for each call, can
   * be 0 */
  curand_init(seed, idx, 0, &states[idx]);
}

/*
 * check whether the given state belongs to meta state 0. Return true if the
 * state belongs to meta state 0. offset is related to the number of threads =
 * #threads*n
 */
__device__ bool checkMetaStateInt(int stateSize, int *currentState,
                                  int offset) {

  // printf("stateSize=%d,positiveIndex[0]=%d,negativeIndex[0]=%d\n",stateSize,positiveIndex[0],negativeIndex[0]);
  for (int i = 0; i < stateSize; i++) {
    if (((constantPositiveIndex[i] & currentState[i + offset]) ^
         constantPositiveIndex[i]) != 0) {
      return false;
    }
  }
  for (int i = 0; i < stateSize; i++) {
    if ((constantNegativeIndex[i] & currentState[offset + i]) != 0)
      return false;
  }
  return true;
}

/*
 * check whether the given state belongs to meta state 0. Return true if the
 * state belongs to meta state 0. offset is related to the number of threads =
 * #threads*n
 */
__device__ bool checkMetaStateKernel(int currentState, int offset) {
  if (((constantPositiveIndex[offset] & currentState) ^
       constantPositiveIndex[offset]) != 0) {
    return false;
  }
  if ((constantNegativeIndex[offset] & currentState) != 0)
    return false;
  return true;
}

/**
 * kernel for n=1-128, maximum integer 4
 */
template <size_t stateSize>
__global__ void
kernel1(curandState_t *states, unsigned short *gpu_cumNv, int *gpu_F,
        unsigned short *gpu_varF, int *gpu_initialState, int *gpu_steps,
        long *gpu_stateA, long *gpu_stateB, int *gpu_transitionsLastChain,
        int *gpu_bridge, int *gpu_stateSize, int *gpu_extraF,
        int *gpu_extraFIndex, int *gpu_cumExtraF, int *gpu_extraFCount,
        int *gpu_extraFIndexCount, int *gpu_npLength, int *gpu_npNode) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  gpu_transitionsLastChain[idx * 4] = 0;
  gpu_transitionsLastChain[idx * 4 + 1] = 0;
  gpu_transitionsLastChain[idx * 4 + 2] = 0;
  gpu_transitionsLastChain[idx * 4 + 3] = 0;
  // if(idx==0)printf("start %d\n",idx);
  // int n = *gpu_n; //make variables local
  // int stateSize = *gpu_stateSize;
  // let's make shared memory
  unsigned short *cumNv = (unsigned short *)arrays;
  int *F;
  if ((constantCumNf[nodeNum[0]] + 1) % 2 != 0) {
    F = (int *)&cumNv[constantCumNf[nodeNum[0]] + 2]; // add allinment
  } else {
    F = (int *)&cumNv[constantCumNf[nodeNum[0]] + 1];
  }
  unsigned short *varF = (unsigned short *)&F[constantCumNf[nodeNum[0]]];
  int *l_extraF;
  if (gpu_cumNv[constantCumNf[nodeNum[0]]] % 2 != 0) {
    l_extraF = (int *)&varF[gpu_cumNv[constantCumNf[nodeNum[0]]] + 1];
  } else {
    l_extraF = (int *)&varF[gpu_cumNv[constantCumNf[nodeNum[0]]]];
  }
  int *l_extraFIndex = (int *)&l_extraF[*gpu_extraFCount];
  int *l_cumExtraF = (int *)&l_extraFIndex[*gpu_extraFIndexCount];
  int *npLength = (int *)&l_cumExtraF[*gpu_extraFIndexCount + 1];
  int *np = (int *)&npLength[1];
  int initialStateCopy[4];
  int initialState[4];
  // The first thread in the block does the allocation and initialization
  // and then shares the pointer with all other threads through shared memory,
  // so that access can easily be coalesced.

  if (threadIdx.x == 0) {

    for (int i = 0; i < constantCumNf[nodeNum[0]]; i++) {
      // nv[i] = gpu_nv[i];
      cumNv[i] = gpu_cumNv[i];
      F[i] = gpu_F[i];
      // cij[i] = gpu_cij[i];
      // printf("%d %d %f %d \nodeNum[0]", nv[i], F[i], cij[i], cumNv[i]);
    }

    cumNv[constantCumNf[nodeNum[0]]] = gpu_cumNv[constantCumNf[nodeNum[0]]];
    // printf("varF:\nodeNum[0]");
    for (int i = 0; i < cumNv[constantCumNf[nodeNum[0]]]; i++) {
      varF[i] = gpu_varF[i];
      // printf("%d \nodeNum[0]", varF[i]);
    }

    for (int i = 0; i < *gpu_extraFCount; i++) {
      l_extraF[i] = gpu_extraF[i];
    }
    for (int i = 0; i < *gpu_extraFIndexCount; i++) {
      l_extraFIndex[i] = gpu_extraFIndex[i];
      l_cumExtraF[i] = gpu_cumExtraF[i];
    }
    l_cumExtraF[*gpu_extraFIndexCount] = gpu_cumExtraF[*gpu_extraFIndexCount];
    npLength[0] = *gpu_npLength;
    for (int i = 0; i < npLength[0]; i++) {
      np[i] = gpu_npNode[i];
    }
  }

  __syncthreads();

  // if(idx==0)printf("in %d\nodeNum[0]",idx);
  int relativeIndex = idx * stateSize;
  for (int i = 0; i < stateSize; i++) {
    initialState[i] = gpu_initialState[relativeIndex + i];
    initialStateCopy[i] = initialState[i];
  }

  int steps = *gpu_steps;
  // Copy state to local memory for efficiency
  curandState_t localState = states[idx];
  // printf("idx %d", idx);
  float rand;
  bool perturbation = false;
  // int nv_size = constantCumNf[nodeNum[0]];
  int stateA = 0; // how many steps are in state A
  int stateB = 0;
  int transitions[2][2]; // maybe put this in shared memory to speed up
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      transitions[i][j] = 0;
    }
  }
  int bridge, index1;
  bridge = gpu_bridge[idx];
  for (int j = 0; j < steps; j++) {
    perturbation = false;
    // check perturbation
    int indexState = 0, indexShift = 0;
    int start = 0;
    for (int t = 0; t < *npLength; t++) {
      for (int i = start; i < np[t]; i++) {
        rand = curand_uniform(&localState);
        // if(idx==0) printf("\trand %d-%d: %f\nodeNum[0]",j,i,rand);
        if (rand < constantP[0]) {
          perturbation = true;
          indexState = i / 32;
          indexShift = indexState * 32;
          initialStateCopy[indexState] =
              initialStateCopy[indexState] ^
              (1 << (i - indexShift)); // might use constant memory to replace
                                       // i/32 and i%32
        }
      }
      start = np[t] + 1;
    }
    if (!perturbation) {
      indexShift = 0;
      indexState = 0;
      for (int i = 0; i < nodeNum[0]; i++) {
        if (indexShift == 32) {
          indexState++;
          indexShift = 0;
        }
        relativeIndex = 0;
        rand = curand_uniform(&localState);
        while (rand > constantCij[constantCumNf[i] + relativeIndex]) {
          relativeIndex++;
        }
        start = constantCumNf[i] + relativeIndex;
        int elementF = F[start];
        int startVarFIndex = cumNv[start];
        int resultStateSize = cumNv[start + 1] - startVarFIndex;
        int shifNum = 0;
        for (int ind = 0; ind < resultStateSize; ind++) {
          relativeIndex = varF[startVarFIndex + ind] / 32;
          relativeIndex = initialState[relativeIndex];
          if (((relativeIndex >> (varF[startVarFIndex + ind] % 32)) & 1) != 0) {
            shifNum += powNum[1][ind];
          }
        }
        if (shifNum > 32) {
          int tt = 0;
          // if(idx==0)printf("in %d when shifNum>32\nodeNum[0]",idx);
          while (l_extraFIndex[tt] != start) {
            tt++;
          }
          elementF = l_extraF[l_cumExtraF[tt] + ((shifNum - 32) / 32)];
          shifNum = shifNum % 32;
        }
        elementF = elementF >>
                   shifNum; // after shifting, the last bit will be the value;
        initialStateCopy[indexState] ^=
            (-(elementF & 1) ^ initialStateCopy[indexState]) &
            (1 << (i - indexState * 32));
        indexShift++;
      }
    }
    // if(idx==0)printf("in %d when simulation finished\nodeNum[0]",idx);
    // simulation finished
    // update initialState to the new state
    for (int i = 0; i < stateSize; i++) {
      initialState[i] = initialStateCopy[i];
      // if(idx==0)printf("initialState[%d]=%d",i,initialState[i]);
    }
    // if(idx==0)printf("\nodeNum[0]");
    relativeIndex = 0;
    while (relativeIndex < stateSize) {
      if (!checkMetaStateKernel(initialState[relativeIndex], relativeIndex)) {
        relativeIndex = 1000;
        stateB++;
        index1 = 0;
      }
      relativeIndex++;
    }
    if (relativeIndex == stateSize) {
      stateA++;
      index1 = 1;
    }
    transitions[bridge][index1]++;
    // need to update bridge for next time usage
    bridge = index1;
  }
  // update state
  states[idx] = localState;
  relativeIndex = 4 * idx;
  for (int i = 0; i < stateSize; i++)
    gpu_initialState[relativeIndex + i] = initialStateCopy[i];
  // copy local data to global data
  gpu_bridge[idx] = index1;
  gpu_stateA[idx] = stateA;
  gpu_stateB[idx] = stateB;

  // printf("idx=%d, finished!\nodeNum[0]",idx);
  gpu_transitionsLastChain[relativeIndex] = transitions[0][0];
  gpu_transitionsLastChain[relativeIndex + 1] = transitions[0][1];
  gpu_transitionsLastChain[relativeIndex + 2] = transitions[1][0];
  gpu_transitionsLastChain[relativeIndex + 3] = transitions[1][1];
  // printf("idx=%d,stateA=%d,stateB=%d,gpu_transitionsLastChain[%d]=%d,
  // gpu_transitionsLastChain[%d]=%d, gpu_transitionsLastChain[%d]=%d,
  // gpu_transitionsLastChain[%d]=%d\n",idx,stateA,stateB,relativeIndex,gpu_transitionsLastChain[relativeIndex],relativeIndex+1,gpu_transitionsLastChain[relativeIndex+1],relativeIndex+2,gpu_transitionsLastChain[relativeIndex+2],relativeIndex+3,gpu_transitionsLastChain[relativeIndex+3]);
}

/**
 * for nodeNum[0]=1-128, maximum 4 integers
 */
template <size_t stateSize>
__global__ void
kernelConverge1(curandState_t *states, unsigned short *gpu_cumNv, int *gpu_F,
                unsigned short *gpu_varF, int *gpu_initialState,
                float *gpu_mean, float *gpu_trajectory,
                int *gpu_trajectoryKernel, int *gpu_steps, int *gpu_stateSize,
                int *gpu_extraF, int *gpu_extraFIndex, int *gpu_cumExtraF,
                int *gpu_extraFCount, int *gpu_extraFIndexCount,
                int *gpu_npLength, int *gpu_npNode) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // let's make shared memory
  unsigned short *cumNv = (unsigned short *)arrays;
  int *F;
  if ((constantCumNf[nodeNum[0]] + 1) % 2 != 0) {
    F = (int *)&cumNv[constantCumNf[nodeNum[0]] + 2]; // add allinment
  } else {
    F = (int *)&cumNv[constantCumNf[nodeNum[0]] + 1];
  }
  unsigned short *varF = (unsigned short *)&F[constantCumNf[nodeNum[0]]];
  int *l_extraF;
  if (gpu_cumNv[constantCumNf[nodeNum[0]]] % 2 != 0) {
    l_extraF = (int *)&varF[gpu_cumNv[constantCumNf[nodeNum[0]]] + 1];
  } else {
    l_extraF = (int *)&varF[gpu_cumNv[constantCumNf[nodeNum[0]]]];
  }
  int *l_extraFIndex = (int *)&l_extraF[*gpu_extraFCount];
  int *l_cumExtraF = (int *)&l_extraFIndex[*gpu_extraFIndexCount];
  int *npLength = (int *)&l_cumExtraF[*gpu_extraFIndexCount + 1];
  int *np = (int *)&npLength[1];

  int initialStateCopy[stateSize];
  int initialState[stateSize];

  // printf("Finish outputting initial states!\nodeNum[0]");
  //  The first thread in the block does the allocation and initialization
  //  and then shares the pointer with all other threads through shared memory,
  //  so that access can easily be coalesced.

  if (threadIdx.x == 0) {
    for (int i = 0; i < constantCumNf[nodeNum[0]]; i++) {
      // nv[i] = gpu_nv[i];
      cumNv[i] = gpu_cumNv[i];
      F[i] = gpu_F[i];
      // cij[i] =  gpu_cij[i];
    }

    cumNv[constantCumNf[nodeNum[0]]] = gpu_cumNv[constantCumNf[nodeNum[0]]];
    // printf("varF:\nodeNum[0]");
    for (int i = 0; i < cumNv[constantCumNf[nodeNum[0]]]; i++) {
      varF[i] = gpu_varF[i];
      // printf("%d \nodeNum[0]", varF[i]);
    }
    for (int i = 0; i < *gpu_extraFCount; i++) {
      l_extraF[i] = gpu_extraF[i];
      // if(idx==0)printf("extraF[%d]=%d\nodeNum[0]",i,l_extraF[i]);
    }
    for (int i = 0; i < *gpu_extraFIndexCount; i++) {
      l_extraFIndex[i] = gpu_extraFIndex[i];
      l_cumExtraF[i] = gpu_cumExtraF[i];
      // if(idx==0)printf("l_extraFIndex[%d]=%d,
      // l_cumExtraF[%d]=%d\nodeNum[0]",i,l_extraFIndex[i], i, l_cumExtraF[i]);
    }
    l_cumExtraF[*gpu_extraFIndexCount] = gpu_cumExtraF[*gpu_extraFIndexCount];
    // if(idx==0)printf("l_cumExtraF[%d]=%d\nodeNum[0]",*gpu_extraFIndexCount,
    // l_cumExtraF[*gpu_extraFIndexCount]);
    npLength[0] = *gpu_npLength;
    for (int i = 0; i < npLength[0]; i++) {
      np[i] = gpu_npNode[i];
    }
  }

  __syncthreads();
  int relativeIndex = idx * stateSize;
  for (int i = 0; i < stateSize; i++) {
    initialState[i] = gpu_initialState[relativeIndex + i];
    initialStateCopy[i] = initialState[i];
  }

  // int steps = *gpu_steps;
  //  Copy state to local memory for efficiency
  curandState_t localState = states[idx];
  float rand;
  bool perturbation = false;
  int prefix = gridDim.x * blockDim.x;
  int offset2 =
      -prefix; // gpu_currentTrajectorySize is the current trajectory size
  int prefix2 = stateSize * prefix;
  int offset4 = relativeIndex - prefix2;

  float result = 0;
  float mean = 0;
  int elementF, startVarFIndex, resultStateSize, shifNum;
  for (int j = 0; j < *gpu_steps; j++) {
    result = 0;
    perturbation = false;
    offset2 += prefix;
    offset4 += prefix2;
    // check perturbation
    int indexState = 0, indexShift = 0;
    int start = 0;
    for (int t = 0; t < *npLength; t++) {
      for (int i = start; i < np[t]; i++) {
        rand = curand_uniform(&localState);
        // if(idx==0) printf("\trand %d-%d: %f\nodeNum[0]",j,i,rand);
        if (rand < constantP[0]) {
          perturbation = true;
          indexState = i / 32;
          indexShift = indexState * 32;
          initialStateCopy[indexState] =
              initialStateCopy[indexState] ^
              (1 << (i - indexShift)); // might use constant memory to replace
                                       // i/32 and i%32
        }
      }
      start = np[t] + 1;
    }
    if (!perturbation) {
      indexShift = 0;
      indexState = 0;
      for (int i = 0; i < nodeNum[0]; i++) {
        if (indexShift == 32) {
          indexState++;
          indexShift = 0;
        }
        relativeIndex = 0;
        rand = curand_uniform(&localState);
        while (rand > constantCij[constantCumNf[i] + relativeIndex]) {
          relativeIndex++;
        }
        start = constantCumNf[i] + relativeIndex;
        elementF = F[start];
        startVarFIndex = cumNv[start];
        resultStateSize = cumNv[start + 1] - startVarFIndex;
        shifNum = 0;
        for (int ind = 0; ind < resultStateSize; ind++) {
          relativeIndex = varF[startVarFIndex + ind] / 32;
          relativeIndex = initialState[relativeIndex];
          if (((relativeIndex >> (varF[startVarFIndex + ind] % 32)) & 1) != 0) {
            shifNum += powNum[1][ind];
          }
        }
        if (shifNum > 32) {
          int tt = 0;
          while (l_extraFIndex[tt] != start) {
            tt++;
          }
          elementF = l_extraF[l_cumExtraF[tt] + ((shifNum - 32) / 32)];
          // if(idx==0)printf("start=%d,tt=%d,cumExtraF[tt]=%d,
          // shifNum=%d,elementF=%d\nodeNum[0]",start,tt,l_cumExtraF[tt],shifNum,elementF);
          shifNum = shifNum % 32;
        }
        elementF = elementF >>
                   shifNum; // after shifting, the last bit will be the value;

        initialStateCopy[indexState] ^=
            (-(elementF & 1) ^ initialStateCopy[indexState]) &
            (1 << (i - indexState * 32));
        indexShift++;
      }
    }
    // simulation finished
    // update initialState to the new state
    float times = 1;

    for (int i = 0; i < stateSize; i++) {
      initialState[i] = initialStateCopy[i];
      gpu_trajectoryKernel[offset4 + i] = initialStateCopy[i];
      result += ((unsigned int)initialState[i]) * times;
      times *= (float)powNum[1][31] * 2;
    }
    mean += result;
    gpu_trajectory[idx + offset2] = result;
  }
  prefix2 = *gpu_steps; // prefix2 value changed
  mean = mean / prefix2;
  gpu_mean[idx * 2] = mean;
  offset2 = -prefix;
  float minu;
  float variance = 0;
  for (int i = 0; i < prefix2; i++) {
    offset2 += prefix;
    minu = gpu_trajectory[idx + offset2] - mean;
    variance += minu * minu;
  }
  variance = variance / (prefix2 - 1);
  gpu_mean[idx * 2 + 1] = variance;
  // update state
  states[idx] = localState;
  relativeIndex = stateSize * idx;
  for (int i = 0; i < stateSize; i++) {
    gpu_initialState[relativeIndex + i] = initialState[i];
  }
}

/**
 * run converge initial for nodeNum[0]=1-128, maximum 4 integers
 * gpu_currentTrajectorySize the current trajectory size, initially it is 0
 * gpu_steps how many steps to be simulated in this call
 */
template <size_t stateSize>
__global__ void kernelConvergeInitial1(
    curandState_t *states, unsigned short *gpu_cumNv, int *gpu_F,
    unsigned short *gpu_varF, int *gpu_initialState, int *gpu_steps,
    int *gpu_stateSize, int *gpu_extraF, int *gpu_extraFIndex,
    int *gpu_cumExtraF, int *gpu_extraFCount, int *gpu_extraFIndexCount,
    int *gpu_npLength, int *gpu_npNode) {

  // printf("Hello from kernelconvergeinitial\n");
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // one register

  // let's make shared memory
  unsigned short *cumNv =
      (unsigned short *)arrays; // MC: trzymamy wszystkie dane o PBNie w jednej
                                // duzej tabeli arrays
  // MC: czyli w arrays sa trzymane: cumNv | F | varF | l_extraF(co to?) |
  // l_extraFIndex | l_cumExtraF | npLength | np
  int *F;
  if ((constantCumNf[nodeNum[0]] + 1) % 2 != 0) {
    F = (int *)&cumNv[constantCumNf[nodeNum[0]] + 2]; // add allinment
  } else {
    F = (int *)&cumNv[constantCumNf[nodeNum[0]] + 1];
  }
  unsigned short *varF = (unsigned short *)&F[constantCumNf[nodeNum[0]]];
  int *l_extraF;
  if (gpu_cumNv[constantCumNf[nodeNum[0]]] % 2 != 0) {
    l_extraF = (int *)&varF[gpu_cumNv[constantCumNf[nodeNum[0]]] + 1];
  } else {
    l_extraF = (int *)&varF[gpu_cumNv[constantCumNf[nodeNum[0]]]];
  }
  int *l_extraFIndex = (int *)&l_extraF[*gpu_extraFCount];
  int *l_cumExtraF = (int *)&l_extraFIndex[*gpu_extraFIndexCount];
  int *npLength = (int *)&l_cumExtraF[*gpu_extraFIndexCount + 1];
  int *np = (int *)&npLength[1];

  // can be put in texture  //MC: z jakiegos powodu mowia na pamiec lokalna w
  // gpu "texture", nwm czy to jest to samo
  int initialStateCopy[stateSize]; // MC: jest mniej niz 4 * 32 = 128 wezlow
  int initialState[stateSize];     // MC: czyli statesize <= 4

  if (threadIdx.x == 0) { // MC: mozliwe warp divergence??
    for (int i = 0; i < constantCumNf[nodeNum[0]]; i++) {
      // nv[i] = gpu_nv[i];
      cumNv[i] = gpu_cumNv[i];
      F[i] = gpu_F[i];
      // cij[i] =  gpu_cij[i];
    }

    cumNv[constantCumNf[nodeNum[0]]] = gpu_cumNv[constantCumNf[nodeNum[0]]];
    // printf("varF:\nodeNum[0]");
    for (int i = 0; i < cumNv[constantCumNf[nodeNum[0]]]; i++) {
      varF[i] = gpu_varF[i];
      //	printf("i=%d, %d \t",i, varF[i]);
    }
    for (int i = 0; i < *gpu_extraFCount; i++) {
      l_extraF[i] = gpu_extraF[i];
    }
    for (int i = 0; i < *gpu_extraFIndexCount; i++) {
      l_extraFIndex[i] = gpu_extraFIndex[i];
      l_cumExtraF[i] = gpu_cumExtraF[i];
    }
    l_cumExtraF[*gpu_extraFIndexCount] = gpu_cumExtraF[*gpu_extraFIndexCount];
    npLength[0] = *gpu_npLength;
    for (int i = 0; i < npLength[0]; i++) {
      np[i] = gpu_npNode[i];
    }
    /*		if(idx==0)printf("varF:\nodeNum[0]");
                    for (int i = 0; i < cumNv[constantCumNf[nodeNum[0]]]; i++) {
                            if(idx==0)printf("i=%d, %d \nodeNum[0]",i, varF[i]);
                    }*/
  }

  __syncthreads();

  // int stateSize = *gpu_stateSize;
  int relativeIndex = idx * stateSize;
  for (int i = 0; i < stateSize; i++) {
    initialState[i] =
        gpu_initialState[relativeIndex + i]; // MC: kazdy thread wpisuje sobie
                                             // swoja czesc initialstateow
    initialStateCopy[i] = initialState[i];
  }

  int steps = *gpu_steps; // can be put in texture
  // Copy state to local memory for efficiency
  curandState_t localState = states[idx];
  float rand;
  bool perturbation = false;

  for (int j = 0; j < steps; j++) {
    perturbation = false;
    // check perturbation
    int indexState = 0, indexShift = 0;
    int start = 0;
    for (int t = 0; t < *npLength; t++) {
      for (int i = start; i < np[t];
           i++) { // MC: ostatni element np czyli tablicy z wezlami bez
                  // perturbacji to zawsze n (ilosc wezlow)
        // MC: czyli idziemy od 0 do np[0], od np[0] + 1 do np[1] itd po
        // wszystkich wezlach bez perturbacji
        rand = curand_uniform(&localState);
        // printf("RAND: %f\n", rand);
        // if(idx==0 && j<10) printf("\trand %d-%d: %f\nodeNum[0]",j,i,rand);
        if (rand < constantP[0]) { // stale takie jak perturbation sa trzymane w
                                   // jednoelementowych tablicach
          // printf("\nPERTURBATION!!!!\n");
          perturbation = true;
          indexState = i / 32;
          indexShift =
              indexState * 32; // MC: ciekawy sposob na liczenie i mod 32
          initialStateCopy[indexState] =
              initialStateCopy[indexState] ^
              (1 << (i - indexShift)); // might use constant memory to replace
                                       // i/32 and i%32
        }
      }
      start = np[t] + 1;
    }

    if (!perturbation) { // MC: jesli nie ma perturbacji to liczymy z funkcji
                         // boolowskich nastepny stan
      indexShift = 0;
      indexState = 0;
      for (int i = 0; i < nodeNum[0];
           i++) { // MC: indexState i indexShift to kolejno numer inta i numer
                  // bitu w incie ze stanem danego wezla.
        if (indexShift == 32) {
          indexState++;
          indexShift = 0;
        }
        relativeIndex = 0;
        rand = curand_uniform(&localState);
        while (rand > constantCij[constantCumNf[i] + relativeIndex]) {
          relativeIndex++; // MC: losujemy ktorej funkcji dla wezla uzywamy,
                           // chyba cij jest przepisywane w german_run na
                           // skumulowane prawdopodobienstwo
        }
        start = constantCumNf[i] +
                relativeIndex; // MC: start to indeks funkcji ktora wybieramy
        int elementF = F[start];
        int startVarFIndex =
            cumNv[start]; // MC: indeks numeru pierwszego wezla w funkcji F
        int resultStateSize = cumNv[start + 1] -
                              startVarFIndex; // MC: ile jest wezlow w funkcji F
        int shifNum = 0;
        for (int ind = 0; ind < resultStateSize;
             ind++) { // MC: lecimy po wszystkich wezlach funkcji F
          relativeIndex = varF[startVarFIndex + ind] / 32;
          relativeIndex = initialState[relativeIndex];
          if (((relativeIndex >> (varF[startVarFIndex + ind] % 32)) & 1) != 0) {
            shifNum += powNum[1][ind];
          }
        }
        if (shifNum > 32) {
          int tt = 0;
          while (l_extraFIndex[tt] != start) {
            tt++;
          }
          elementF = l_extraF[l_cumExtraF[tt] + ((shifNum - 32) / 32)];
          shifNum = shifNum % 32;
        }
        elementF = elementF >>
                   shifNum; // after shifting, the last bit will be the value;

        // relativeIndex = i / 32;
        initialStateCopy[indexState] ^=
            (-(elementF & 1) ^ initialStateCopy[indexState]) &
            (1 << (i -
                   indexState * 32)); // MC: xorujemy stany z tymi ktore znamy
        indexShift++;
      }
    }
    // simulation finished
    // update initialState to the new state
    for (int ll = 0; ll < stateSize; ll++)
      initialState[ll] = initialStateCopy[ll];
  }
  // update state
  states[idx] = localState;
  relativeIndex = stateSize * idx;
  for (int i = 0; i < stateSize; i++) {
    gpu_initialState[relativeIndex + i] = initialStateCopy[i];
  }
  // printf("idx=%d", idx);
  // MC: jak widac w ConvergeInitial nie zapisujemy nigdzie informacji o tym
  // jakie stany mielismy do tej pory, patrzymy tylko gdzie znalezlismy sie po
  // steps krokach.
}

__global__ void kernelUpdateTrajectory(unsigned short *gpu_cumNv,
                                       int *gpu_trajectoryKernel,
                                       int *gpu_steps, long *gpu_stateA,
                                       long *gpu_stateB,
                                       int *gpu_transitionsLastChain,
                                       int *gpu_bridge, int *gpu_stateSize) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  int stateSize = *gpu_stateSize; // make variables local

  int n = *gpu_steps; // use n to store steps
  int offset = idx * stateSize;
  int prefix = stateSize * blockDim.x * gridDim.x;
  int stateA = 0; // how many steps are in state A
  int stateB = 0;
  int transitions[2][2]; // maybe put this in shared memory to speed up
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      transitions[i][j] = 0;
    }
  }
  int index1, bridge;
  // printf("\n");
  if (checkMetaStateInt(stateSize, gpu_trajectoryKernel, offset)) {
    stateA++;
    index1 = 1;
  } else {
    stateB++;
    index1 = 0;
  }
  // printf("stateA=%d,stateB=%d\n",stateA,stateB);
  for (int j = 1; j < n; j++) {
    bridge = index1;
    offset = offset + prefix;

    if (checkMetaStateInt(stateSize, gpu_trajectoryKernel, offset)) {
      stateA++;
      index1 = 1;
    } else {
      stateB++;
      index1 = 0;
    }
    transitions[bridge][index1]++;
    // if(idx==0)printf("currentState=(%d,%d,%d),
    // check=%d\n",currentState._array[offset + 0],currentState._array[offset +
    // 1],currentState._array[offset + 2],index1);
    // if(idx==0)printf("initialState[%d]=%d,index1=%d\n",offsetBlock,initialState[offsetBlock],index1);
  }

  // printf("idx=%d,stateA=%d,stateB=%d\n",idx,stateA,stateB);
  // copy local data to global data
  gpu_bridge[idx] = index1;
  gpu_stateA[idx] = stateA;
  gpu_stateB[idx] = stateB;
  offset = idx * 4;
  gpu_transitionsLastChain[offset] = transitions[0][0];
  gpu_transitionsLastChain[offset + 1] = transitions[0][1];
  gpu_transitionsLastChain[offset + 2] = transitions[1][0];
  gpu_transitionsLastChain[offset + 3] = transitions[1][1];
  /*	printf(
           "idx=%d,stateA=%d, stateB=%d,
     transitions=%d,transitions=%d,transitions[1][0]=%d,transitions[1][1]=%d\n",
           idx, stateA,stateB,transitions[0][0], transitions[0][1],
     transitions[1][0], transitions[1][1]);*/
}

// int fromVector(vector<bool> myvector) {
//   int retval = 0;
//   int i = 0;
//   for (vector<bool>::iterator it = myvector.begin(); it != myvector.end();
//        it++, i++) {
//     if (*it) {
//       retval |= 1 << i;
//     }
//   }
//   return retval;
// }

/**consider the case where len is greater than 32*/
int fromVector(py::list myvector, int len) {
  int retval = 0;
  int i = 0;
  int prefix = 0;
  int other = 0;
  if (len > 32) {
    for (i = 0; i < 32; i++) {
      if (myvector[i + prefix].cast<bool>()) {
        retval |= 1 << i;
      }
    }
    prefix += 32;
    len = len - 32;
    // cout << "if extra " << retval << endl;
  } else { // len is equal to or smaller than 32, return directly
    for (i = 0; i < len; i++) {
      if (myvector[i].cast<bool>()) {
        retval |= 1 << i;
      }
    }
    return retval;
  }
  while (len > 32) {
    other = 0;
    for (i = 0; i < 32; i++) {
      if (myvector[i + prefix].cast<bool>()) {
        other |= 1 << i;
      }
    }
    prefix += 32;
    len = len - 32;
    extraF[extraFInitialIndex] = other;
    extraFInitialIndex++;
    // cout << "while extra " << other << endl;
  }
  other = 0;
  for (i = 0; i < len; i++) {
    if (myvector[i + prefix].cast<bool>()) {
      other |= 1 << i;
    }
  }
  // cout << "final extra " << other << endl;
  extraF[extraFInitialIndex] = other;
  extraFInitialIndex++;
  // cout<<"extraFInitialIndex="<<extraFInitialIndex<<endl;
  return retval; // return the first retval
}

// alpha is stored in alphabeta[0]; and beta is stored in alphabeta[1]
void calAlphaBeta(long transitionsLast[][2], float *alphabeta) {
  // delete the first n transitions (n state) in each chain.

  if (transitionsLast[0][0] + transitionsLast[0][1] == 0)
    alphabeta[1] = 0;
  else
    alphabeta[1] = (float)transitionsLast[0][1] /
                   (float)(transitionsLast[0][0] + transitionsLast[0][1]);
  if (transitionsLast[1][0] + transitionsLast[1][1] == 0)
    alphabeta[0] = 0;
  else
    alphabeta[0] = (float)transitionsLast[1][0] /
                   (float)(transitionsLast[1][0] + transitionsLast[1][1]);
  // printf("Calculated alpha=%f, beta=%f\n",alphabeta[0],alphabeta[1]);
}
/**
 * sizeSharedMemory1 is the amount of shared memory used to store the PBN and
 * the property, not including the initial states blockInfor[0] will store
 * blockSize; blockInfor[1] will store # Blocks
 */
void computeDeviceInfor(int sizeSharedMemory1, int stateSize, int *blockInfor) {
  // get device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // always use the first device
  int numMP = prop.multiProcessorCount;
  // int maxThreadPerBlock = 1024;
  int maxSharedMemoryPerBlock = prop.sharedMemPerBlock; // in bytes
  // int maxRegistersPerBlock; //in unit. 1 unit=32 bits
  int numRegisterPerMP =
      64 * 1024; // in unit. 1 unit=32 bits, specially for K20m
  int major = prop.major;
  int minor = prop.minor;
  int wrapSize = 32;
  // int countSize = 0;
  int numBlock, blockSize; // numRegistersPerBlock;
  // int allowedThreadPerBlock;
  int maxBlocksPerMP =
      16; // Maximum number of resident blocks per multiprocessor
  bool possible = true;
  int occupancy = 0;
  int selectBlockSize = 32, selectNumBlock = 1;
  int activeBLocksPerMp;
  if (major < 3 || (major < 4 && minor < 2)) {
    numRegisterPerMP = 32 * 1024;
    printf(
        "Caution! The device computational capability is too small. The "
        "program requires computational capability to be bigger than 3.0.\n");
  }
  if (major < 3) {
    maxBlocksPerMP = 8;
  } else if (major >= 5) {
    maxBlocksPerMP = 32;
  }
  /*if (major == 5 && minor == 3) {
   maxRegistersPerBlock = 32 * 1024;
   } else {
   maxRegistersPerBlock = 64 * 1024;
   }*/
  if (sizeSharedMemory1 > maxSharedMemoryPerBlock) {
    printf("The PBN is too large to handle with the current device.\n");
  }
  int countBlockSize = 1, countBlocks = 1;
  blockSize = countBlockSize * wrapSize;
  numBlock = countBlocks * numMP;
  // countSize = sizeSharedMemory1
  //	+ blockSize * numBlock * stateSize * sizeof(int);
  int i = 1, tmp = 0;
  while (possible) {
    possible = false;
    countBlockSize = 1;
    activeBLocksPerMp = countBlocks;
    if (activeBLocksPerMp > maxBlocksPerMP)
      activeBLocksPerMp = maxBlocksPerMP;
    blockSize = countBlockSize * wrapSize;
    // printf("(numRegisterPerMP / countBlocks) /
    // RegPerThread=%f\n",(numRegisterPerMP / countBlocks) / RegPerThread);
    while (blockSize < (numRegisterPerMP / countBlocks) / RegPerThread &&
           sizeSharedMemory1 < maxSharedMemoryPerBlock / countBlocks &&
           blockSize * countBlocks * numMP <= 7680) {
      // printf("maxSharedMemoryPerBlock=%d,countBlocks=%d,
      // sizeSharedMemory=%d\n",maxSharedMemoryPerBlock,countBlocks,sizeSharedMemory1);
      // shared memory limit
      if (maxSharedMemoryPerBlock / countBlocks < activeBLocksPerMp)
        activeBLocksPerMp = maxSharedMemoryPerBlock / countBlocks;
      // register limit
      if ((numRegisterPerMP / RegPerThread / blockSize) < activeBLocksPerMp) {
        activeBLocksPerMp = (numRegisterPerMP / RegPerThread / blockSize);
      }
      tmp = activeBLocksPerMp * blockSize;
      if (tmp > occupancy || (tmp == occupancy && numBlock > selectNumBlock)) {
        occupancy = tmp;
        selectBlockSize = blockSize;
        selectNumBlock = numBlock;
      }
      possible = true;
      // printf(
      //		"Possible solution %d: blockSize = %d, numBlock=%d,
      // occupancy=%d.\n", 		i, blockSize, numBlock, tmp);
      countBlockSize++;
      blockSize = countBlockSize * wrapSize;
      i++;
    }
    countBlocks++;
    numBlock = countBlocks * numMP;
    // numRegistersPerBlock = numRegisterPerMP / countBlocks;
  }

  // printf("Choose solution: blockSize = %d, numBlock=%d.\n", selectBlockSize,
  //		selectNumBlock);
  blockInfor[0] = selectBlockSize;
  blockInfor[1] = selectNumBlock;
  // printf("block=%d,blockSize=%d\n", numBlock, blockSize);
}

void initialisePBN_GPU(py::object PBN) {

  // n
  n = PBN.attr("getN")().cast<int>();

  stateSize = n / 32;
  if (stateSize * 32 < n)
    stateSize++;

  // nf
  py::list nf_py = PBN.attr("getNf")();

  nf = (uint16_t *)malloc(sizeof(uint16_t) * n);

  int idx = 0;
  for (auto elem : nf_py) {

    nf[idx++] = elem.cast<uint16_t>();
    cout << nf[idx - 1] << " - nf\n";
  }
  cout << "n - " << n << "\n";

  // nv
  py::list nv_py = PBN.attr("getNv")();

  int nv_len = py::len(nv_py);
  nv = (uint16_t *)malloc(sizeof(uint16_t) * nv_len);

  int cumNv = 0;
  extraFCount = 0;
  extraFIndexCount = 0;

  idx = 0;
  for (auto elem : nv_py) {
    uint16_t value = elem.cast<uint16_t>();

    if (value > 5) {
      extraFIndexCount++;
      extraFCount += (int)pow(2, value - 5) - 1;
    }
    cumNv += value;

    nv[idx++] = value;
  }

  // extraF
  if (extraFCount > 0) {
    extraFInitialIndex = 0;
    extraFIndex =
        (unsigned short *)malloc(sizeof(unsigned short) * extraFIndexCount);
    cumExtraF = (unsigned short *)malloc(sizeof(unsigned short) *
                                         (extraFIndexCount + 1));

    cumExtraF[0] = 0;
    extraF = (int *)malloc(sizeof(int) * extraFCount);
    int tmpIndex = 0;
    for (int i = 0; i < nv_len; i++) {
      if (nv[i] > 5) {
        extraFIndex[tmpIndex] = (unsigned short)i;
        tmpIndex++;
        cumExtraF[tmpIndex] =
            cumExtraF[tmpIndex - 1] + (unsigned short)pow(2, nv[i] - 5) - 1;
      }
    }
  } else { // create dummy ones
    extraFCount = 1;
    extraFIndexCount = 1;
    extraFIndex =
        (unsigned short *)malloc(sizeof(unsigned short) * extraFIndexCount);
    extraF = (int *)malloc(sizeof(int) * extraFCount);
    cumExtraF = (unsigned short *)malloc(sizeof(unsigned short) *
                                         (extraFIndexCount + 1));
  }

  // F
  py::list F_py = PBN.attr("getF")();

  int sizeF = py::len(F_py);

  F = (int *)malloc(sizeof(int) * sizeF);

  idx = 0;
  for (auto elem : F_py) {
    py::list elem_list = py::cast<py::list>(elem);

    int elem_len = py::len(elem);

    F[idx++] = fromVector(elem_list, elem_len);
  }

  // varF
  py::list varF_py = PBN.attr("getVarFInt")();

  varF = (uint16_t *)malloc(sizeof(uint16_t) * cumNv);
  // cout<<"num varF="<<cumNv<<endl;
  // cout<<"varF"<<endl;
  idx = 0;
  for (auto varF_elem : varF_py) {
    py::list varF_elem_list = py::cast<py::list>(varF_elem);
    for (auto elem : varF_elem_list) {
      varF[idx++] = elem.cast<uint16_t>();
    }
  }

  // cij
  py::list cij_py = PBN.attr("getCij")();
  cij = (float *)malloc(sizeof(float) * sizeF);

  idx = 0;
  for (auto cij_sublist : cij_py) {
    for (auto elem : cij_sublist) {
      cij[idx++] = elem.cast<float>();
    }
  }

  // p
  p = PBN.attr("getPerturbation")().cast<float>();

  // npNode
  py::list npNode_py = PBN.attr("getNpNode")();
  int npNode_size = py::len(npNode_py);

  g_npNode = (int *)malloc(sizeof(int) * npNode_size);
  idx = 0;
  for (auto elem : npNode_py) {
    g_npNode[idx++] = elem.cast<int>();
  }
  g_npLength = npNode_size;
}

double *german_gpu_run() {
  int block = 2, blockSize = 3;
  int steps = 5; // german and rubin n
  int *gpu_steps;
  float r = precision;
  int argCount = 1;
  bool useTexture = false;

  // ofstream output;

  // pbn = io.loadPBN(argv[argCount]);
  argCount += 2;
  // argv[2]=property file name
  // output.open(outputName, ios::out | ios::app);

  steps = 100; // stoi(argv[argCount]);

  int N = block * blockSize;
  cout << "***************************\n";
  cout << "running two-state on model with N = " << N;
  // output << argv[1];

  std::clock_t cpu_start;
  double duration;
  cpu_start = std::clock();

  int size_sharedMemory = 0;

  int *gpu_n;

  // calculate cumulative number of functions
  unsigned short *cumNf =
      (unsigned short *)malloc((n + 1) * sizeof(unsigned short));
  unsigned short *gpu_cumNf;
  cumNf[0] = 0;
  for (int i = 0; i < n; i++) {
    cumNf[i + 1] = cumNf[i] + nf[i];
    cout << "cumNf - " << cumNf[i + 1] << "\n";
  }

  // calculate cumulative number of variables
  unsigned short *cumNv = (unsigned short *)malloc(
      (cumNf[n] + 1) * sizeof(unsigned short)); //[cumNf[n] + 1];
  unsigned short *gpu_cumNv;
  cumNv[0] = 0;
  for (int i = 0; i < cumNf[n]; i++) {
    cumNv[i + 1] = cumNv[i] + nv[i];
  }

  int *gpu_F;
  unsigned short *gpu_varF;
  int count = 0;

  float *gpu_p;

  float *gpu_cij;
  count = 0;
  float sum;
  for (int i = 0; i < n; i++) {
    sum = 0;
    for (int j = 0; j < nf[i]; j++) {
      sum += cij[count];
      cij[count] = sum + epsilon;
      count++;
    }
    if (cij[count - 1] < 1) {
      cij[count - 1] = 1 + epsilon;
    }
  }

  cout << "allocating finished\n";
  free(nf);
  free(nv);

  // cout<<"size_sharedMemory="<<size_sharedMemory<<endl;
  size_sharedMemory += (cumNf[n] + 1) * sizeof(unsigned short); // cumNv

  if ((cumNf[n] + 1) % 2 != 0)
    size_sharedMemory += sizeof(unsigned short); // padding for cumNv

  size_sharedMemory += cumNf[n] * sizeof(int);                   // F
  size_sharedMemory += cumNv[cumNf[n]] * sizeof(unsigned short); // varF
  if (cumNv[cumNf[n]] != 0)
    size_sharedMemory += sizeof(unsigned short); // padding for varF

  if (extraFCount != 0) {
    size_sharedMemory += extraFIndexCount * sizeof(int);       // extraFIndex
    size_sharedMemory += extraFCount * sizeof(int);            // extraF
    size_sharedMemory += (1 + extraFIndexCount) * sizeof(int); // cumExtraF
    // size_sharedMemory += 2 * sizeof(int); //extraFIndexCount and extraFCount
    // those two are not used in shared memory
  }
  size_sharedMemory += sizeof(int) * (g_npLength + 1); // npNode and its length

  // determine whether to put all information in shared memory
  if (size_sharedMemory > maxAllowedSharedMemory) {
    int tmp = size_sharedMemory;
    // size_sharedMemory=size_sharedMemory-(cumNf[n] + 1) * sizeof(int); //cumNv
    // size_sharedMemory=size_sharedMemory-cumNf[n] * sizeof(int); //F
    if (extraFCount != 0) {
      size_sharedMemory -= extraFIndexCount * sizeof(int);       // extraFIndex
      size_sharedMemory -= extraFCount * sizeof(int);            // extraF
      size_sharedMemory -= (1 + extraFIndexCount) * sizeof(int); // cumExtraF
      // size_sharedMemory -= 2 * sizeof(int); //extraFIndexCount and
      // extraFCount
    }
    // size_sharedMemory -= cumNv[cumNf[n]] * sizeof(int); //varF
    useTexture = true;
    cout << "Using texture memory to store extraF, extraFindex, cumExtraF "
            "size texture="
         << (tmp - size_sharedMemory) << "bytes. " << endl;
  }

  if (blockInfor[0] != 0) {
    blockSize = blockInfor[1];
    block = blockInfor[0];
  } else {
    computeDeviceInfor(size_sharedMemory, stateSize, blockInfor);
    block = blockInfor[1];
    blockSize = blockInfor[0];
  }

  block /= 3;
  blockSize /= 3;

  N = block * blockSize; // Alokujemy wiecej blokow
  // size_sharedMemory += stateSize * N * sizeof(int);
  cout << "blockSize=" << blockSize << ", block=" << block
       << ", precision=" << r << ", sharedMemorySize=" << size_sharedMemory
       << " bytes.\n";

  float memsettime;
  cudaEvent_t start, stop;

  // initialize CUDA timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // for German and Runbin method
  int currentTrajectorySize = 0; // store the current trajectory size
  int *gpu_currentTrajectorySize;
  int trajectoryLength = 1000;
  float *mean = (float *)malloc(2 * N * sizeof(float)); //[2 * N];
  float *gpu_mean;

  float *gpu_trajectory;

  int *initialState; //[N * stateSize];
  initialState = (int *)malloc(N * stateSize * sizeof(int));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < stateSize; j++) {
      initialState[i * stateSize + j] = 25;
    }
  }
  // cout<<"after allocating initial state\n";
  int *gpu_initialState;
  // int* gpu_initialStateCopy;
  int *gpu_stateSize;
  int *gpu_trajectoryKernel;
  int *gpu_positiveIndex;
  int *gpu_negativeIndex;
  int *gpu_extraF;
  int *gpu_extraFIndex;
  int *gpu_cumExtraF;
  int *gpu_extraFCount;
  int *gpu_extraFIndexCount;

  int *gpu_npNode;
  int *gpu_npLength;

  HANDLE_ERROR(cudaMalloc((void **)&gpu_extraF, sizeof(int) * extraFCount));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_extraFIndex,
                          sizeof(unsigned short) * extraFIndexCount));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_cumExtraF,
                          sizeof(unsigned short) * (extraFIndexCount + 1)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_extraFCount, sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_extraFIndexCount, sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_npNode, sizeof(int) * g_npLength));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_npLength, sizeof(int)));

  if (extraFCount != 0) {
    HANDLE_ERROR(cudaMemcpy(gpu_extraF, extraF, extraFCount * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_extraFIndex, extraFIndex,
                            extraFIndexCount * sizeof(unsigned short),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_cumExtraF, cumExtraF,
                            (extraFIndexCount + 1) * sizeof(unsigned short),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_extraFCount, &extraFCount, sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_extraFIndexCount, &extraFIndexCount,
                            sizeof(int), cudaMemcpyHostToDevice));
  }
  HANDLE_ERROR(cudaMemcpy(gpu_npNode, g_npNode, g_npLength * sizeof(int),
                          cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMemcpy(gpu_npLength, &g_npLength, sizeof(int),
                          cudaMemcpyHostToDevice));

  // allocate method in device
  HANDLE_ERROR(cudaMalloc((void **)&gpu_n, sizeof(int)));
  // HANDLE_ERROR(cudaMalloc((void** ) &gpu_nf, n * sizeof(unsigned short)));
  // HANDLE_ERROR(cudaMalloc((void** ) &gpu_nv, cumNf[n] * sizeof(unsigned
  // short)));
  HANDLE_ERROR(
      cudaMalloc((void **)&gpu_cumNf, (n + 1) * sizeof(unsigned short)));
  HANDLE_ERROR(
      cudaMalloc((void **)&gpu_cumNv, (cumNf[n] + 1) * sizeof(unsigned short)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_F, (cumNf[n]) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_varF,
                          (cumNv[cumNf[n]]) * sizeof(unsigned short)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_cij, (cumNf[n]) * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_p, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_steps, sizeof(int)));
  // HANDLE_ERROR(cudaMalloc((void**) &gpu_countGPU, n * N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_currentTrajectorySize, sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_mean, 2 * N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_trajectory,
                          N * trajectoryLength * sizeof(float)));
  HANDLE_ERROR(
      cudaMalloc((void **)&gpu_initialState, stateSize * N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&gpu_stateSize, sizeof(int)));

  // copy data from host to device
  HANDLE_ERROR(cudaMemcpy(gpu_n, &n, sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpu_cumNf, cumNf, (n + 1) * sizeof(unsigned short),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpu_cumNv, cumNv,
                          (cumNf[n] + 1) * sizeof(unsigned short),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(gpu_F, F, cumNf[n] * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpu_varF, varF,
                          (cumNv[cumNf[n]]) * sizeof(unsigned short),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpu_cij, cij, cumNf[n] * sizeof(float),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpu_p, &p, sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(gpu_steps, &steps, sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpu_currentTrajectorySize, &currentTrajectorySize,
                          sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpu_stateSize, &stateSize, sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(gpu_initialState, initialState,
                 N * stateSize * sizeof(int), // MC: dlaczego tu byl float? xd
                 cudaMemcpyHostToDevice));

  // host constant data
  int hPowNum[2][32];
  hPowNum[1][0] = 1;
  hPowNum[0][0] = 0;
  for (int i = 1; i < 32; i++) {
    hPowNum[0][i] = 0;
    hPowNum[1][i] = hPowNum[1][i - 1] * 2;
  }
  // copy host data to constant memory
  cudaMemcpyToSymbol(powNum, hPowNum, sizeof(int) * 32 * 2);
  cudaMemcpyToSymbol(nodeNum, &n, sizeof(int));
  cudaMemcpyToSymbol(constantP, &p, sizeof(float));
  cudaMemcpyToSymbol(constantCumNf, cumNf, sizeof(unsigned short) * (n + 1));
  cudaMemcpyToSymbol(constantCij, cij, sizeof(float) * cumNf[n]);

  // free variables in host
  free(cij);
  free(F);
  free(varF);
  free(extraF);
  free(cumExtraF);
  free(extraFIndex);
  free(initialState);
  free(cumNf);
  free(cumNv);
  // CUDA's random number library uses curandState_t to keep track of the seed
  // value we will store a random state for every thread
  curandState_t *states;

  // allocate space on the GPU for the random states
  HANDLE_ERROR(cudaMalloc((void **)&states, N * sizeof(curandState_t)));

  // invoke the GPU to initialize all of the random states
  init<<<block, blockSize>>>(time(0), states);

  // bind texture reference with linear memory
  // cudaBindTexture(0,texcumNv,gpu_cumNv,sizeof(int)*(cumNf[n] + 1));
  // cudaBindTexture(0,texVarF,gpu_varF,sizeof(int)*cumNv[cumNf[n]]);
  cudaBindTexture(0, texExtraF, gpu_extraF, sizeof(int) * extraFCount);
  cudaBindTexture(0, texExtraFIndex, gpu_extraFIndex,
                  sizeof(unsigned short) * extraFIndexCount);
  cudaBindTexture(0, texCumExtraF, gpu_cumExtraF,
                  sizeof(unsigned short) * (extraFIndexCount + 1));

  German german;
  float psrf;
  bool done = false, done1 = false;
  float threshold = 1e-3; // judge when to converge
  currentTrajectorySize = 0;

  // german and rubin method
  cout << "before calling kernelconvergeInitial\n";

  if (n < 129) {
    // printf("1\n");
    cout << "Running converge1\n";
    kernelConvergeInitial1<4><<<block, blockSize, size_sharedMemory>>>(
        states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
        gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
        gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  } else if (n < 161) {
    // printf("5\n")
    kernelConvergeInitial1<5><<<block, blockSize, size_sharedMemory>>>(
        states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
        gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
        gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  } else if (n < 193) {
    // printf("6\n");
    kernelConvergeInitial1<6><<<block, blockSize, size_sharedMemory>>>(
        states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
        gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
        gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  } else if (n < 513) {
    // if (useTexture)
    //   kernelConvergeInitial7_texture<<<block, blockSize,
    //   size_sharedMemory>>>(
    //       states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
    //       gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
    //       gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
    // else
    kernelConvergeInitial1<16><<<block, blockSize, size_sharedMemory>>>(
        states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
        gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
        gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  } else if (n < 2049) {
    // printf("8\n");
    kernelConvergeInitial1<64><<<block, blockSize, size_sharedMemory>>>(
        states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
        gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
        gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  }

  // testKernel<<<1,10>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed %s\n", cudaGetErrorString(err));
  else
    printf("cuda success\n");

  cudaDeviceSynchronize();

  // currentTrajectorySize = steps;
  // printf("finish converge initial\n");
  // cout<<"before calling kernelConverge  currentTrajectorySize: " <<
  // currentTrajectorySize << "  printing "<< N <<" pbn states\n\n";

  // MC: wypisujemy wszystkie stany ktore otrzymalismy z 25tek za pomoca
  // convergeinitial:

  // HANDLE_ERROR(cudaMemcpy(perturbations, gpu_perturbations, sizeof(int),
  // cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaMemcpy(initialState, gpu_initialState,
                          N * stateSize * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    cout << "______INITIAL STATE OF PBN " << i << " ______\n";
    for (int j = 0; j < stateSize; j++) {
      for (size_t k = 0; k < sizeof(int); k++) {
        unsigned char *bytes = reinterpret_cast<unsigned char *>(
            &(initialState[i * stateSize + j]));
        bitset<8> bits = bytes[k];
        cout << bits << " ";
      }
    }
  }
  // cout<< "\nNumber of perturbations that occured: " << perturbations[0] <<
  // "\n"; cout << "Kernel converge:\n"; while (!done) {
  //   // call kernel function, need to allocate the shared method size here as
  //   the
  //   // third parameters
  //   if (n < 129) {

  //     kernelConverge1<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_mean,
  //         gpu_trajectory, gpu_trajectoryKernel, gpu_steps, gpu_stateSize,
  //         gpu_extraF, gpu_extraFIndex, gpu_cumExtraF, gpu_extraFCount,
  //         gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 161) {
  //     kernelConverge5<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_mean,
  //         gpu_trajectory, gpu_trajectoryKernel, gpu_steps, gpu_stateSize,
  //         gpu_extraF, gpu_extraFIndex, gpu_cumExtraF, gpu_extraFCount,
  //         gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 193) {
  //     kernelConverge6<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_mean,
  //         gpu_trajectory, gpu_trajectoryKernel, gpu_steps, gpu_stateSize,
  //         gpu_extraF, gpu_extraFIndex, gpu_cumExtraF, gpu_extraFCount,
  //         gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 513) {
  //     if (useTexture)
  //       kernelConverge7_texture<<<block, blockSize, size_sharedMemory>>>(
  //           states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_mean,
  //           gpu_trajectory, gpu_trajectoryKernel, gpu_steps, gpu_stateSize,
  //           gpu_extraF, gpu_extraFIndex, gpu_cumExtraF, gpu_extraFCount,
  //           gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //     else
  //       kernelConverge7<<<block, blockSize, size_sharedMemory>>>(
  //           states, gpu_cumNv, gpu_F, gpu_varF, gpu_p, gpu_initialState,
  //           gpu_mean, gpu_trajectory, gpu_trajectoryKernel, gpu_steps,
  //           gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //           gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 2049) {
  //     kernelConverge8<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_mean,
  //         gpu_trajectory, gpu_trajectoryKernel, gpu_steps, gpu_stateSize,
  //         gpu_extraF, gpu_extraFIndex, gpu_cumExtraF, gpu_extraFCount,
  //         gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   }
  //   currentTrajectorySize = currentTrajectorySize + steps;
  //   // printf("currentTrajectorySize=%d\n",currentTrajectorySize);
  //   HANDLE_ERROR(cudaMemcpy(mean, gpu_mean, 2 * N * sizeof(float),
  //                           cudaMemcpyDeviceToHost));

  //   psrf = german.computePstr(mean, currentTrajectorySize, n, N);

  //   //free(mean);

  //   if (abs(1 - psrf) > threshold) {
  //     done = false;
  //     done1 = false;
  //     steps = currentTrajectorySize;
  //   } else {
  //     if (done1) {
  //       done = true;
  //     } else {
  //       done1 = true;
  //       steps = currentTrajectorySize;
  //     }
  //   }
  //   if (steps > trajectoryLength) {
  //     done = true;
  //     steps = currentTrajectorySize / 2;
  //   }
  //   HANDLE_ERROR(
  //       cudaMemcpy(gpu_steps, &steps, sizeof(int), cudaMemcpyHostToDevice));
  // }

  // cout << "\n\n ---Finished big kernel loop----\n";

  // // printf("converged! current trajectory size = %d\n",
  // currentTrajectorySize); cudaFree(gpu_mean); cudaFree(gpu_trajectory);
  // // printf("free memory\n");
  // //*****************************two state***************************

  // // cout<<"before allocating stateA\n";
  // float invPhi = ltqnorm(0.5 * (1 + confidence));
  // int kstep = 1;
  // long stateASum = 0, stateBSum = 0;
  // // printf("before allocating memory stateA \n");
  // long *stateA = (long *)malloc(
  //     N * sizeof(long)); //[N];			//how many steps are in
  //     state A
  // for (int i = 0; i < N; i++) {
  //   stateA[i] = 0;
  // }

  // // printf("before allocating memory stateB \n");
  // long *stateB = (long *)malloc(
  //     N *
  //     sizeof(long)); //[N];
  //                    // element 0=transitions from meta state B to meta state
  //                    B
  //                    // element 1=transitions from meta state B to meta state
  //                    A
  //                    // element 2=transitions from meta state A to meta state
  //                    B
  //                    // element 3=transitions from meta state A to meta state
  //                    A
  // long transitionsLast[2][2];
  // // printf("before allocating memory transitionsLastChain %d\n",N);
  // int *transitionsLastChain = (int *)malloc(N * 4 * sizeof(int)); //[N * 4];
  // // printf("before allocating memory gpu_stateA \n");
  // long *gpu_stateA;
  // long *gpu_stateB;
  // int *gpu_transitionsLastChain;
  // int *gpu_bridge;
  // // printf("before allocating memory gpu_transitionsLastChain \n");

  // // cout<<"before kernelUpdateTrajectory\n";
  // HANDLE_ERROR(
  //     cudaMalloc((void **)&gpu_transitionsLastChain, N * 4 * sizeof(int)));
  // HANDLE_ERROR(cudaMalloc((void **)&gpu_stateA, N * sizeof(long)));
  // HANDLE_ERROR(cudaMalloc((void **)&gpu_stateB, N * sizeof(long)));
  // HANDLE_ERROR(cudaMalloc((void **)&gpu_bridge, N * sizeof(int)));
  // // printf("before calling update trajectory kernel\n");
  // kernelUpdateTrajectory<<<block, blockSize, size_sharedMemory>>>(
  //     gpu_cumNv, gpu_trajectoryKernel, gpu_steps, gpu_stateA, gpu_stateB,
  //     gpu_transitionsLastChain, gpu_bridge, gpu_stateSize);
  // // cout<<"after calling update trajectory kernel"<<endl;
  // HANDLE_ERROR(
  //     cudaMemcpy(stateA, gpu_stateA, N * sizeof(long),
  //     cudaMemcpyDeviceToHost));
  // HANDLE_ERROR(
  //     cudaMemcpy(stateB, gpu_stateB, N * sizeof(long),
  //     cudaMemcpyDeviceToHost));
  // HANDLE_ERROR(cudaMemcpy(transitionsLastChain, gpu_transitionsLastChain,
  //                         4 * N * sizeof(int), cudaMemcpyDeviceToHost));

  // transitionsLast[0][0] = 0;
  // transitionsLast[0][1] = 0;
  // transitionsLast[1][0] = 0;
  // transitionsLast[1][1] = 0;
  // for (int i = 0; i < N; i++) {
  //   // printf("stateA[%d]=%d, stateb[%d]=%d, transitionsLastChain[%d-1]=%d,
  //   // transitionsLastChain[%d-2]=%d, transitionsLastChain[%d-3]=%d,
  //   //
  //   transitionsLastChain[%d-4]=%d\n",i,stateA[i],i,stateB[i],i,transitionsLastChain[i*4],i,transitionsLastChain[i*4+1],i,transitionsLastChain[i*4+2],i,transitionsLastChain[i*4+3]);
  //   stateASum += stateA[i];
  //   stateBSum += stateB[i];
  //   transitionsLast[0][0] += transitionsLastChain[i * 4];
  //   transitionsLast[0][1] += transitionsLastChain[i * 4 + 1];
  //   transitionsLast[1][0] += transitionsLastChain[i * 4 + 2];
  //   transitionsLast[1][1] += transitionsLastChain[i * 4 + 3];
  //   /*if(stateA[i]+stateB[i]!=transitionsLastChain[i *
  //   4]+transitionsLastChain[i
  //   * 4+1]+transitionsLastChain[i * 4+2]+transitionsLastChain[i * 4+3]){
  //           printf("stateA[%d]=%d, stateb[%d]=%d,
  //           transitionsLastChain[%d-1]=%d,
  //   transitionsLastChain[%d-2]=%d, transitionsLastChain[%d-3]=%d,
  //   transitionsLastChain[%d-4]=%d\n",i,stateA[i],i,stateB[i],i,transitionsLastChain[i*4],i,transitionsLastChain[i*4+1],i,transitionsLastChain[i*4+2],i,transitionsLastChain[i*4+3]);
  //   }*/
  // }
  // cout << "stateASum=" << stateASum << " stateBSum=" << stateBSum << endl;
  // cout << "transitionsLast[0][0]=" << transitionsLast[0][0]

  //        << ", transitionsLast[0][1]=" << transitionsLast[0][1]

  //        << ", transitionsLast[1][0]=" << transitionsLast[1][0]
  //        << ", transitionsLast[1][1]=" << transitionsLast[1][1] << endl;
  // cout << "Prob. is " << (float)stateASum / (float)(stateBSum + stateASum)
  //        << "\n";

  // /*cout << "stateASum=" << stateASum << " stateBSum=" << stateBSum << endl;
  // cout << "transitionsLast[0][0]=" << transitionsLast[0][0]

  // << ", transitionsLast[0][1]=" << transitionsLast[0][1]

  // << ", transitionsLast[1][0]=" << transitionsLast[1][0]
  // << ", transitionsLast[1][1]=" << transitionsLast[1][1] << endl;
  // cout << "Prob. is " << (float) stateASum / (float) (stateBSum + stateASum)
  // << "\n";*/

  // float alphabeta[2];
  // alphabeta[0] = 0;
  // alphabeta[1] = 0;
  // calAlphaBeta(transitionsLast, alphabeta);

  // cout << "alpha=" << alphabeta[0] << ", beta=" << alphabeta[1] << "\n";

  // long maxLength = transitionsLast[0][0] + transitionsLast[0][1] +
  //                  transitionsLast[1][0] + transitionsLast[1][1];
  // long TSN;
  // if (alphabeta[0] == 0 || alphabeta[1] == 0)
  //   TSN = (long)pow(2, 12);
  // else
  //   TSN = ceil(*alphabeta * *(alphabeta + 1) *
  //              (2 - *alphabeta - *(alphabeta + 1)) /
  //              (pow(*alphabeta + *(alphabeta + 1), 3) * pow(r / invPhi, 2)))
  //              *
  //         kstep;

  // int extensionPerChain;
  // int round = 0;
  // cout << "maxLength=" << maxLength << ", N=" << TSN << "\n";

  // // cout << "maxLength=" << maxLength << ", N=" << TSN << "\n";
  // while (TSN > maxLength) {
  //   extensionPerChain =
  //       (int)ceil((TSN - maxLength) * kstep / (double)N); // consider kstep
  //       here
  //   cout << "extensionPerChain=" << extensionPerChain << endl;
  //   // cout << "extensionPerChain=" << extensionPerChain << endl;
  //   // cout<<"before calling kernel\n";
  //   HANDLE_ERROR(cudaMemcpy(gpu_steps, &extensionPerChain, sizeof(int),
  //                           cudaMemcpyHostToDevice));
  //   if (n < 97 && n > 64) {
  //     kernel3<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
  //         gpu_stateA, gpu_stateB, gpu_transitionsLastChain, gpu_bridge,
  //         gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //         gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);

  //   } else if (n < 129) {
  //     kernel1<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
  //         gpu_stateA, gpu_stateB, gpu_transitionsLastChain, gpu_bridge,
  //         gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //         gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 161) {
  //     kernel5<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
  //         gpu_stateA, gpu_stateB, gpu_transitionsLastChain, gpu_bridge,
  //         gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //         gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 193) {
  //     kernel6<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
  //         gpu_stateA, gpu_stateB, gpu_transitionsLastChain, gpu_bridge,
  //         gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //         gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 513) {
  //     if (useTexture)
  //       kernel7_texture<<<block, blockSize, size_sharedMemory>>>(
  //           states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
  //           gpu_stateA, gpu_stateB, gpu_transitionsLastChain, gpu_bridge,
  //           gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //           gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //     else
  //       kernel7<<<block, blockSize, size_sharedMemory>>>(
  //           states, gpu_cumNv, gpu_F, gpu_varF, gpu_p, gpu_initialState,
  //           gpu_steps,

  //           gpu_stateA, gpu_stateB, gpu_transitionsLastChain, gpu_bridge,
  //           gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //           gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   } else if (n < 2049) {
  //     kernel8<<<block, blockSize, size_sharedMemory>>>(
  //         states, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps,
  //         gpu_stateA, gpu_stateB, gpu_transitionsLastChain, gpu_bridge,
  //         gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF,
  //         gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode);
  //   }
  //   HANDLE_ERROR(cudaMemcpy(stateA, gpu_stateA, N * sizeof(long),
  //                           cudaMemcpyDeviceToHost));
  //   HANDLE_ERROR(cudaMemcpy(stateB, gpu_stateB, N * sizeof(long),
  //                           cudaMemcpyDeviceToHost));
  //   HANDLE_ERROR(cudaMemcpy(transitionsLastChain, gpu_transitionsLastChain,
  //                           4 * N * sizeof(int), cudaMemcpyDeviceToHost));
  //   for (int i = 0; i < N; i++) {
  //     stateASum += stateA[i];
  //     stateBSum += stateB[i];
  //     transitionsLast[0][0] += transitionsLastChain[i * 4];
  //     transitionsLast[0][1] += transitionsLastChain[i * 4 + 1];
  //     transitionsLast[1][0] += transitionsLastChain[i * 4 + 2];
  //     transitionsLast[1][1] += transitionsLastChain[i * 4 + 3];
  //     // if(stateA[i]+stateB[i]!=transitionsLastChain[i *
  //     // 4]+transitionsLastChain[i * 4+1]+transitionsLastChain[i *
  //     // 4+2]+transitionsLastChain[i * 4+3]){ 	printf("stateA[%d]=%d,
  //     // stateb[%d]=%d, transitionsLastChain[%d-1]=%d,
  //     // transitionsLastChain[%d-2]=%d, transitionsLastChain[%d-3]=%d,
  //     //
  //     transitionsLastChain[%d-4]=%d\n",i,stateA[i],i,stateB[i],i,transitionsLastChain[i*4],i,transitionsLastChain[i*4+1],i,transitionsLastChain[i*4+2],i,transitionsLastChain[i*4+3]);
  //     // }
  //   }
  //   free(stateA);
  //   free(stateB);
  //   free(transitionsLastChain);
  //   cout << "stateASum=" << stateASum << " stateBSum=" << stateBSum << endl;
  //   cout << "transitionsLast[0][0]=" << transitionsLast[0][0]

  //          << ", transitionsLast[0][1]=" << transitionsLast[0][1]

  //          << ", transitionsLast[1][0]=" << transitionsLast[1][0]
  //          << ", transitionsLast[1][1]=" << transitionsLast[1][1] << endl;
  //   cout << "Prob. is " << (float)stateASum / (float)(stateBSum + stateASum)
  //          << "\n";
  //   calAlphaBeta(transitionsLast, alphabeta);
  //   cout << "alpha=" << alphabeta[0] << ", beta=" << alphabeta[1] << "\n";
  //   if (alphabeta[0] == 0 || alphabeta[1] == 0)
  //     TSN = TSN * 2;
  //   else
  //     TSN =
  //         ceil(alphabeta[0] * alphabeta[1] * (2 - alphabeta[0] -
  //         alphabeta[1]) /
  //              (pow(alphabeta[0] + alphabeta[1], 3) * pow(r / invPhi, 2))) *
  //         kstep;
  //   maxLength = transitionsLast[0][0] + transitionsLast[0][1] +
  //               transitionsLast[1][0] + transitionsLast[1][1];
  //   round++;
  //   cout << "maxLength=" << maxLength << ", N=" << TSN << "\n";
  // }

  // cout << "Finished in " << round << " round. Prob. is "
  //        << (float)stateASum / (float)(stateBSum + stateASum) << "\n";
  // duration = (std::clock() - cpu_start) / (double)CLOCKS_PER_SEC;

  // cout << "time duration : " << duration << "s\n";

  // stop CUDA timer

  cout << "BEFORE CUDA TIMER\n";
  cudaEventRecord(stop, 0);
  err = cudaEventQuery(stop);
  if (err != cudaSuccess)
    printf("Kernel launch failed %s\n", cudaGetErrorString(err));
  else
    printf("cuda success\n");
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&memsettime, start, stop);
  cout << " *** CUDA execution time: " << memsettime << " ms*** \n";
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // output.close();
  double *newArray = new double[4];

  cudaFree(gpu_n);
  // cudaFree(gpu_nf);
  // cudaFree(gpu_nv);
  cudaFree(gpu_cumNf);
  cudaFree(gpu_cumNv);
  cudaFree(gpu_F);
  cudaFree(gpu_varF);
  cudaFree(gpu_cij);
  cudaFree(gpu_p);
  cudaFree(gpu_steps);
  // cudaFree(gpu_positiveIndex);
  // cudaFree(gpu_negativeIndex);
  // cudaFree(gpu_stateA);
  // cudaFree(gpu_stateB);
  // cudaFree(gpu_transitionsLastChain);
  // cudaFree(gpu_bridge);
  cudaFree(gpu_currentTrajectorySize);
  cudaFree(gpu_initialState);
  cudaFree(gpu_stateSize);
  // cudaFree(gpu_trajectoryKernel);
  cudaFree(states);
  // cudaFree(gpu_transitionsLastChain);
  // cudaFree(gpu_stateA);
  // cudaFree(gpu_stateB);
  // cudaFree(gpu_bridge);
  cudaFree(gpu_extraF);
  cudaFree(gpu_extraFIndex);
  cudaFree(gpu_cumExtraF);
  cudaFree(gpu_extraFIndexCount);
  cudaFree(gpu_extraFCount);
  cudaFree(gpu_npLength);
  cudaFree(gpu_npNode);

  // write result array
  // jsize length = 4;
  // jdoubleArray newArray = env->NewDoubleArray(length);
  // jdouble *narr = env->GetDoubleArrayElements(newArray, NULL);
  // narr[0] = (float)stateASum / (float)(stateBSum + stateASum);
  // narr[1] = stateBSum + stateASum + steps * N; // counting burn-in as well
  // narr[2] = duration;
  // narr[3] = memsettime / 1000.0;
  // env->ReleaseDoubleArrayElements(newArray, narr, (jint)NULL);

  // newArray[0] = (float)stateASum / (float)(stateBSum + stateASum);
  // newArray[1] = stateBSum + stateASum + steps * N; // counting burn-in as
  // well newArray[2] = duration; newArray[3] = memsettime / 1000.0;

  return newArray;
}

namespace py = pybind11;

PYBIND11_MODULE(_gpu_stable, m) {
  m.def("german_gpu_run", &german_gpu_run, "Run German GPU method");
  m.def("initialise_PBN", &initialisePBN_GPU, "Initialise PBN on GPU",
        py::arg("PBN"));
}