#include "wb.h"

#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                        \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void histogramKernel(unsigned int *deviceInput, int inputLength,
                                unsigned int *deviceBins, int numBins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < inputLength) {
    auto idx = deviceInput[i];
    if (idx >= numBins) return;
    if (deviceBins[idx] < 127) {
      auto previous = atomicAdd(&deviceBins[idx], 1);
      if (previous >= 127) {
        atomicMin(&deviceBins[idx], 127u);
      }
    }
  }
}

static void writeData(char *filename, unsigned int *data, int num) {
  FILE *handle = fopen(filename, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%d", *data++);
  }
  fflush(handle);
  fclose(handle);
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integers");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Выделите память GPU
  CUDA_CHECK(cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int)));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Скопируйте память с хоста на GPU
  CUDA_CHECK(cudaMemcpy(deviceInput, hostInput,
                        inputLength * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Запуск ядра
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Выполните вычисления в ядре
  const int threadsPerBlock = 256;
  const int blocksPerGrid =
      (inputLength + threadsPerBlock - 1) / threadsPerBlock;
  histogramKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, inputLength,
                                                      deviceBins, NUM_BINS);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceBins));
  wbTime_stop(GPU, "Freeing GPU Memory");

  // Тестовое сохранение результатов
  writeData("solution.raw", hostBins, NUM_BINS);

  // Проверка корректности результатов
  // -----------------------------------------------------
  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}