/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ***************************************************************************/

/**
 * @file Solver_GPU_GlobalMemory_compressed.cu
 * @brief Child class of the Solver that runs the convergence
 * algorithm using Global Memory on a single GPU.
 *
 * @sa Solver
 * @sa Solver_GPU_GlobalMemory_compressed
 */

#include "Solver_GPU_GlobalMemory_compressed.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <map>
#include <array>  
#include <cstring>
#include <cstdint>

using namespace std::chrono;
using namespace std;
using std::ofstream;
using std::ifstream;
using std::istringstream;
using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;
using std::to_string;

#define BLOCKSIZE 1024

// every cell has 6 coefficients, one per wall: e, f, g, h, m, n
#define COEFFICIENTS_PER_CELL 6
// a uint16_t holds the numbers 0 ... 65535
#define MAX_CODES 65536
#define cudaCheck(x) _cudaCheck(x, #x, __FILE__, __LINE__)


template<typename T>
void Solver_GPU_GlobalMemory_compressed::_cudaCheck(T e, const char *func, const char *call, const int line)
{
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Divergence CUDA Kernel.
// The divergence kernel ...
//
__global__ void divergenceGlobal_compressed(float *d_u0, float *d_v0, float *d_w0, float *d_R, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, int alpha1, int nx, int ny, int nz, float dx, float dy, float *d_dz_array)
{

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int k = id / ((nx - 1) * (ny - 1));
  int j = (id - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = id - k * (nx - 1) * (ny - 1) - j * (nx - 1);
  int icell_face = i + j * nx + k * nx * ny;

  // Would be nice to figure out how to not have this branch check...
  if ((i < nx - 1) && (j < ny - 1) && (k < nz - 1) && (i >= 0) && (j >= 0) && (k > 0)) {

    // Divergence equation
    d_R[id] = (-2 * pow(alpha1, 2.0))
              * (((d_e[id] * d_u0[icell_face + 1] - d_f[id] * d_u0[icell_face]) * dx)
                 + ((d_g[id] * d_v0[icell_face + nx] - d_h[id] * d_v0[icell_face]) * dy)
                 + (d_m[id] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k + 1]) * d_w0[icell_face + nx * ny]
                    - d_n[id] * d_w0[icell_face] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k - 1])));
  }
}


// SOR red-black kernel
// Same as the fuse kernel, but instead of getting the coefficients of each cell from
// six big float arrays, each cell has a 2-byte code, and the six values for
// that code are read from a small table that stays in the GPU's cache
__global__ void SOR_RB_Global_compressed(float *d_lambda, int nx, int ny, int nz, float omega, float A, float B,
                                         const uint16_t *d_cell_codes, const float *d_coefficient_table,
                                         float *d_R, int offset, float *d_error)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int k = id / ((nx - 1) * (ny - 1));
  int j = (id - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = id - k * (nx - 1) * (ny - 1) - j * (nx - 1);
  float error = 0.0f; 

  if ((i > 0) && (i < nx - 2) && (j > 0) && (j < ny - 2) && (k < nz - 2) && (k > 0)
      && ((i + j + k) % 2) == offset) {
    // Look up this cell's six coefficients
    // - Get the cell's code
    // - Get the code's row of the table. 
    int code = d_cell_codes[id];
    const float *coefficients = &d_coefficient_table[code * COEFFICIENTS_PER_CELL];
    float e = coefficients[0];  // east wall   
    float f = coefficients[1];  // west wall   
    float g = coefficients[2];  // north wall  
    float h = coefficients[3];  // south wall  
    float m = coefficients[4];  // top wall    
    float n = coefficients[5];  // bottom wall 
    // Fused SOR update
    float new_lambda = (omega / (e + f + g + h + m + n))
                     * (e * d_lambda[id + 1]
                        + f * d_lambda[id - 1]
                        + g * d_lambda[id + (nx - 1)]
                        + h * d_lambda[id - (nx - 1)]
                        + m * d_lambda[id + (nx - 1) * (ny - 1)]
                        + n * d_lambda[id - (nx - 1) * (ny - 1)]
                        - d_R[id])
                   + (1.0 - omega) * d_lambda[id];
    error = fabs(new_lambda - d_lambda[id]); 
    d_lambda[id] = new_lambda;
  }
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  float warp_max = cg::reduce(warp, error, cg::greater<float>());
  if (warp.thread_rank() == 0 && warp_max > *d_error) {
    atomicMax((int *)d_error, __float_as_int(warp_max));
  }
}


// Fallback when the grid has more than MAX_CODES different coefficient sets (for large grids with cut cells)
__global__ void SOR_RB_Float_compressed(float *d_lambda, int nx, int ny, int nz, float omega, float A, float B,
                                        float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n,
                                        float *d_R, int offset, float *d_error)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int k = id / ((nx - 1) * (ny - 1));
  int j = (id - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = id - k * (nx - 1) * (ny - 1) - j * (nx - 1);
  float error = 0.0f;

  if ((i > 0) && (i < nx - 2) && (j > 0) && (j < ny - 2) && (k < nz - 2) && (k > 0)
      && ((i + j + k) % 2) == offset) {
    // Fused SOR update, coefficients straight from the float arrays
    float new_lambda = (omega / (d_e[id] + d_f[id] + d_g[id] + d_h[id] + d_m[id] + d_n[id]))
                     * (d_e[id] * d_lambda[id + 1]
                        + d_f[id] * d_lambda[id - 1]
                        + d_g[id] * d_lambda[id + (nx - 1)]
                        + d_h[id] * d_lambda[id - (nx - 1)]
                        + d_m[id] * d_lambda[id + (nx - 1) * (ny - 1)]
                        + d_n[id] * d_lambda[id - (nx - 1) * (ny - 1)]
                        - d_R[id])
                   + (1.0 - omega) * d_lambda[id];
    error = fabs(new_lambda - d_lambda[id]);
    d_lambda[id] = new_lambda;
  }
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  float warp_max = cg::reduce(warp, error, cg::greater<float>());
  if (warp.thread_rank() == 0 && warp_max > *d_error) {
    atomicMax((int *)d_error, __float_as_int(warp_max));
  }
}


__global__ void applyNeumannBCGlobal_compressed(float *d_lambda, int nx, int ny)
{
  // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < (nx - 1) * (ny - 1)) {
    d_lambda[id] = d_lambda[id + 1 * (nx - 1) * (ny - 1)];
  }
}

// Euler Final Velocity kernel
__global__ void finalVelocityGlobal_compressed(float *d_lambda, float *d_u, float *d_v, float *d_w, int *d_icellflag, float *d_f, float *d_h, float *d_n, int alpha1, int alpha2, float dx, float dy, float dz, float *d_dz_array, int nx, int ny, int nz)
{

  int icell_face = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_face / (nx * ny);
  int j = (icell_face - k * nx * ny) / nx;
  int i = icell_face - k * nx * ny - j * nx;
  int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);// Lineralized index for cell centered values


  if ((i > 0) && (i < nx - 1) && (j > 0) && (j < ny - 1) && (k < nz - 2) && (k > 0)) {

    d_u[icell_face] = d_u[icell_face]
                      + (1.0 / (2.0 * pow(alpha1, 2.0)))
                          * d_f[icell_cent]
                          * dx
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - 1]);

    d_v[icell_face] = d_v[icell_face]
                      + (1.0 / (2.0 * pow(alpha1, 2.0)))
                          * d_h[icell_cent]
                          * dy
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1)]);

    d_w[icell_face] = d_w[icell_face]
                      + (1.0 / (2.0 * pow(alpha2, 2.0)))
                          * d_n[icell_cent]
                          * d_dz_array[k]
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1) * (ny - 1)]);
  }
  if ((i >= 0) && (i < nx - 1)
      && (j >= 0) && (j < ny - 1)
      && (k < nz - 1) && (k >= 1)
      && ((d_icellflag[icell_cent] == 0) || (d_icellflag[icell_cent] == 2))) {
    d_u[icell_face] = 0;
    d_u[icell_face + 1] = 0;
    d_v[icell_face] = 0;
    d_v[icell_face + nx] = 0;
    d_w[icell_face] = 0;
    d_w[icell_face + nx * ny] = 0;
  }
}

Solver_GPU_GlobalMemory_compressed::Solver_GPU_GlobalMemory_compressed(qes::Domain domain_in, const float &tolerance)
  : Solver(std::move(domain_in), tolerance)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[Solver]\t Initializing Global Memory Solver (GPU, compressed) ..." << std::endl;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    std::cerr << "\n===================================================================" << std::endl;
    std::cerr << "[ERROR]\t cudaGetDeviceCount returned "
              << static_cast<int>(error_id) << "\n\t-> "
              << cudaGetErrorString(error_id) << std::endl;
    std::cerr << "===================================================================" << std::endl;
    exit(EXIT_FAILURE);
  }

  // This function call returnsq 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    std::cerr << "\n===================================================================" << std::endl;
    std::cerr << "[ERROR]\t There are no available device(s) that support CUDA\n";
    std::cerr << "===================================================================" << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaSetDevice(0);
}

// float to uint32 conversion that keeps bits the same
static uint32_t bitsOf(float value)
{
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Build the coefficient dictionary on the CPU
// Every different combination of six coefficients (e,f,g,h,m,n) gets a "code"
// Returns true if it fits, false if the grid has more than MAX_CODES different sets
static bool buildCoefficientDictionary(WINDSGeneralData *WGD, long numcell_cent,
                                       std::vector<uint16_t> &cell_codes,
                                       std::vector<float> &coefficient_table)
{
  // key: the bits of one cell's six coefficients, value: code for that set
  std::map<std::array<uint32_t, COEFFICIENTS_PER_CELL>, uint16_t> code_of_set;

  cell_codes.resize(numcell_cent);
  coefficient_table.clear();

  for (long id = 0; id < numcell_cent; id++) {
    // this cell's six coefficients, and their exact bits (the dictionary key)
    float coefficients[COEFFICIENTS_PER_CELL] = { WGD->e[id], WGD->f[id], WGD->g[id],
                                                  WGD->h[id], WGD->m[id], WGD->n[id] };
    std::array<uint32_t, COEFFICIENTS_PER_CELL> key;
    for (int wall = 0; wall < COEFFICIENTS_PER_CELL; wall++) {
      key[wall] = bitsOf(coefficients[wall]);
    }

    auto entry = code_of_set.find(key);
    // we have seen this combination before
    if (entry != code_of_set.end()) {
      cell_codes[id] = entry->second;
    } 
    // new combination, add to dict
    else {
      // if this grid doesn't fit in 2-byte codes
      if (code_of_set.size() == MAX_CODES) {
        // Throw away the dictionary and run the original kernel instead
        cell_codes.clear();
        cell_codes.shrink_to_fit();
        coefficient_table.clear();
        coefficient_table.shrink_to_fit();
        return false;
      }
      uint16_t new_code = code_of_set.size();
      code_of_set[key] = new_code;
      for (int wall = 0; wall < COEFFICIENTS_PER_CELL; wall++) {
        coefficient_table.push_back(coefficients[wall]);
      }
      cell_codes[id] = new_code;
    }
  }
  return true;// every cell got a code :)
}

void Solver_GPU_GlobalMemory_compressed::solve(WINDSGeneralData *WGD, const int &itermax)
{
  auto [nx, ny, nz] = domain.getDomainCellNum();
  auto [dx, dy, dz] = domain.getDomainSize();
  long numcell_face = domain.numFaceCentered();
  long numcell_cent = domain.numCellCentered();

  R.resize(numcell_cent, 0.0);

  std::cout << "[Solver] Running Global Memory Solver (GPU, compressed) ..." << std::endl;

  float *d_u, *d_v, *d_w;
  int *d_icellflag;
  float *d_dz_array;
  float *d_error;

  // Build the coefficient dictionary once on the CPU before running the wind solve
  if (!dictionary_built) {
    auto build_start = std::chrono::high_resolution_clock::now();
    dictionary_fits = buildCoefficientDictionary(WGD, numcell_cent, cell_codes, coefficient_table);
    dictionary_built = true;
    std::chrono::duration<float> build_time = std::chrono::high_resolution_clock::now() - build_start;
    if (dictionary_fits) {
      long number_of_codes = coefficient_table.size() / COEFFICIENTS_PER_CELL;
      std::cout << "[Solver] compressed: " << number_of_codes << " distinct coefficient sets, built in "
                << build_time.count() << " s" << std::endl;
    } else {
      std::cout << "[Solver] compressed: more than " << MAX_CODES << " distinct coefficient sets, "
                << "using the float coefficients instead (same answer, fuse speed)" << std::endl;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();// Start recording execution time

  // Copy the codes to the GPU if building the dictionary was successful
  uint16_t *d_cell_codes = nullptr;
  float *d_coefficient_table = nullptr;
  if (dictionary_fits) {
    long cell_codes_bytes = numcell_cent * sizeof(uint16_t);
    long table_bytes = coefficient_table.size() * sizeof(float);
    cudaMalloc((void **)&d_cell_codes, cell_codes_bytes);
    cudaMalloc((void **)&d_coefficient_table, table_bytes);
    cudaMemcpy(d_cell_codes, cell_codes.data(), cell_codes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coefficient_table, coefficient_table.data(), table_bytes, cudaMemcpyHostToDevice);
  }

  // d_e ... d_n are still copied below because the divergence and final velocity kernels still read them
  cudaMalloc((void **)&d_e, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_f, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_g, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_h, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_m, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_n, numcell_cent * sizeof(float));

  cudaMalloc((void **)&d_R, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_lambda, numcell_cent * sizeof(float));

  cudaMalloc((void **)&d_u, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_v, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_w, numcell_face * sizeof(float));

  cudaMalloc((void **)&d_dz_array, (nz - 1) * sizeof(float));
  cudaMalloc((void **)&d_icellflag, numcell_cent * sizeof(int));

  cudaMemcpy(d_u, WGD->u0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, WGD->v0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, WGD->w0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemset(d_R, 0.0, numcell_cent * sizeof(float));
  cudaMemcpy(d_lambda, lambda.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_e, WGD->e.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, WGD->f.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, WGD->g.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h, WGD->h.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, WGD->m.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, WGD->n.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dz_array, WGD->domain.dz_array.data(), (nz - 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_icellflag, WGD->icellflag.data(), numcell_cent * sizeof(int), cudaMemcpyHostToDevice);

  dim3 numberOfThreadsPerBlock(BLOCKSIZE, 1, 1);
  dim3 numberOfBlocks(ceil(domain.numCellCentered() / (float)(BLOCKSIZE)), 1, 1);

  // Invoke divergence kernel
  divergenceGlobal_compressed<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_u, d_v, d_w, d_R, d_e, d_f, d_g, d_h, d_m, d_n, alpha1, nx, ny, nz, dx, dy, d_dz_array);


  /////////////////////////////////////////////////
  //                 SOR solver              //////
  /////////////////////////////////////////////////

  int iter = 0;
  float max_error = 1.0;

  cudaMalloc((void **)&d_error, 1 * sizeof(float));

  // Main solver loop
  while ((iter < itermax) && (max_error > tol)) {
    // reset error on GPU
    cudaMemset(d_error, 0, sizeof(float));  
    
    if (dictionary_fits) {
      SOR_RB_Global_compressed<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_cell_codes, d_coefficient_table, d_R, 0, d_error);
      cudaCheck(cudaGetLastError());
      SOR_RB_Global_compressed<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_cell_codes, d_coefficient_table, d_R, 1, d_error);
      cudaCheck(cudaGetLastError());
    } else {
      SOR_RB_Float_compressed<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_e, d_f, d_g, d_h, d_m, d_n, d_R, 0, d_error);
      cudaCheck(cudaGetLastError());
      SOR_RB_Float_compressed<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_e, d_f, d_g, d_h, d_m, d_n, d_R, 1, d_error);
      cudaCheck(cudaGetLastError());
    }

    dim3 numberOfBlocks2(ceil(domain.numHorizontalCellCentered() / (float)(BLOCKSIZE)), 1, 1);
    // Invoke kernel to apply Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
    applyNeumannBCGlobal_compressed<<<numberOfBlocks2, numberOfThreadsPerBlock>>>(d_lambda, nx, ny);

    // get largest error onto cpu for the while loop condition
    cudaMemcpy(&max_error, d_error, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // incrementation of iteration counter
    iter += 1;
  }

  dim3 numberOfBlocks3(ceil(domain.numFaceCentered() / (float)(BLOCKSIZE)), 1, 1);
  // Invoke final velocity (Euler) kernel
  finalVelocityGlobal_compressed<<<numberOfBlocks3, numberOfThreadsPerBlock>>>(d_lambda, d_u, d_v, d_w, d_icellflag, d_f, d_h, d_n, alpha1, alpha2, dx, dy, dz, d_dz_array, nx, ny, nz);
  cudaCheck(cudaGetLastError());

  cudaMemcpy(WGD->u.data(), d_u, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->v.data(), d_v, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->w.data(), d_w, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_R);
  cudaFree(d_lambda);
  cudaFree(d_error);
  if (dictionary_fits) {
    cudaFree(d_cell_codes);
    cudaFree(d_coefficient_table);
  }

  cudaFree(d_e);
  cudaFree(d_f);
  cudaFree(d_g);
  cudaFree(d_h);
  cudaFree(d_m);
  cudaFree(d_n);

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);

  cudaFree(d_dz_array);
  cudaFree(d_icellflag);

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time

  std::chrono::duration<float> elapsed = finish - start;
  printf("[Solver]\t Residual after %d itertations: %2.9f\n", iter, max_error);
  std::cout << "\t\t elapsed time: " << elapsed.count() << " s\n";// Print out elapsed execution time
}
