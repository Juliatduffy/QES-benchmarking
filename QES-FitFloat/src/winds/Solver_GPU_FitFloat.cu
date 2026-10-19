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
 ****************************************************************************/

/**
 * @file Solver_GPU_FitFloat.cu
 * @brief FitFloat study: Solver_GPU_GlobalMemory with e..n and/or R stored as FitFloat
 * arrays. The kernel bodies are copied from Solver_GPU_GlobalMemory.cu unchanged; only the
 * array types became template parameters (C for e..n, RT for R).
 *
 * @sa Solver_GPU_FitFloat.h for the list of variants
 */

#include "Solver_GPU_FitFloat.h"
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "FitFloat.h"

#define BLOCKSIZE 1024
#define cudaCheck(x) _cudaCheck(x, #x, __FILE__, __LINE__)


template<typename T>
void Solver_GPU_FitFloat::_cudaCheck(T e, const char *func, const char *call, const int line)
{
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Divergence kernel (writes R)
template<class C, class RT>
__global__ void divergenceFF(float *d_u0, float *d_v0, float *d_w0, RT d_R, C d_e, C d_f, C d_g, C d_h, C d_m, C d_n, int alpha1, int nx, int ny, int nz, float dx, float dy, float *d_dz_array)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int k = id / ((nx - 1) * (ny - 1));
  int j = (id - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = id - k * (nx - 1) * (ny - 1) - j * (nx - 1);
  int icell_face = i + j * nx + k * nx * ny;

  if ((i < nx - 1) && (j < ny - 1) && (k < nz - 1) && (i >= 0) && (j >= 0) && (k > 0)) {
    // Divergence equation
    d_R[id] = (-2 * pow(alpha1, 2.0))
              * (((d_e[id] * d_u0[icell_face + 1] - d_f[id] * d_u0[icell_face]) * dx)
                 + ((d_g[id] * d_v0[icell_face + nx] - d_h[id] * d_v0[icell_face]) * dy)
                 + (d_m[id] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k + 1]) * d_w0[icell_face + nx * ny]
                    - d_n[id] * d_w0[icell_face] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k - 1])));
  }
}

// SOR red/black kernel (reads e..n and R)
template<class C, class RT>
__global__ void SOR_RB_FF(float *d_lambda, int nx, int ny, int nz, float omega, float A, float B, C d_e, C d_f, C d_g, C d_h, C d_m, C d_n, RT d_R, int offset)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int k = id / ((nx - 1) * (ny - 1));
  int j = (id - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = id - k * (nx - 1) * (ny - 1) - j * (nx - 1);

  if ((i > 0) && (i < nx - 2) && (j > 0) && (j < ny - 2) && (k < nz - 2) && (k > 0)
      && ((i + j + k) % 2) == offset) {
    // SOR formulation
    d_lambda[id] = (omega / (d_e[id] + d_f[id] + d_g[id] + d_h[id] + d_m[id] + d_n[id]))
                     * (d_e[id] * d_lambda[id + 1]
                        + d_f[id] * d_lambda[id - 1]
                        + d_g[id] * d_lambda[id + (nx - 1)]
                        + d_h[id] * d_lambda[id - (nx - 1)]
                        + d_m[id] * d_lambda[id + (nx - 1) * (ny - 1)]
                        + d_n[id] * d_lambda[id - (nx - 1) * (ny - 1)]
                        - d_R[id])
                   + (1.0 - omega) * d_lambda[id];
  }
}

__global__ void saveLambdaFF(float *d_lambda, float *d_lambda_old, int d_size)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < d_size) {
    d_lambda_old[id] = d_lambda[id];
  }
}

__global__ void applyNeumannBCFF(float *d_lambda, int nx, int ny)
{
  // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < nx * ny) {
    d_lambda[id] = d_lambda[id + 1 * (nx - 1) * (ny - 1)];
  }
}

__global__ void calculateErrorFF(float *d_lambda, float *d_lambda_old, float *d_value, int d_size)
{
  int ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < d_size) {
    float error = fabs(d_lambda[ii] - d_lambda_old[ii]);
    // atomicMAX using atomicCAS
    int r = __float_as_int(*d_value);
    while (error > __int_as_float(r)) {
      int o = r;
      if ((r = atomicCAS((int *)d_value, o, __float_as_int(error))) == o)
        break;
    }
  }
}

// Euler final velocity kernel (reads f, h, n)
template<class C>
__global__ void finalVelocityFF(float *d_lambda, float *d_u, float *d_v, float *d_w, int *d_icellflag, C d_f, C d_h, C d_n, int alpha1, int alpha2, float dx, float dy, float dz, float *d_dz_array, int nx, int ny, int nz)
{
  int icell_face = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_face / (nx * ny);
  int j = (icell_face - k * nx * ny) / nx;
  int i = icell_face - k * nx * ny - j * nx;
  int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);

  if ((i > 0) && (i < nx - 1) && (j > 0) && (j < ny - 1) && (k < nz - 2) && (k > 0)) {
    d_u[icell_face] = d_u[icell_face]
                      + (1.0 / (2.0 * pow(alpha1, 2.0))) * d_f[icell_cent] * dx
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - 1]);
    d_v[icell_face] = d_v[icell_face]
                      + (1.0 / (2.0 * pow(alpha1, 2.0))) * d_h[icell_cent] * dy
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1)]);
    d_w[icell_face] = d_w[icell_face]
                      + (1.0 / (2.0 * pow(alpha2, 2.0))) * d_n[icell_cent] * d_dz_array[k]
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1) * (ny - 1)]);
  }

  if ((i >= 0) && (i < nx - 1) && (j >= 0) && (j < ny - 1) && (k < nz - 1) && (k >= 1)
      && ((d_icellflag[icell_cent] == 0) || (d_icellflag[icell_cent] == 2))) {
    d_u[icell_face] = 0;
    d_u[icell_face + 1] = 0;
    d_v[icell_face] = 0;
    d_v[icell_face + nx] = 0;
    d_w[icell_face] = 0;
    d_w[icell_face + nx * ny] = 0;
  }
}

Solver_GPU_FitFloat::Solver_GPU_FitFloat(qes::Domain domain_in, const float &tolerance, const std::string &variant_in)
  : Solver(std::move(domain_in), tolerance), variant(variant_in)
{
  if (variant == "ff-ctl") {
    coef_bits = 32;
  } else if (variant == "ff-en16") {
    coef_bits = 16;
  } else if (variant == "ff-en4") {
    coef_bits = 4;
  } else if (variant == "ff-r16") {
    r16 = true;
  } else if (variant == "ff-en16r16") {
    coef_bits = 16;
    r16 = true;
  } else {
    std::cerr << "[ERROR]\t unknown FitFloat variant '" << variant << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[Solver]\t Initializing Global Memory Solver (GPU, FitFloat variant " << variant << ") ..." << std::endl;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess || deviceCount == 0) {
    std::cerr << "[ERROR]\t no CUDA device available" << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaSetDevice(0);
}

// Decodes every packed value on the GPU and counts the ones that differ (bitwise) from what the
// CPU packer says they decode to. Must be 0, otherwise the GPU sees different coefficients.
template<class C>
__global__ void checkDecodeFF(C packed, const float *expected, long n, unsigned long long *mismatches)
{
  long id = (long)blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n && __float_as_uint((float)packed[id]) != __float_as_uint(expected[id])) {
    atomicAdd(mismatches, 1ULL);
  }
}

// Packs one coefficient array into a device FitFloat array with FitFloat's host packer.
// Adds to `changed` (values that differ after packing), `out_of_range` (nonzero values that
// became 0 or inf), `gpu_mismatch` (GPU decode != CPU decode) and keeps the largest relative change in `max_rel`.
// Blocks of 32 values start on a 32-bit word boundary, so threads never write the same word.
template<unsigned int E, unsigned int M>
static FitFloatArray<float, E, M> *pack(const std::vector<float> &src, long &changed, long &out_of_range, double &max_rel, long &gpu_mismatch)
{
  const size_t n = src.size();
  HostFitFloatArray<float, E, M> h(n);
  const long nb = (long)((n + 31) / 32);
#pragma omp parallel for
  for (long b = 0; b < nb; b++) {
    for (size_t i = (size_t)b * 32; i < std::min(n, (size_t)b * 32 + 32); i++) {
      h.quick_write(i, src[i]);
    }
  }
  long ch = 0, oor = 0;
  double mr = 0.0;
#pragma omp parallel for reduction(+ : ch, oor) reduction(max : mr)
  for (long i = 0; i < (long)n; i++) {
    const float v = h[i];
    if (std::memcmp(&v, &src[i], sizeof(float)) != 0) {
      ch++;
      if (src[i] != 0.0f && (v == 0.0f || isinf(v))) {
        oor++;
      } else if (src[i] != 0.0f) {
        mr = std::max(mr, fabs(((double)v - src[i]) / src[i]));
      }
    }
  }
  changed += ch;
  out_of_range += oor;
  max_rel = std::max(max_rel, mr);

  // copy the packed words to the GPU (plain device memory, see FitFloatArray's constructor)
  FitFloatArray<float, E, M> *dst = new FitFloatArray<float, E, M>(n);
  cudaMemcpy(dst->data_pointer(), h.data_pointer(), sizeof(unsigned int) * h.actual_size(), cudaMemcpyHostToDevice);

  // decode everything on the GPU and compare with the CPU decode
  std::vector<float> expected(n);
#pragma omp parallel for
  for (long i = 0; i < (long)n; i++) {
    expected[i] = h[i];
  }
  float *d_expected;
  unsigned long long *d_mismatches, mismatches = 0;
  cudaMalloc((void **)&d_expected, n * sizeof(float));
  cudaMalloc((void **)&d_mismatches, sizeof(unsigned long long));
  cudaMemcpy(d_expected, expected.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_mismatches, 0, sizeof(unsigned long long));
  checkDecodeFF<<<(n + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(*dst, d_expected, (long)n, d_mismatches);
  cudaMemcpy(&mismatches, d_mismatches, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "[FitFloat]\t packing or GPU decode check failed: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaFree(d_expected);
  cudaFree(d_mismatches);
  gpu_mismatch += (long)mismatches;
  return dst;
}

template<class C, class RT>
void Solver_GPU_FitFloat::runSolver(WINDSGeneralData *WGD, const int &itermax, C ce, C cf, C cg, C ch, C cm, C cn, RT dR)
{
  auto [nx, ny, nz] = domain.getDomainCellNum();
  auto [dx, dy, dz] = domain.getDomainSize();
  long numcell_face = domain.numFaceCentered();
  long numcell_cent = domain.numCellCentered();

  R.resize(numcell_cent, 0.0);

  float *d_u, *d_v, *d_w;
  int *d_icellflag;
  float *d_dz_array;
  float *d_error;

  cudaMalloc((void **)&d_lambda, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_lambda_old, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_u, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_v, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_w, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_dz_array, (nz - 1) * sizeof(float));
  cudaMalloc((void **)&d_icellflag, numcell_cent * sizeof(int));

  cudaMemcpy(d_u, WGD->u0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, WGD->v0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, WGD->w0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lambda_old, lambda_old.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lambda, lambda.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dz_array, WGD->domain.dz_array.data(), (nz - 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_icellflag, WGD->icellflag.data(), numcell_cent * sizeof(int), cudaMemcpyHostToDevice);

  dim3 numberOfThreadsPerBlock(BLOCKSIZE, 1, 1);
  dim3 numberOfBlocks(ceil(domain.numCellCentered() / (float)(BLOCKSIZE)), 1, 1);

  divergenceFF<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_u, d_v, d_w, dR, ce, cf, cg, ch, cm, cn, alpha1, nx, ny, nz, dx, dy, d_dz_array);
  cudaCheck(cudaGetLastError());

  int iter = 0;
  float max_error = 1.0;
  cudaMalloc((void **)&d_error, 1 * sizeof(float));

  while ((iter < itermax) && (max_error > tol)) {
    saveLambdaFF<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, numcell_cent);
    cudaCheck(cudaGetLastError());

    // Red nodes pass
    SOR_RB_FF<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, ce, cf, cg, ch, cm, cn, dR, 0);
    cudaCheck(cudaGetLastError());

    // Black nodes pass
    SOR_RB_FF<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, ce, cf, cg, ch, cm, cn, dR, 1);
    cudaCheck(cudaGetLastError());

    dim3 numberOfBlocks2(ceil(domain.numHorizontalCellCentered() / (float)(BLOCKSIZE)), 1, 1);
    applyNeumannBCFF<<<numberOfBlocks2, numberOfThreadsPerBlock>>>(d_lambda, nx, ny);

    cudaMemset(d_error, 0, sizeof(float));
    calculateErrorFF<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, d_error, numcell_cent);
    cudaMemcpy(&max_error, d_error, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    iter += 1;
  }

  dim3 numberOfBlocks3(ceil(domain.numFaceCentered() / (float)(BLOCKSIZE)), 1, 1);
  finalVelocityFF<<<numberOfBlocks3, numberOfThreadsPerBlock>>>(d_lambda, d_u, d_v, d_w, d_icellflag, cf, ch, cn, alpha1, alpha2, dx, dy, dz, d_dz_array, nx, ny, nz);
  cudaCheck(cudaGetLastError());

  cudaMemcpy(WGD->u.data(), d_u, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->v.data(), d_v, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->w.data(), d_w, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_lambda);
  cudaFree(d_lambda_old);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_dz_array);
  cudaFree(d_icellflag);
  cudaFree(d_error);

  printf("[Solver]\t Residual after %d itertations: %2.9f\n", iter, max_error);
}

template<class RT>
void Solver_GPU_FitFloat::runWithCoefficients(WINDSGeneralData *WGD, const int &itermax, RT dR)
{
  if (coef_kind == 32) {
    runSolver(WGD, itermax, *ff32[0], *ff32[1], *ff32[2], *ff32[3], *ff32[4], *ff32[5], dR);
  } else if (coef_kind == 16) {
    runSolver(WGD, itermax, *ff16[0], *ff16[1], *ff16[2], *ff16[3], *ff16[4], *ff16[5], dR);
  } else if (coef_kind == 4) {
    runSolver(WGD, itermax, *ff4[0], *ff4[1], *ff4[2], *ff4[3], *ff4[4], *ff4[5], dR);
  } else {// plain float e..n, copied every solve like Solver_GPU_GlobalMemory
    long numcell_cent = domain.numCellCentered();
    float **d[6] = { &d_e, &d_f, &d_g, &d_h, &d_m, &d_n };
    const std::vector<float> *src[6] = { &WGD->e, &WGD->f, &WGD->g, &WGD->h, &WGD->m, &WGD->n };
    for (int a = 0; a < 6; a++) {
      cudaMalloc((void **)d[a], numcell_cent * sizeof(float));
      cudaMemcpy(*d[a], src[a]->data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
    }
    runSolver(WGD, itermax, d_e, d_f, d_g, d_h, d_m, d_n, dR);
    for (int a = 0; a < 6; a++) {
      cudaFree(*d[a]);
    }
  }
}

void Solver_GPU_FitFloat::solve(WINDSGeneralData *WGD, const int &itermax)
{
  auto start = std::chrono::high_resolution_clock::now();// includes the one-time packing

  // Pack e..n once: they are set in the WINDSGeneralData constructor and never change afterwards.
  if (coef_kind < 0) {
    coef_kind = 0;
    if (coef_bits > 0) {
      auto pack_start = std::chrono::high_resolution_clock::now();
      const std::vector<float> *src[6] = { &WGD->e, &WGD->f, &WGD->g, &WGD->h, &WGD->m, &WGD->n };
      long changed = 0, out_of_range = 0, gpu_mismatch = 0;
      double max_rel = 0.0;
      for (int a = 0; a < 6; a++) {
        if (coef_bits == 32) ff32[a] = pack<8, 23>(*src[a], changed, out_of_range, max_rel, gpu_mismatch);
        if (coef_bits == 16) ff16[a] = pack<8, 7>(*src[a], changed, out_of_range, max_rel, gpu_mismatch);
        if (coef_bits == 4) ff4[a] = pack<3, 0>(*src[a], changed, out_of_range, max_rel, gpu_mismatch);
      }
      std::chrono::duration<float> pack_elapsed = std::chrono::high_resolution_clock::now() - pack_start;
      std::cout << "[FitFloat]\t e..n in " << coef_bits << " bits: " << changed << " of " << 6 * src[0]->size()
                << " values changed, max relative change " << max_rel << ", " << out_of_range << " out of range" << std::endl;
      std::cout << "[FitFloat]\t GPU decode check: " << gpu_mismatch << " values decode differently on the GPU than on the CPU (must be 0)" << std::endl;
      if (out_of_range > 0) {
        std::cout << "[FitFloat]\t e..n out of range for " << coef_bits << " bits -> NOT packed, float e..n used instead" << std::endl;
        for (int a = 0; a < 6; a++) {
          delete ff32[a];
          delete ff16[a];
          delete ff4[a];
          ff32[a] = nullptr;
          ff16[a] = nullptr;
          ff4[a] = nullptr;
        }
      } else {
        coef_kind = coef_bits;
      }
      std::cout << "[FitFloat]\t packing time: " << pack_elapsed.count() << " s" << std::endl;
    }
    std::cout << "[FitFloat]\t using: e..n " << (coef_kind == 0 ? "float" : std::to_string(coef_kind) + "-bit FitFloat")
              << ", R " << (r16 ? "16-bit FitFloat" : "float") << std::endl;
  }

  std::cout << "[Solver] Running Global Memory Solver (GPU, FitFloat " << variant << ") ..." << std::endl;
  if (r16) {
    if (R16 == nullptr) {
      R16 = new FitFloatArray<float, 8, 7>(domain.numCellCentered());
    }
    runWithCoefficients(WGD, itermax, *R16);
  } else {
    cudaMalloc((void **)&d_R, domain.numCellCentered() * sizeof(float));
    cudaMemset(d_R, 0, domain.numCellCentered() * sizeof(float));
    runWithCoefficients(WGD, itermax, d_R);
    cudaFree(d_R);
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "\t\t elapsed time: " << elapsed.count() << " s\n";
}
