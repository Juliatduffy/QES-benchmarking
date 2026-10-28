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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/** @file Interpolation_FitFloat.cu
 * FitFloat study for QES-Plume (see Interpolation_FitFloat.h). This is the only file that
 * includes FitFloat.h. The interpolation code below is the code of Interpolation.cu with the
 * field-array type as a template parameter; the index functions (setInterp3Dindex_*) are the
 * ones from Interpolation.cu (relocatable device code).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

// FitFloat.h defines non-inline helper functions (get_ffarray_size_*, *H2FFDmemcpy). Solver_GPU_FitFloat.cu
// (qeswindsgpu) already defines them and qesPlumeGPU links both libraries, so this unit includes the header
// inside a namespace and gets its own copies. The system headers FitFloat.h includes are included above, at
// global scope, so their include guards keep them out of the namespace.
namespace plume_ff {
#include "winds/FitFloat.h"
}
using plume_ff::FitFloatArray;
using plume_ff::HostFitFloatArray;
#include "plume/CUDA/Interpolation_FitFloat.h"
#include "plume/CUDA/Interpolation.h"
#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"
#include "util/VectorMath.h"

#define FF_BLOCKSIZE 256

// ---------------------------------------------------------------------------------------------
// Packed device fields: same member names as QESWindsData / QESTurbData so the kernels read
// data.u, data.txx, ... exactly as Interpolation.cu does. tke is not interpolated and not packed.
// ---------------------------------------------------------------------------------------------
template<unsigned int E, unsigned int M>
struct WindsFF
{
  FitFloatArray<float, E, M> u, v, w;
  explicit WindsFF(size_t n) : u(n), v(n), w(n) {}
};

template<unsigned int E, unsigned int M>
struct TurbFF
{
  FitFloatArray<float, E, M> txx, txy, txz, tyy, tyz, tzz, div_tau_x, div_tau_y, div_tau_z, CoEps, nuT;
  explicit TurbFF(size_t n)
    : txx(n), txy(n), txz(n), tyy(n), tyz(n), tzz(n), div_tau_x(n), div_tau_y(n), div_tau_z(n), CoEps(n), nuT(n) {}
};

// ---------------------------------------------------------------------------------------------
// Trilinear interpolation (Interpolation.cu, with the array type as a template parameter).
// data[idx] is a float for a float array and a decoded float for a FitFloat array.
// ---------------------------------------------------------------------------------------------
template<class F>
__device__ __forceinline__ void interp3D_faceVar_ff(float &out, F &data, const interpWeight &wgt, const QESgrid qes_grid)
{
  float cube[2][2][2] = { 0.0 };
  for (int kkk = 0; kkk <= 1; ++kkk) {
    for (int jjj = 0; jjj <= 1; ++jjj) {
      for (int iii = 0; iii <= 1; ++iii) {
        int idx = (wgt.kk + kkk) * (qes_grid.ny * qes_grid.nx) + (wgt.jj + jjj) * (qes_grid.nx) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = data[idx];
      }
    }
  }
  float u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                + wgt.iw * wgt.jw * cube[1][1][0]
                + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  float u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                 + wgt.iw * wgt.jw * cube[1][1][1]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][1];
  out = (u_high - u_low) * wgt.kw + u_low;
}

template<class F>
__device__ __forceinline__ void interp3D_cellVar_ff(float &out, F &data, const interpWeight &wgt, const QESgrid qes_grid)
{
  float cube[2][2][2] = { 0.0 };
  for (int kkk = 0; kkk <= 1; ++kkk) {
    for (int jjj = 0; jjj <= 1; ++jjj) {
      for (int iii = 0; iii <= 1; ++iii) {
        int idx = (wgt.kk + kkk) * (qes_grid.ny - 1) * (qes_grid.nx - 1) + (wgt.jj + jjj) * (qes_grid.nx - 1) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = data[idx];
      }
    }
  }
  float u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                + wgt.iw * wgt.jw * cube[1][1][0]
                + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  float u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                 + wgt.iw * wgt.jw * cube[1][1][1]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][1];
  out = (u_high - u_low) * wgt.kw + u_low;
}

// winds at the particle positions (= interpolate(int, particle_array, QESWindsData, QESgrid))
template<class WD>
__global__ void interpolate_winds_ff(int length, particle_array d_particle_list, WD data, const QESgrid qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };
      float u, v, w;
      setInterp3Dindex_uFace(d_particle_list.pos[idx], wgt, qes_grid);
      interp3D_faceVar_ff(u, data.u, wgt, qes_grid);
      setInterp3Dindex_vFace(d_particle_list.pos[idx], wgt, qes_grid);
      interp3D_faceVar_ff(v, data.v, wgt, qes_grid);
      setInterp3Dindex_wFace(d_particle_list.pos[idx], wgt, qes_grid);
      interp3D_faceVar_ff(w, data.w, wgt, qes_grid);
      d_particle_list.velMean[idx] = { u, v, w };
    }
  }
}

// turbulence at the particle positions (= interpolate(int, particle_array, QESTurbData, QESgrid))
template<class TD>
__global__ void interpolate_turb_ff(int length, particle_array d_particle_list, TD data, const QESgrid qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };
      setInterp3Dindex_cellVar(d_particle_list.pos[idx], wgt, qes_grid);

      float CoEps_out;
      interp3D_cellVar_ff(CoEps_out, data.CoEps, wgt, qes_grid);
      if (CoEps_out <= 1e-6) {
        CoEps_out = 1e-6;
      }
      d_particle_list.CoEps[idx] = CoEps_out;

      float txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out;
      interp3D_cellVar_ff(txx_out, data.txx, wgt, qes_grid);
      interp3D_cellVar_ff(txy_out, data.txy, wgt, qes_grid);
      interp3D_cellVar_ff(txz_out, data.txz, wgt, qes_grid);
      interp3D_cellVar_ff(tyy_out, data.tyy, wgt, qes_grid);
      interp3D_cellVar_ff(tyz_out, data.tyz, wgt, qes_grid);
      interp3D_cellVar_ff(tzz_out, data.tzz, wgt, qes_grid);
      d_particle_list.tau[idx] = { txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out };

      float flux_div_x_out, flux_div_y_out, flux_div_z_out;
      interp3D_cellVar_ff(flux_div_x_out, data.div_tau_x, wgt, qes_grid);
      interp3D_cellVar_ff(flux_div_y_out, data.div_tau_y, wgt, qes_grid);
      interp3D_cellVar_ff(flux_div_z_out, data.div_tau_z, wgt, qes_grid);
      d_particle_list.flux_div[idx] = { flux_div_x_out, flux_div_y_out, flux_div_z_out };

      float nuT_out;
      interp3D_cellVar_ff(nuT_out, data.nuT, wgt, qes_grid);
      d_particle_list.nuT[idx] = nuT_out;
    }
  }
}

// tau and sigma for new particles (= interpolate(int, const vec3*, mat3sym*, vec3*, QESTurbData, QESgrid))
template<class TD>
__global__ void interpolate_new_ff(int length, const vec3 *pos, mat3sym *tau, vec3 *sigma, TD data, const QESgrid qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };
    setInterp3Dindex_cellVar(pos[idx], wgt, qes_grid);

    float txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out;
    interp3D_cellVar_ff(txx_out, data.txx, wgt, qes_grid);
    interp3D_cellVar_ff(txy_out, data.txy, wgt, qes_grid);
    interp3D_cellVar_ff(txz_out, data.txz, wgt, qes_grid);
    interp3D_cellVar_ff(tyy_out, data.tyy, wgt, qes_grid);
    interp3D_cellVar_ff(tyz_out, data.tyz, wgt, qes_grid);
    interp3D_cellVar_ff(tzz_out, data.tzz, wgt, qes_grid);
    tau[idx] = { txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out };

    float sig_x_out, sig_y_out, sig_z_out;
    sig_x_out = std::sqrt(std::abs(txx_out));
    if (sig_x_out == 0.0)
      sig_x_out = 1e-8;
    sig_y_out = std::sqrt(std::abs(tyy_out));
    if (sig_y_out == 0.0)
      sig_y_out = 1e-8;
    sig_z_out = std::sqrt(std::abs(tzz_out));
    if (sig_z_out == 0.0)
      sig_z_out = 1e-8;
    sigma[idx] = { sig_x_out, sig_y_out, sig_z_out };
  }
}

// ---------------------------------------------------------------------------------------------
// Packing (same procedure as Solver_GPU_FitFloat.cu): FitFloat's host packer on 32-element
// blocks (one 32-bit word boundary per block, so threads never share a word), statistics of
// what packing changed, the packed words copied to the device, and a GPU decode of every value
// compared with the CPU decode (must agree bit for bit).
// ---------------------------------------------------------------------------------------------
struct PackStats
{
  long n = 0, changed = 0, out_of_range = 0, gpu_mismatch = 0;
  double max_rel = 0.0;
};

template<class C>
__global__ void checkDecodeFF_plume(C packed, const float *expected, long n, unsigned long long *mismatches)
{
  long id = (long)blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n && __float_as_uint((float)packed[id]) != __float_as_uint(expected[id])) {
    atomicAdd(mismatches, 1ULL);
  }
}

template<unsigned int E, unsigned int M>
static void pack_field(const std::vector<float> &src, FitFloatArray<float, E, M> &dst, PackStats &st)
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
      if (src[i] != 0.0f && (v == 0.0f || std::isinf(v))) {
        oor++;
      } else if (src[i] != 0.0f) {
        mr = std::max(mr, std::fabs(((double)v - src[i]) / src[i]));
      }
    }
  }
  cudaMemcpy(dst.data_pointer(), h.data_pointer(), sizeof(unsigned int) * h.actual_size(), cudaMemcpyHostToDevice);

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
  checkDecodeFF_plume<<<(unsigned int)((n + FF_BLOCKSIZE - 1) / FF_BLOCKSIZE), FF_BLOCKSIZE>>>(dst, d_expected, (long)n, d_mismatches);
  cudaMemcpy(&mismatches, d_mismatches, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "[FitFloat]\t packing or GPU decode check failed: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaFree(d_expected);
  cudaFree(d_mismatches);

  st.n += (long)n;
  st.changed += ch;
  st.out_of_range += oor;
  st.max_rel = std::max(st.max_rel, mr);
  st.gpu_mismatch += (long)mismatches;
}

static void report(const char *what, int fields, unsigned int E, unsigned int M, const PackStats &st)
{
  printf("[FitFloat]\t %s: 1+%u+%u bits, %d fields x %ld values: %.1f%% changed by packing, %ld out of range (-> 0 or inf), max rel change %.2e\n",
         what, E, M, fields, st.n / std::max(fields, 1), 100.0 * st.changed / std::max(st.n, 1L), st.out_of_range, st.max_rel);
}

// ---------------------------------------------------------------------------------------------
// One instance of the packed fields per variant
// ---------------------------------------------------------------------------------------------
struct InterpolationFF::Fields
{
  virtual ~Fields() = default;
  virtual void get(particle_array d_particle, const QESgrid &qes_grid, int num_particle) = 0;
  virtual void getNew(particle_array d_new_particle, const QESgrid &qes_grid, int num_new_particle) = 0;
  virtual size_t packedBytes() const = 0;
};

template<unsigned int EW, unsigned int MW, unsigned int ET, unsigned int MT>
struct FieldsT : InterpolationFF::Fields
{
  WindsFF<EW, MW> *winds = nullptr;
  TurbFF<ET, MT> *turb = nullptr;

  FieldsT(const WINDSGeneralData *WGD, const TURBGeneralData *TGD)
  {
    const size_t nf = WGD->domain.numFaceCentered();
    const size_t nc = TGD->domain.numCellCentered();
    winds = new WindsFF<EW, MW>(nf);
    turb = new TurbFF<ET, MT>(nc);

    PackStats sw, st;
    pack_field(WGD->u, winds->u, sw);
    pack_field(WGD->v, winds->v, sw);
    pack_field(WGD->w, winds->w, sw);
    pack_field(TGD->txx, turb->txx, st);
    pack_field(TGD->txy, turb->txy, st);
    pack_field(TGD->txz, turb->txz, st);
    pack_field(TGD->tyy, turb->tyy, st);
    pack_field(TGD->tyz, turb->tyz, st);
    pack_field(TGD->tzz, turb->tzz, st);
    pack_field(TGD->div_tau_x, turb->div_tau_x, st);
    pack_field(TGD->div_tau_y, turb->div_tau_y, st);
    pack_field(TGD->div_tau_z, turb->div_tau_z, st);
    pack_field(TGD->CoEps, turb->CoEps, st);
    pack_field(TGD->nuT, turb->nuT, st);
    report("winds", 3, EW, MW, sw);
    report("turb", 11, ET, MT, st);
    printf("[FitFloat]\t GPU decode mismatches: %ld (must be 0)\n", sw.gpu_mismatch + st.gpu_mismatch);
    if (sw.gpu_mismatch + st.gpu_mismatch != 0) {
      std::cerr << "[FitFloat]\t the GPU decodes packed values differently than the CPU packer" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  ~FieldsT() override
  {
    delete winds;
    delete turb;
  }

  void get(particle_array d_particle, const QESgrid &qes_grid, int num_particle) override
  {
    int numBlocks = (num_particle + FF_BLOCKSIZE - 1) / FF_BLOCKSIZE;
    interpolate_winds_ff<<<numBlocks, FF_BLOCKSIZE>>>(num_particle, d_particle, *winds, qes_grid);
    interpolate_turb_ff<<<numBlocks, FF_BLOCKSIZE>>>(num_particle, d_particle, *turb, qes_grid);
    cudaDeviceSynchronize();
  }

  void getNew(particle_array d_new_particle, const QESgrid &qes_grid, int num_new_particle) override
  {
    int numBlocks = (num_new_particle + FF_BLOCKSIZE - 1) / FF_BLOCKSIZE;
    interpolate_new_ff<<<numBlocks, FF_BLOCKSIZE>>>(num_new_particle, d_new_particle.pos, d_new_particle.tau, d_new_particle.velFluct_old, *turb, qes_grid);
    cudaDeviceSynchronize();
  }

  size_t packedBytes() const override
  {
    return sizeof(unsigned int) * (3 * winds->u.actual_size() + 11 * turb->txx.actual_size());
  }
};

// ---------------------------------------------------------------------------------------------
// InterpolationFF
// ---------------------------------------------------------------------------------------------
static const char *const ff_variants[] = { "ff-ctl", "ff-wt24", "ff-wt20", "ff-wt16", "ff-wt16e5", "ff-wt12", "ff-w16", "ff-t16" };

bool InterpolationFF::isValid(const std::string &variant)
{
  for (const char *v : ff_variants) {
    if (variant == v) return true;
  }
  return false;
}

std::string InterpolationFF::variants()
{
  std::string s;
  for (const char *v : ff_variants) {
    s += (s.empty() ? "" : ", ") + std::string(v);
  }
  return s;
}

InterpolationFF::InterpolationFF(const std::string &variant, const WINDSGeneralData *WGD, const TURBGeneralData *TGD)
  : m_variant(variant)
{
  if (!isValid(variant)) {
    std::cerr << "[ERROR]\t unknown FitFloat variant '" << variant << "' (valid: " << variants() << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "[FitFloat]\t packing winds and turbulence fields, variant " << variant << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  if (variant == "ff-ctl") {
    m_fields = new FieldsT<8, 23, 8, 23>(WGD, TGD);
  } else if (variant == "ff-wt24") {
    m_fields = new FieldsT<8, 15, 8, 15>(WGD, TGD);
  } else if (variant == "ff-wt20") {
    m_fields = new FieldsT<8, 11, 8, 11>(WGD, TGD);
  } else if (variant == "ff-wt16") {
    m_fields = new FieldsT<8, 7, 8, 7>(WGD, TGD);
  } else if (variant == "ff-wt16e5") {
    m_fields = new FieldsT<5, 10, 5, 10>(WGD, TGD);
  } else if (variant == "ff-wt12") {
    m_fields = new FieldsT<8, 3, 8, 3>(WGD, TGD);
  } else if (variant == "ff-w16") {
    m_fields = new FieldsT<8, 7, 8, 23>(WGD, TGD);
  } else if (variant == "ff-t16") {
    m_fields = new FieldsT<8, 23, 8, 7>(WGD, TGD);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  m_packingTime = std::chrono::duration<double>(t1 - t0).count();
  m_nativeBytes = sizeof(float) * (3 * (size_t)WGD->domain.numFaceCentered() + 11 * (size_t)TGD->domain.numCellCentered());
  printf("[FitFloat]\t device bytes for the 14 fields: %.1f MB as float -> %.1f MB packed\n", m_nativeBytes / 1e6, packedBytes() / 1e6);
  printf("[FitFloat]\t packing time (CPU, incl. GPU decode check): %.3f s\n", m_packingTime);
}

InterpolationFF::~InterpolationFF()
{
  delete m_fields;
}

size_t InterpolationFF::packedBytes() const
{
  return m_fields ? m_fields->packedBytes() : 0;
}

void InterpolationFF::get(particle_array d_particle, const QESgrid &qes_grid, const int &num_particle)
{
  m_fields->get(d_particle, qes_grid, num_particle);
}

void InterpolationFF::getNew(particle_array d_new_particle, const QESgrid &qes_grid, const int &num_new_particle)
{
  m_fields->getNew(d_new_particle, qes_grid, num_new_particle);
}
