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
/** @file qesPlumeGPU.cpp
 * FitFloat study: benchmark driver for the GPU particle model (src/plume/CUDA), which is
 * otherwise only reachable from tests/unitTests/test_CUDA_plume.cpp. Same uniform-flow case and
 * Gaussian-plume check as that test, with the domain, release and run length on the command
 * line, and -c <variant> to store the wind/turbulence fields as FitFloat (Interpolation_FitFloat.h).
 * Without -c it runs the original GPU code: that is the baseline.
 *
 *   qesPlumeGPU [-c ff-wt16] [-n 1500] [-s 500] [-N 10000] [-L buffer] [-x 102 -y 102 -z 141] [-o basename]
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

#include <cuda.h>
#include <curand.h>

#include "util/ArgumentParsing.h"
#include <chrono>

// util/TimerTool.h's Timer has no accessor for the elapsed time, which the summary line needs
struct Stopwatch
{
  explicit Stopwatch(std::string n) : name(std::move(n)) { start(); }
  void start() { t0 = std::chrono::high_resolution_clock::now(); }
  void stop() { elapsed += std::chrono::high_resolution_clock::now() - t0; }
  double seconds() const { return elapsed.count(); }
  void show() const
  {
    std::string padded = name;
    if (padded.length() < 25) padded.append(25 - padded.length(), ' ');
    std::cout << "Elapsed time for " << padded << elapsed.count() << " s\n";
  }
  std::string name;
  std::chrono::duration<double> elapsed{ 0.0 };
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;
};
#include "util/VectorMath.h"
#include "util/QESFileOutput_v2.h"
#include "util/QESNetCDFOutput_v2.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "plume/ParticleIDGen.h"
#include "plume/Particle.h"
#include "plume/CUDA/QES_data.h"
#include "plume/CUDA/Interpolation.h"
#include "plume/CUDA/Interpolation_FitFloat.h"
#include "plume/CUDA/Partition.h"
#include "plume/CUDA/RandomGenerator.h"
#include "plume/CUDA/Model.h"
#include "plume/CUDA/Concentration.h"

class PlumeGPUArgs : public ArgumentParsing
{
public:
  PlumeGPUArgs()
  {
    reg("help", "help/usage information", ArgumentParsing::NONE, 'h');
    reg("fitFloat", "FitFloat variant for the wind/turbulence fields: " + InterpolationFF::variants() + " (default: original float arrays)", ArgumentParsing::STRING, 'c');
    reg("steps", "number of time steps of 1 s (default 1500)", ArgumentParsing::INT, 'n');
    reg("spinup", "steps before the concentration averaging starts (default 500)", ArgumentParsing::INT, 's');
    reg("newParticles", "particles released per step at (20, 50, 70) (default 10000)", ArgumentParsing::INT, 'N');
    reg("bufferLength", "particle buffer length (default: 1.25 x the particles that fit in the domain)", ArgumentParsing::INT, 'L');
    reg("nx", "domain cells in x (default 102; fields exceed the GPU's L2 cache from about 402 x 202 x 141)", ArgumentParsing::INT, 'x');
    reg("ny", "domain cells in y (default 102)", ArgumentParsing::INT, 'y');
    reg("nz", "domain cells in z (default 141)", ArgumentParsing::INT, 'z');
    reg("outbasename", "basename of the concentration file <basename>_plumeOut.nc (default: no file)", ArgumentParsing::STRING, 'o');
  }
  std::string fitFloat, outbasename;
  int steps = 1500, spinup = 500, newParticles = 10000, bufferLength = 0, nx = 102, ny = 102, nz = 141;

  void process(int argc, char *argv[])
  {
    processCommandLineArgs(argc, argv);
    if (isSet("help")) {
      printUsage();
      exit(EXIT_SUCCESS);
    }
    isSet("fitFloat", fitFloat);
    isSet("outbasename", outbasename);
    isSet("steps", steps);
    isSet("spinup", spinup);
    isSet("newParticles", newParticles);
    isSet("bufferLength", bufferLength);
    isSet("nx", nx);
    isSet("ny", ny);
    isSet("nz", nz);
    if (!fitFloat.empty() && !InterpolationFF::isValid(fitFloat)) {
      std::cerr << "[ERROR]\t unknown FitFloat variant '" << fitFloat << "' (valid: " << InterpolationFF::variants() << ")" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (nx < 60 || ny < 60 || nz < 80) {
      std::cerr << "[ERROR]\t the domain must hold the source (20, 50, 70) and the collection box (x 0-100, y 1-99, z 1-139): use nx >= 102, ny >= 102, nz >= 141, or at least 60 x 60 x 80" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};

// device copies of the native fields (QES_data.cpp is not part of the GPU library; same code as test_CUDA_plume.cpp)
static void copy_winds_gpu(const WINDSGeneralData *WGD, QESWindsData &d)
{
  long n = WGD->domain.numFaceCentered();
  cudaMalloc((void **)&d.u, n * sizeof(float));
  cudaMemcpy(d.u, WGD->u.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d.v, n * sizeof(float));
  cudaMemcpy(d.v, WGD->v.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d.w, n * sizeof(float));
  cudaMemcpy(d.w, WGD->w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

static void copy_turb_gpu(const TURBGeneralData *TGD, QESTurbData &d)
{
  long n = TGD->domain.numCellCentered();
  const std::vector<float> *src[] = { &TGD->txx, &TGD->txy, &TGD->txz, &TGD->tyy, &TGD->tyz, &TGD->tzz, &TGD->div_tau_x, &TGD->div_tau_y, &TGD->div_tau_z, &TGD->CoEps, &TGD->nuT, &TGD->tke };
  float **dst[] = { &d.txx, &d.txy, &d.txz, &d.tyy, &d.tyz, &d.tzz, &d.div_tau_x, &d.div_tau_y, &d.div_tau_z, &d.CoEps, &d.nuT, &d.tke };
  for (int f = 0; f < 12; f++) {
    cudaMalloc((void **)dst[f], n * sizeof(float));
    cudaMemcpy(*dst[f], src[f]->data(), n * sizeof(float), cudaMemcpyHostToDevice);
  }
}

static float calcRMSE(const std::vector<float> &A, const std::vector<float> &B)
{
  double s = 0.0;
  for (size_t k = 0; k < A.size(); ++k) s += pow(A[k] - B[k], 2);
  return (float)sqrt(s / (double)A.size());
}

static float calcMaxAbsErr(const std::vector<float> &A, const std::vector<float> &B)
{
  float m = 0.0;
  for (size_t k = 0; k < A.size(); ++k) m = std::max(m, std::abs(A[k] - B[k]));
  return m;
}

int main(int argc, char *argv[])
{
  PlumeGPUArgs args;
  args.process(argc, argv);

  std::cout << "###################################################################" << std::endl;
  std::cout << "[qesPlumeGPU]\t GPU particle model, uniform-flow case"
            << (args.fitFloat.empty() ? " (original float fields)" : " (FitFloat " + args.fitFloat + ")") << std::endl;
  std::cout << "###################################################################" << std::endl;

  Stopwatch initTimer("initialization");
  Stopwatch gpuTimer("GPU");
  Stopwatch gpuInitTimer("GPU initialization");
  Stopwatch timeLoopTimer("time loop");
  Stopwatch sourceTimer("new particles creation");
  Stopwatch partitonTimer("particle partitioning");
  Stopwatch interpTimer("interpolation");
  Stopwatch advectTimer("advection");
  Stopwatch concenTimer("concentration");

  // same flow and turbulence profiles as tests/unitTests/test_CUDA_plume.cpp
  float uMean = 2.0;
  float uStar = 0.174;
  float H = 70;
  float C0 = 5.7;
  float zi = 1000;

  QESgrid qes_grid;
  BC_Params bc_param;
  ConcentrationParam param;

  qes::Domain domain(args.nx, args.ny, args.nz, 1.0, 1.0, 1.0);
  std::tie(qes_grid.nx, qes_grid.ny, qes_grid.nz) = domain.getDomainCellNum();
  std::tie(qes_grid.dx, qes_grid.dy, qes_grid.dz) = domain.getDomainSize();

  auto *WGD = new WINDSGeneralData(domain);
  WGD->timestamp.emplace_back("2020-01-01T00:00:00");
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  for (int k = 0; k < domain.nz(); ++k) {
    for (int j = 0; j < domain.ny(); ++j) {
      for (int i = 0; i < domain.nx(); ++i) {
        WGD->u[domain.face(i, j, k)] = uMean;
      }
    }
  }
  for (int k = 1; k < domain.nz() - 1; ++k) {
    for (int j = 0; j < domain.ny() - 1; ++j) {
      for (int i = 0; i < domain.nx() - 1; ++i) {
        int cellID = domain.cell(i, j, k);
        TGD->txx[cellID] = pow(2.50 * uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);
        TGD->tyy[cellID] = pow(1.78 * uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);
        TGD->tzz[cellID] = pow(1.27 * uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);
        TGD->txz[cellID] = -pow(uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);
        TGD->tke[cellID] = pow(uStar / 0.55, 2.0);
        TGD->CoEps[cellID] = C0 * pow(uStar, 3) / (0.4 * domain.z[k]) * pow(1 - 0.85 * domain.z[k] / zi, 3.0 / 2.0);
      }
    }
  }
  TGD->divergenceStress();
  initTimer.stop();

  // particles leave the domain where the trilinear stencil would leave the arrays
  bc_param.xStartDomain = 0.5 * domain.dx();
  bc_param.yStartDomain = 0.5 * domain.dy();
  bc_param.zStartDomain = 0;
  bc_param.xEndDomain = (domain.nx() - 1.5) * domain.dx();
  bc_param.yEndDomain = (domain.ny() - 1.5) * domain.dy();
  bc_param.zEndDomain = (domain.nz() - 2.5) * domain.dz();

  // collection box of the test case
  param.lbndx = 0.0;
  param.lbndy = 1.0;
  param.lbndz = 1.0;
  param.ubndx = 100.0;
  param.ubndy = 99.0;
  param.ubndz = 139.0;
  param.nx = 20;
  param.ny = 49;
  param.nz = 69;
  param.dx = (param.ubndx - param.lbndx) / (param.nx);
  param.dy = (param.ubndy - param.lbndy) / (param.ny);
  param.dz = (param.ubndz - param.lbndz) / (param.nz);

  const int new_particle = args.newParticles;
  const int ntest = args.steps;
  int length = args.bufferLength;
  if (length <= 0) {
    // particles released per step x steps to cross the domain, with margin
    length = (int)(1.25 * new_particle * std::ceil((bc_param.xEndDomain - 20.0) / uMean)) + new_particle;
  }
  std::cout << "[qesPlumeGPU]\t domain " << args.nx << " x " << args.ny << " x " << args.nz << " cells, " << ntest << " steps, "
            << new_particle << " particles/step, buffer " << length << ", averaging from step " << args.spinup << std::endl;

  int gpuID = 0;
  if (cudaGetDevice(&gpuID) != cudaSuccess) {
    std::cerr << "[ERROR]\t no CUDA device" << std::endl;
    return EXIT_FAILURE;
  }

  gpuTimer.start();
  gpuInitTimer.start();

  auto *random = new RandomGenerator();
  auto *interpolation = new Interpolation();
  auto *model = new Model();

  QESWindsData d_qes_winds_data;
  QESTurbData d_qes_turb_data;
  InterpolationFF *ff = nullptr;
  if (args.fitFloat.empty()) {
    copy_winds_gpu(WGD, d_qes_winds_data);
    copy_turb_gpu(TGD, d_qes_turb_data);
  } else {
    ff = new InterpolationFF(args.fitFloat, WGD, TGD);
  }

  auto *concentration = new Concentration(param);
  auto *partition = new Partition(length);

  particle_array d_particle[2];
  partition->allocate_device_particle_list(d_particle[0], length);
  partition->allocate_device_particle_list(d_particle[1], length);
  random->create("advect", 3 * length);
  cudaDeviceSynchronize();
  gpuInitTimer.stop();

  float timeStep = 1.0;
  int idx = 0;

  timeLoopTimer.start();
  for (int k = 0; k < ntest; ++k) {
    int num_particle = partition->active() + new_particle;
    if (num_particle > length) {
      std::cerr << "[ERROR]\t particle buffer too small (" << num_particle << " > " << length << "): use -L" << std::endl;
      return EXIT_FAILURE;
    }

    partitonTimer.start();
    idx = partition->run(k, d_particle);
    partitonTimer.stop();

    sourceTimer.start();
    if (ff) {
      model->getNewParticle(new_particle, d_particle[idx], qes_grid, random, ff, partition);
    } else {
      model->getNewParticle(new_particle, d_particle[idx], d_qes_turb_data, qes_grid, random, interpolation, partition);
    }
    sourceTimer.stop();

    interpTimer.start();
    if (ff) {
      ff->get(d_particle[idx], qes_grid, num_particle);
    } else {
      interpolation->get(d_particle[idx], d_qes_winds_data, d_qes_turb_data, qes_grid, num_particle);
    }
    interpTimer.stop();

    advectTimer.start();
    model->advectParticle(d_particle[idx], num_particle, bc_param, random);
    advectTimer.stop();

    if (k >= args.spinup) {
      concenTimer.start();
      concentration->collect(timeStep, d_particle[idx], num_particle);
      concenTimer.stop();
    }
  }
  timeLoopTimer.stop();

  // any kernel failure in the loop would otherwise go unnoticed and produce NaN statistics
  cudaError_t cuda_err = cudaDeviceSynchronize();
  if (cuda_err == cudaSuccess) cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::cerr << "[ERROR]\t CUDA error during the time loop: " << cudaGetErrorString(cuda_err) << std::endl;
    return EXIT_FAILURE;
  }

  int active_count = 0, empty_count = 0;
  partition->check(d_particle[idx], active_count, empty_count);
  concentration->copyback();

  partition->free_device_particle_list(d_particle[0]);
  partition->free_device_particle_list(d_particle[1]);
  delete partition;
  delete random;
  delete interpolation;
  delete model;
  gpuTimer.stop();

  // ---- compare with the Gaussian plume model (test_CUDA_plume.cpp) ----
  float xS = 20, yS = 50, zS = 70;
  float Q = new_particle;
  float CNorm = (uMean * H * H / Q);
  float tAvg = concentration->ongoingAveragingTime;
  float CC = timeStep / tAvg / concentration->volume;

  std::vector<float> xBoxCen(param.nx), yBoxCen(param.ny), zBoxCen(param.nz);
  for (int k = 0; k < param.nz; ++k) zBoxCen[k] = param.lbndz + k * param.dz + param.dz / 2.0;
  for (int j = 0; j < param.ny; ++j) yBoxCen[j] = param.lbndy + j * param.dy + param.dy / 2.0;
  for (int i = 0; i < param.nx; ++i) xBoxCen[i] = param.lbndx + i * param.dx + param.dx / 2.0;

  float sigV = 1.78f * uStar;
  float sigW = powf(1.0f / (3.0f * 0.4f * 0.4f), 1.0f / 3.0f) * uStar;
  float SSres = 0.0, SStot = 0.0;
  std::vector<float> xoH, maxRelErr, relRMSE;
  for (int i = 7; i < param.nx; ++i) {
    float x = xBoxCen[i] - xS;
    float t = x / uMean;
    float Ti = powf(2.5f * uStar / zi, -1);
    float Fy = powf(1.0f + powf(t / Ti, 0.5), -1);
    float Fz = Fy;
    float sigY = sigV * t * Fy;
    float sigZ = sigW * t * Fz;
    std::vector<float> CStarModel(param.ny * param.nz, 0.0), CStarQES(param.ny * param.nz, 0.0);
    for (int k = 0; k < param.nz; ++k) {
      for (int j = 0; j < param.ny; ++j) {
        float y = yBoxCen[j] - yS;
        float z = zBoxCen[k] - zS;
        CStarModel[j + k * param.ny] = Q / (2 * M_PI * uMean * sigY * sigZ) * expf(-0.5f * powf(y, 2) / powf(sigY, 2)) * expf(-0.5f * powf(z, 2) / powf(sigZ, 2)) * CNorm;
        int id = i + j * param.nx + k * param.nx * param.ny;
        CStarQES[j + k * param.ny] = concentration->h_pBox[id] * CC * CNorm;
      }
    }
    auto maxModel = *max_element(std::begin(CStarModel), std::end(CStarModel));
    xoH.push_back(x / H);
    maxRelErr.push_back(calcMaxAbsErr(CStarQES, CStarModel) / maxModel);
    relRMSE.push_back(calcRMSE(CStarQES, CStarModel) / maxModel);
    float CStarQESMean = 0.0;
    for (auto v : CStarQES) CStarQESMean += v;
    CStarQESMean /= (float)CStarQES.size();
    for (size_t k = 0; k < CStarQES.size(); ++k) {
      SSres += powf(CStarModel[k] - CStarQES[k], 2);
      SStot += powf(CStarQES[k] - CStarQESMean, 2);
    }
  }
  float R2 = 1.0f - SSres / SStot;

  if (!args.outbasename.empty()) {
    auto *outfile = new QESNetCDFOutput_v2(args.outbasename + "_plumeOut.nc");
    outfile->newDimension("x_c", "x-center collection box", "m", &xBoxCen);
    outfile->newDimension("y_c", "y-center collection box", "m", &yBoxCen);
    outfile->newDimension("z_c", "z-center collection box", "m", &zBoxCen);
    outfile->newDimensionSet("concentration", { "t", "z_c", "y_c", "x_c" });
    outfile->newField("t_avg", "Averaging time", "s", "time", &concentration->ongoingAveragingTime);
    outfile->newField("p_count", "number of particle per box", "#ofPar", "concentration", &concentration->h_pBox);
    std::vector<float> CStar(param.nx * param.ny * param.nz);
    for (size_t id = 0; id < CStar.size(); ++id) CStar[id] = concentration->h_pBox[id] * CC * CNorm;
    outfile->newField("c", "normalized concentration", "--", "concentration", &CStar);
    QEStime time;
    outfile->pushAllFieldsToFile(time);
    delete outfile;
  }

  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "Gaussian plume check:  x / H \t maxRelErr \t relRMSE" << std::endl;
  for (size_t k = 0; k < xoH.size(); ++k) printf("%.3f \t %.6f \t %.6f\n", xoH[k], maxRelErr[k], relRMSE[k]);
  std::cout << "QES-Plume r2 coeff: r2 = " << R2 << std::endl;
  std::cout << "-------------------------------------------------------------------" << std::endl;
  initTimer.show();
  gpuTimer.show();
  gpuInitTimer.show();
  timeLoopTimer.show();
  partitonTimer.show();
  sourceTimer.show();
  interpTimer.show();
  advectTimer.show();
  concenTimer.show();
  std::cout << "-------------------------------------------------------------------" << std::endl;
  // one line for scripts
  printf("[qesPlumeGPU]\t summary: variant=%s domain=%dx%dx%d steps=%d new=%d buffer=%d active=%d loop=%.4f interp=%.4f advect=%.4f source=%.4f partition=%.4f conc=%.4f packing=%.4f r2=%.5f\n",
         args.fitFloat.empty() ? "original" : args.fitFloat.c_str(), args.nx, args.ny, args.nz, ntest, new_particle, length, active_count,
         timeLoopTimer.seconds(), interpTimer.seconds(), advectTimer.seconds(), sourceTimer.seconds(), partitonTimer.seconds(), concenTimer.seconds(),
         ff ? ff->packingTime() : 0.0, R2);

  delete concentration;
  delete ff;
  delete WGD;
  delete TGD;
  return EXIT_SUCCESS;
}
