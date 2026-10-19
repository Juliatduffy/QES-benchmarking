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

/** @file Solver_GPU_FitFloat.h */

#pragma once

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Solver.h"

// FitFloat (github.com/burtscher/FitFloat, BSD-3) is included only by Solver_GPU_FitFloat.cu.
template<typename T, unsigned int const USR_BITS_EXPO, unsigned int const USR_BITS_MANT>
class FitFloatArray;


/**
 * @class Solver_GPU_FitFloat
 * @brief FitFloat study: the Global Memory GPU solver with some of its arrays
 * stored bit-packed (FitFloat). The math stays float; only the storage shrinks.
 *
 * Variants, chosen with -s 3 -c <variant>:
 *   ff-ctl      e..n stored as 32-bit FitFloat (1+8+23): same values, no bytes saved (measures the unpacking cost)
 *   ff-en16     e..n stored as 16-bit FitFloat (1+8+7)
 *   ff-en4      e..n stored as  4-bit FitFloat (1+3+0)
 *   ff-r16      R stored as 16-bit FitFloat (1+8+7), written by the divergence kernel every solve
 *   ff-en16r16  both e..n and R as 16-bit
 *
 * e..n are packed once on the host (first solve) and stay on the GPU. Mantissas are
 * truncated, so packing may change values (that is the point of the study), but if any
 * nonzero coefficient would become 0 or inf (out of the format's range) the solver uses
 * plain float e..n instead, because a zero row would divide by zero in the SOR update.
 *
 * @sa Solver_GPU_GlobalMemory
 */
class Solver_GPU_FitFloat : public Solver
{
private:
  template<typename T>
  void _cudaCheck(T e, const char *func, const char *call, const int line);

  /** One solve with coefficient arrays of type C and R of type RT (float* or FitFloatArray). */
  template<class C, class RT>
  void runSolver(WINDSGeneralData *WGD, const int &itermax, C ce, C cf, C cg, C ch, C cm, C cn, RT dR);

  /** Picks the coefficient storage (packed or float) and calls runSolver with R of type RT. */
  template<class RT>
  void runWithCoefficients(WINDSGeneralData *WGD, const int &itermax, RT dR);

  std::string variant; /**< ff-ctl, ff-en16, ff-en4, ff-r16 or ff-en16r16 */
  int coef_bits = 0; /**< requested e..n storage: 0 = float, 32, 16 or 4 */
  bool r16 = false; /**< R stored as 16-bit FitFloat */
  int coef_kind = -1; /**< -1 not packed yet, 0 float, 32/16/4 = packed with that many bits */

  FitFloatArray<float, 8, 23> *ff32[6] = {}; /**< 1 sign + 8 exponent + 23 mantissa bits (control) */
  FitFloatArray<float, 8, 7> *ff16[6] = {}; /**< 1 + 8 + 7 */
  FitFloatArray<float, 3, 0> *ff4[6] = {}; /**< 1 + 3 + 0 */
  FitFloatArray<float, 8, 7> *R16 = nullptr; /**< R, 1 + 8 + 7 */

public:
  Solver_GPU_FitFloat(qes::Domain domain_in, const float &tolerance, const std::string &variant);

protected:
  ///@{
  /** Solver coefficient on device (GPU), float fallback */
  float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n;
  ///@}
  float *d_R; /**< Divergence of initial velocity field on device (GPU), float version */
  ///@{
  /** Lagrange multiplier on device (GPU) */
  float *d_lambda, *d_lambda_old;
  ///@}

  virtual void solve(WINDSGeneralData *, const int &) override;
};
