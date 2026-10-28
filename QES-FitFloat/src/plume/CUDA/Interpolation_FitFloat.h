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
/** @file Interpolation_FitFloat.h */

#pragma once

#include <string>
#include <cstddef>

#include "plume/Particle.h"
#include "plume/CUDA/QES_data.h"

class WINDSGeneralData;
class TURBGeneralData;

/**
 * @class InterpolationFF
 * @brief FitFloat study for QES-Plume: the wind and turbulence fields that the GPU particle model
 * interpolates from (read-only during a run) are stored bit-packed with FitFloat
 * (github.com/burtscher/FitFloat, BSD-3; src/winds/FitFloat.h). The math stays float: every read
 * decodes the packed element to a float and the trilinear interpolation is the same code as
 * Interpolation.cu, with the array type as a template parameter. Nothing is ever written to the
 * packed arrays after packing, so no atomics are involved.
 *
 * Variants, chosen with qesPlumeGPU -c <variant> (bits are sign+exponent+mantissa):
 *   ff-ctl     winds and turbulence in 32 bits (1+8+23): same values, no bytes saved -> the decode cost alone
 *   ff-wt24    both in 24 bits (1+8+15)
 *   ff-wt20    both in 20 bits (1+8+11)
 *   ff-wt16    both in 16 bits (1+8+7)
 *   ff-wt16e5  both in 16 bits (1+5+10): fp16-like, more mantissa, less exponent range
 *   ff-wt12    both in 12 bits (1+8+3)
 *   ff-w16     winds in 16 bits (1+8+7), turbulence in 32 bits (ctl)
 *   ff-t16     winds in 32 bits (ctl), turbulence in 16 bits (1+8+7)
 *
 * FitFloat.h is included only by Interpolation_FitFloat.cu (it defines non-inline functions).
 *
 * @sa Interpolation
 */
class InterpolationFF
{
public:
  InterpolationFF(const std::string &variant, const WINDSGeneralData *WGD, const TURBGeneralData *TGD);
  ~InterpolationFF();

  static bool isValid(const std::string &variant);
  static std::string variants();

  /** winds (velMean) and turbulence (CoEps, tau, flux_div, nuT) at the particle positions;
   *  same as Interpolation::get(particle_array, QESWindsData, QESTurbData, QESgrid, int). */
  void get(particle_array d_particle, const QESgrid &qes_grid, const int &num_particle);
  /** tau and sigma (velFluct_old) for new particles;
   *  same as Interpolation::get(particle_array, QESTurbData, QESgrid, int). */
  void getNew(particle_array d_new_particle, const QESgrid &qes_grid, const int &num_new_particle);

  double packingTime() const { return m_packingTime; }/**< seconds spent packing on the CPU, incl. the GPU decode check */
  size_t packedBytes() const;/**< device bytes of the packed fields */
  size_t nativeBytes() const { return m_nativeBytes; }/**< device bytes the same fields take as float */

  struct Fields;/**< the packed fields, one template instance per variant (Interpolation_FitFloat.cu) */

private:
  Fields *m_fields = nullptr;
  std::string m_variant;
  double m_packingTime = 0.0;
  size_t m_nativeBytes = 0;
};
