/*
# This file is part of FitFloat, a drop-in floating-point array replacement supporting user-specified precision on GPUs with the goal of reducing storage requirements.
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Andrew Rodriguez and Martin Burtscher
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# URL: The latest version of this code is available at https://github.com/burtscher/FitFloat/.
#
# Publication: This work is described in detail in the following paper.
# Andrew Rodriguez and Martin Burtscher. FitFloat: Read/Write Random-Access Compressed Floating-Point Arrays for GPUs. Proceedings of the 2026 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2026.
#
# Sponsor: This work has been supported by the U.S. Department of Energy, National Nuclear Security Administration under Award DE-NA0003969, by the U.S. National Science Foundation under Award CCF-2403380, and by an equipment donation from NVIDIA Corporation.
*/


#include <cuda.h>
#include <cstdio>
#include <omp.h>


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  #define __any(arg) __any_sync(~0, arg)
  #define __shfl(...) __shfl_sync(~0, __VA_ARGS__)
  #define __shfl_xor(...) __shfl_xor_sync(~0, __VA_ARGS__)
#endif


#define WS 32
#define FF_TPB 512


#ifndef BITS_FOR_EXPONENT_32
  #define BITS_FOR_EXPONENT_32 8
#endif

#ifndef BITS_FOR_MANTISSA_32
  #define BITS_FOR_MANTISSA_32 23
#endif

#ifndef BITS_FOR_EXPONENT_64
  #define BITS_FOR_EXPONENT_64 11
#endif

#ifndef BITS_FOR_MANTISSA_64
  #define BITS_FOR_MANTISSA_64 52
#endif


size_t get_ffarray_size_float(size_t elements, size_t ff_bits)
{
  using U = float;

  const size_t rounded_elements = (elements + 31) / 32 * 32;
  const size_t bits_per_byte = 8;
  const size_t minimum_bytes = (ff_bits * rounded_elements + bits_per_byte - 1) / bits_per_byte;
  const size_t final_num_elements = (minimum_bytes + sizeof(U) - 1) / sizeof(U) + 1;

  return final_num_elements;
}


size_t get_ffarray_size_double(size_t elements, size_t ff_bits)
{
  using U = double;

  const size_t rounded_elements = (elements + 31) / 32 * 32;
  const size_t bits_per_byte = 8;
  const size_t minimum_bytes = (ff_bits * rounded_elements + bits_per_byte - 1) / bits_per_byte;
  const size_t final_num_elements = (minimum_bytes + sizeof(U) - 1) / sizeof(U) + 1;

  return final_num_elements;
}


// ***************
// ** FitFloat **
// ***************


template <typename T, unsigned int const USR_BITS_EXPO, unsigned int const USR_BITS_MANT>
class FitFloatArray
{
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "FitFloatArray can only be instantiated with float or double types");
  using U = typename std::conditional<std::is_same<T, float>::value, unsigned int, unsigned long long>::type;
  static_assert(sizeof(T) == sizeof(U), "FitFloatArray internal error: type sizes do not match");
  static_assert(sizeof(T) * 8 >= 1 + USR_BITS_EXPO + USR_BITS_MANT, "FitFloatArray number of bits exceeded");

  private:
    U* data;
    size_t given_size;
    size_t backing_size;
    bool is_copy;

    static constexpr unsigned int TB = sizeof(T) * 8;  // number of bits in T

    static constexpr unsigned int BITS_SIGN = 1;
    static constexpr unsigned int BITS_EXPO = (TB == 32) ? 8 : 11;
    static constexpr unsigned int BITS_MANT = (TB == 32) ? 23 : 52;

    static_assert(USR_BITS_EXPO <= BITS_EXPO, "FitFloatArray number of exponent bits exceeded");
    static_assert(USR_BITS_MANT <= BITS_MANT, "FitFloatArray number of mantissa bits exceeded");
    
    static constexpr unsigned int BITS_TOTAL = BITS_SIGN + BITS_EXPO + BITS_MANT;
    static constexpr unsigned int USR_BITS_TOTAL = 1 + USR_BITS_EXPO + USR_BITS_MANT;

    static constexpr U MASK_FOR_EXPO = (1ULL << BITS_EXPO) - 1;
    static constexpr U MASK_FOR_MANT = (1ULL << BITS_MANT) - 1;
    static constexpr unsigned int SHIFT_FOR_SIGN = BITS_TOTAL - BITS_SIGN;
    static constexpr unsigned int SHIFT_FOR_EXPO = SHIFT_FOR_SIGN - BITS_EXPO;

    static constexpr U ALL_ONES_EXPO = (1ULL << BITS_EXPO) - 1;
    static constexpr U USR_ALL_ONES_EXPO = (1ULL << USR_BITS_EXPO) - 1;
    static constexpr U QUIET_NAN_MANT = (1ULL << BITS_MANT) - 1; // lower BITS_MANT bits set to 1

    static constexpr U BIAS = (1ULL << (BITS_EXPO - 1)) - 1;
    static constexpr U USR_BIAS = (1ULL << (USR_BITS_EXPO - 1)) - 1;
    static constexpr U ADJUST_BIAS = BIAS - USR_BIAS;
    static constexpr U USR_MAX_EXPO = BIAS + USR_BIAS;
    static constexpr U USR_MIN_EXPO = BIAS - USR_BIAS + 1;

    static constexpr U USR_BITS_MASK = (USR_BITS_TOTAL == 64) ? 0xFFFFFFFFFFFFFFFF : (1ULL << USR_BITS_TOTAL) - 1; // can't shift 1 by 64, so manually set mask
    static constexpr U USR_MASK_FOR_MANT = (1ULL << USR_BITS_MANT) - 1;
    static constexpr U USR_MASK_FOR_EXPO = (1ULL << USR_BITS_EXPO) - 1;

    static constexpr unsigned int MANT_SHIFT = BITS_MANT - USR_BITS_MANT;
    static constexpr U MANTISSA_ERROR_CORRECTION = (USR_BITS_MANT == BITS_MANT) ? 0LL : 1ULL << (MANT_SHIFT - 1);
    static constexpr unsigned int USR_SIGN_SHIFT = USR_BITS_EXPO + USR_BITS_MANT;

    class Proxy
    {
    private:
      size_t idx;
      U* data;

    public:
      __device__
      Proxy(size_t idx, U* data) : idx(idx), data(data) {}

      __device__
      Proxy& operator=(const T& val)
      {
        // float, double -> int, long long
        const U val_as_integer = reinterpret_cast<const U&>(val);

        // shift and mask the sign, exponent, mantissa
        const U sign = val_as_integer >> SHIFT_FOR_SIGN;
        U exponent = (val_as_integer >> SHIFT_FOR_EXPO) & MASK_FOR_EXPO;
        U mantissa = val_as_integer & MASK_FOR_MANT;

        // exponent special cases

        if (exponent == ALL_ONES_EXPO) { // NaN, Inf

          if (mantissa != 0) {
            mantissa = QUIET_NAN_MANT; // quiet and signaling -> quiet
          }
          exponent = USR_ALL_ONES_EXPO;

        } else { // normals

          if (exponent > USR_MAX_EXPO) { // round to Inf
            exponent = USR_ALL_ONES_EXPO;
            mantissa = 0;
          } else if (exponent < USR_MIN_EXPO) { // round to 0
            exponent = 0;
            mantissa = 0;
          } else {
            exponent -= ADJUST_BIAS; // rebias to usr def bias
          }

        }

        // truncate mantissa
        mantissa = mantissa >> MANT_SHIFT;

        // pack the bits together
        const U packed = (sign << USR_SIGN_SHIFT) | (exponent << USR_BITS_MANT) | mantissa;

        // compute positions, shifts, masks
        const size_t bit_pos = idx * USR_BITS_TOTAL;
        const size_t arr_pos = bit_pos / TB;

        const unsigned int shift_lo = bit_pos % TB;
        const U mask_lo = ~(USR_BITS_MASK << shift_lo);

        #ifdef USE_AND_OR
          atomicAnd(&data[arr_pos], mask_lo);
          atomicOr(&data[arr_pos], packed << shift_lo);
        #else
          U lo, old_lo;
          do
          {
            old_lo = data[arr_pos];
            lo = (old_lo & mask_lo) | (packed << shift_lo);
          } while (atomicCAS(&data[arr_pos], old_lo, lo) != old_lo);
        #endif

        const unsigned int shift_hi = TB - shift_lo;

        if (shift_hi < USR_BITS_TOTAL) 
        {
          const U mask_hi = ~(USR_BITS_MASK >> shift_hi);

          #ifdef USE_AND_OR
            atomicAnd(&data[arr_pos + 1], mask_hi);
            atomicOr(&data[arr_pos + 1], packed >> shift_hi);
          #else
            U hi, old_hi;
            do
            {
              old_hi = data[arr_pos + 1]; // read
              hi = (old_hi & mask_hi) | (packed >> shift_hi);
            } while (atomicCAS(&data[arr_pos + 1], old_hi, hi) != old_hi);
          #endif
        }

        return *this;
      }

      // read
      __device__
      operator T() const
      {
        const size_t bit_pos = idx * USR_BITS_TOTAL;
        const size_t arr_pos = bit_pos / TB;

        const U lo = data[arr_pos];
        const U hi = data[arr_pos + 1];

        const unsigned int shift = bit_pos % TB;

        U concat;

        if constexpr (TB == 32) {
          concat = __funnelshift_rc(lo, hi, shift);
        } else {
          concat = lo >> shift;
          if (TB - USR_BITS_TOTAL < shift) {
            concat |= hi << (TB - shift);
          }
        }

        const U mantissa = concat & USR_MASK_FOR_MANT;
        U exponent = (concat >> USR_BITS_MANT) & USR_MASK_FOR_EXPO;
        const U sign = concat >> USR_SIGN_SHIFT;  // also contains garbage bits but that's okay

        // exponent special cases
        if (exponent == USR_ALL_ONES_EXPO) {
          exponent = ALL_ONES_EXPO;
        } else if (exponent != 0) {
          exponent += ADJUST_BIAS; // rebias to IEEE bias
        }

        U reconstructed = (sign << (BITS_EXPO + BITS_MANT)) | (exponent << BITS_MANT) | (mantissa << MANT_SHIFT);

        return reinterpret_cast<T&>(reconstructed);
      }
    };

    public:
    __device__
    void quick_write(const size_t& idx, const T& valp)
    {
      // float, double -> int, long long
      const U val_as_integer = reinterpret_cast<const U&>(valp);

      // shift and mask the sign, exponent, mantissa
      const U sign = val_as_integer >> SHIFT_FOR_SIGN;
      U exponent = (val_as_integer >> SHIFT_FOR_EXPO) & MASK_FOR_EXPO;
      U mantissa = val_as_integer & MASK_FOR_MANT;

      // exponent special cases
      if (exponent == ALL_ONES_EXPO) { // NaN, Inf

        if (mantissa != 0) {
          mantissa = QUIET_NAN_MANT; // quiet and signaling -> quiet
        }
        exponent = USR_ALL_ONES_EXPO;

      } else { // normals

        if (exponent > USR_MAX_EXPO) { // round to Inf
          exponent = USR_ALL_ONES_EXPO;
          mantissa = 0;
        } else if (exponent < USR_MIN_EXPO) { // round to 0
          exponent = 0;
          mantissa = 0;
        } else {
          exponent -= ADJUST_BIAS; // rebias to usr def bias
        }

      }

      // truncate mantissa
      mantissa = mantissa >> MANT_SHIFT;

      // pack the bits together
      U val = (sign << USR_SIGN_SHIFT) | (exponent << USR_BITS_MANT) | mantissa;

      // compute constants
      const unsigned int lane = idx % WS;
      const unsigned int warp = idx / WS;
      const size_t words = WS * USR_BITS_TOTAL / TB;
      const size_t wpos = warp * words;

      // combine subwords into subwords of at least 17 or 33 bits
      unsigned int width1 = USR_BITS_TOTAL;
      unsigned int dist1 = 1;

      while (width1 <= TB / 2) {
        const U recv = __shfl_xor(val, dist1);

        dist1 *= 2;
        if (lane & (dist1 - 1) == 0) {
          val |= recv << width1;
        }
        width1 *= 2;
      }

      const unsigned int pos1 = lane * TB;
      unsigned int src = (pos1 / width1) * dist1;
      const unsigned int drop1 = pos1 % width1;
      unsigned int bits = width1 - drop1;

      // combine larger subwords into actual words
      U res = __shfl(val, src);
      res >>= drop1;

      while (__any((bits < TB) && (lane < words))) {
        src += dist1;
        const U recv = __shfl(val, src);
        if (bits < TB) {
          res |= recv << bits;
        }

        bits += width1;
      }

      // write words
      if (lane < words) {
        data[wpos + lane] = res;
      }
    }

    __device__
    void quick_write_two_values(const size_t& idx, const T& valp1, const T& valp2)
    {
      // float, double -> int, long long
      const U val1_as_integer = reinterpret_cast<const U&>(valp1);

      // shift and mask the sign, exponent, mantissa
      U sign = val1_as_integer >> SHIFT_FOR_SIGN;
      U exponent = (val1_as_integer >> SHIFT_FOR_EXPO) & MASK_FOR_EXPO;
      U mantissa = val1_as_integer & MASK_FOR_MANT;

      // exponent special cases
      if (exponent == ALL_ONES_EXPO) { // NaN, Inf

        if (mantissa != 0) {
          mantissa = QUIET_NAN_MANT; // quiet and signaling -> quiet
        }
        exponent = USR_ALL_ONES_EXPO;

      } else { // normals

        if (exponent > USR_MAX_EXPO) { // round to Inf
          exponent = USR_ALL_ONES_EXPO;
          mantissa = 0;
        } else if (exponent < USR_MIN_EXPO) { // round to 0
          exponent = 0;
          mantissa = 0;
        } else {
          exponent -= ADJUST_BIAS; // rebias to usr def bias
        }

      }

      // truncate mantissa
      mantissa = mantissa >> MANT_SHIFT;

      // pack the bits together
      U val1 = (sign << USR_SIGN_SHIFT) | (exponent << USR_BITS_MANT) | mantissa;

      // float, double -> int, long long
      const U val2_as_integer = reinterpret_cast<const U&>(valp2);

      // shift and mask the sign, exponent, mantissa
      sign = val2_as_integer >> SHIFT_FOR_SIGN;
      exponent = (val2_as_integer >> SHIFT_FOR_EXPO) & MASK_FOR_EXPO;
      mantissa = val2_as_integer & MASK_FOR_MANT;

      // exponent special cases
      if (exponent == ALL_ONES_EXPO) { // NaN, Inf

        if (mantissa != 0) {
          mantissa = QUIET_NAN_MANT; // quiet and signaling -> quiet
        }
        exponent = USR_ALL_ONES_EXPO;

      } else { // normals

        if (exponent > USR_MAX_EXPO) { // round to Inf
          exponent = USR_ALL_ONES_EXPO;
          mantissa = 0;
        } else if (exponent < USR_MIN_EXPO) { // round to 0
          exponent = 0;
          mantissa = 0;
        } else {
          exponent -= ADJUST_BIAS; // rebias to usr def bias
        }

      }

      // truncate mantissa
      mantissa = mantissa >> MANT_SHIFT;

      // pack the bits together
      U val2 = (sign << USR_SIGN_SHIFT) | (exponent << USR_BITS_MANT) | mantissa;

      if constexpr ((TB == 32 && USR_BITS_TOTAL <= 16) || (TB == 64 && USR_BITS_TOTAL <= 32)) {
        U val = val1 | val2 << USR_BITS_TOTAL;

        // compute constants
        const unsigned int lane = idx % WS;
        const unsigned int warp = idx / WS;
        const size_t words = WS * USR_BITS_TOTAL * 2 / TB;
        const size_t wpos = warp * words;

        // combine subwords into subwords of at least 33 bits
        unsigned int width1 = USR_BITS_TOTAL * 2;
        unsigned int dist1 = 1;

        while (width1 <= TB / 2) {
          const U recv = __shfl_xor(val, dist1);

          dist1 *= 2;
          if (lane & (dist1 - 1) == 0) {
            val |= recv << width1;
          }
          width1 *= 2;
        }

        const unsigned int pos1 = lane * TB;
        unsigned int src = (pos1 / width1) * dist1;
        const unsigned int drop1 = pos1 % width1;
        unsigned int bits = width1 - drop1;

        // combine larger subwords into actual words
        U res = __shfl(val, src);
        res >>= drop1;

        while (__any((bits < TB) && (lane < words))) {
          src += dist1;
          const U recv = __shfl(val, src);
          if (bits < TB) {
            res |= recv << bits;
          }

          bits += width1;
        }

        // write words
        if (lane < words) {
          data[wpos + lane] = res;
        }
      } else {
        const U lo = val1 | (val2 << USR_BITS_TOTAL);
        const U hi = val2 >> (TB - USR_BITS_TOTAL);

        // compute constants
        const unsigned int lane = idx % WS;
        const unsigned int warp = idx / WS;
        const size_t words = WS * USR_BITS_TOTAL * 2 / TB;
        const size_t wpos = warp * words;

        const unsigned int width1 = USR_BITS_TOTAL * 2;
        const unsigned int dist1 = 1;
        const unsigned int pos1 = lane * (TB * 2);
        unsigned int src = pos1 / width1;
        const unsigned int drop1 = pos1 % width1;
        unsigned int bits = width1 - drop1;

        // combine larger subwords into actual words
        U res_lo = __shfl(lo, src);
        U res_hi = __shfl(hi, src);

        if (drop1 >= TB) {
          res_lo = res_hi >> (drop1 - TB);
          res_hi = 0;
        } else {
          res_lo >>= drop1;
          res_lo |= res_hi << (TB - drop1);
          res_hi >>= drop1;
        }

        while (__any((bits < TB * 2) && (lane * 2 < words))) {
          src += dist1;
          U recv_lo = __shfl(lo, src);
          U recv_hi = __shfl(hi, src);
          if (bits < TB * 2) {
            if (bits >= TB) {
              recv_hi = recv_lo << (bits - TB);
              recv_lo = 0;
            } else {
              recv_hi <<= bits;
              recv_hi |= recv_lo >> (TB - bits);
              recv_lo <<= bits;
            }
            res_hi |= recv_hi;
            res_lo |= recv_lo;
          }

          bits += width1;
        }

        // write words
        const size_t lane_word_hi = lane * 2 + 1;
        if (lane_word_hi <= words) {
          data[wpos + lane_word_hi - 1] = res_lo;
          if (lane_word_hi < words) {
            data[wpos + lane_word_hi] = res_hi;
          }
        }
      }
    }

    FitFloatArray(size_t sizep) // constructor
    : given_size(sizep),
      is_copy(false)
    {
      const size_t rounded_size = (given_size + 31) / 32 * 32;
      const size_t bits_per_byte = 8;
      const size_t size_in_bytes = (USR_BITS_TOTAL * rounded_size + bits_per_byte - 1) / bits_per_byte;

      backing_size = (size_in_bytes + sizeof(U) - 1) / sizeof(U) + 1;

      // QES change (2026-10-19): plain device memory instead of cudaMallocManaged + a host zero-fill.
      // Under WSL2 the GPU has no concurrent managed access, so managed arrays are never prefetched
      // to the GPU (suspected cause of the 19x slower 32-bit control on the RTX 4070 SUPER).
      // Fill it with cudaMemcpy(..., cudaMemcpyHostToDevice) or from a kernel.
      cudaMalloc(&data, sizeof(U) * backing_size);
      cudaMemset(data, 0, sizeof(U) * backing_size);
    }

    FitFloatArray(FitFloatArray& original) // copy constructor
    : given_size(original.given_size),
      backing_size(original.backing_size),
      data(original.data), is_copy(true) {}

    ~FitFloatArray() // destructor
    {
      if (!is_copy) {
        cudaFree(data);
      }
    }

    __device__
    Proxy operator[](size_t idx)
    {
      return Proxy(idx, data);
    }

    size_t size() const noexcept { return given_size; }

    size_t actual_size() const noexcept { return backing_size; }

    U* data_pointer() const noexcept { return data; }
};

template <typename T, unsigned int const USR_BITS_EXPO, unsigned int const USR_BITS_MANT, bool ftz = true, bool rnd = false>
class HostFitFloatArray
{
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "HostFitFloatArray can only be instantiated with float or double types");
  using U = typename std::conditional<std::is_same<T, float>::value, unsigned int, unsigned long long>::type;
  static_assert(sizeof(T) == sizeof(U), "HostFitFloatArray internal error: type sizes do not match");
  static_assert(sizeof(T) * 8 >= 1 + USR_BITS_EXPO + USR_BITS_MANT, "HostFitFloatArray number of bits exceeded");

  private:
    U* data;
    size_t given_size;
    size_t backing_size;
    bool is_copy;

    static constexpr unsigned int TB = sizeof(T) * 8;  // number of bits in T

    static constexpr unsigned int BITS_SIGN = 1;
    static constexpr unsigned int BITS_EXPO = (TB == 32) ? 8 : 11;
    static constexpr unsigned int BITS_MANT = (TB == 32) ? 23 : 52;
    
    static_assert(USR_BITS_EXPO <= BITS_EXPO, "FitFloatArray number of exponent bits exceeded");
    static_assert(USR_BITS_MANT <= BITS_MANT, "FitFloatArray number of mantissa bits exceeded");

    static constexpr unsigned int BITS_TOTAL = BITS_SIGN + BITS_EXPO + BITS_MANT;
    static constexpr unsigned int USR_BITS_TOTAL = 1 + USR_BITS_EXPO + USR_BITS_MANT;

    static constexpr U MASK_FOR_EXPO = (1ULL << BITS_EXPO) - 1;
    static constexpr U MASK_FOR_MANT = (1ULL << BITS_MANT) - 1;
    static constexpr unsigned int SHIFT_FOR_SIGN = BITS_TOTAL - BITS_SIGN;
    static constexpr unsigned int SHIFT_FOR_EXPO = SHIFT_FOR_SIGN - BITS_EXPO;

    static constexpr U ALL_ONES_EXPO = (1ULL << BITS_EXPO) - 1;
    static constexpr U USR_ALL_ONES_EXPO = (1ULL << USR_BITS_EXPO) - 1;
    static constexpr U QUIET_NAN_MANT = (1ULL << BITS_MANT) - 1; // lower BITS_MANT bits set to 1

    static constexpr U BIAS = (1ULL << (BITS_EXPO - 1)) - 1;
    static constexpr U USR_BIAS = (1ULL << (USR_BITS_EXPO - 1)) - 1;
    static constexpr U ADJUST_BIAS = BIAS - USR_BIAS;
    static constexpr U USR_MAX_EXPO = BIAS + USR_BIAS;
    static constexpr U USR_MIN_EXPO = BIAS - USR_BIAS + 1;

    static constexpr U USR_BITS_MASK = (USR_BITS_TOTAL == 64) ? 0xFFFFFFFFFFFFFFFF : (1ULL << USR_BITS_TOTAL) - 1; // can't shift 1 by 64, so manually set mask
    static constexpr U USR_MASK_FOR_MANT = (1ULL << USR_BITS_MANT) - 1;
    static constexpr U USR_MASK_FOR_EXPO = (1ULL << USR_BITS_EXPO) - 1;

    static constexpr unsigned int MANT_SHIFT = BITS_MANT - USR_BITS_MANT;
    static constexpr U MANTISSA_ERROR_CORRECTION = (USR_BITS_MANT == BITS_MANT) ? 0LL : 1ULL << (MANT_SHIFT - 1);
    static constexpr unsigned int USR_SIGN_SHIFT = USR_BITS_EXPO + USR_BITS_MANT;

    class Proxy
    {
    private:
      size_t idx;
      U* data;

    public:
      Proxy(size_t idx, U* data) : idx(idx), data(data) {}

      Proxy& operator=(const T& val)
      {
        // float, double -> int, long long
        const U val_as_integer = reinterpret_cast<const U&>(val);

        // shift and mask the sign, exponent, mantissa
        const U sign = val_as_integer >> SHIFT_FOR_SIGN;
        U exponent = (val_as_integer >> SHIFT_FOR_EXPO) & MASK_FOR_EXPO;
        U mantissa = val_as_integer & MASK_FOR_MANT;

        // exponent special cases

        if (exponent == ALL_ONES_EXPO) { // NaN, inf

          if (mantissa != 0) {
            mantissa = QUIET_NAN_MANT; // quiet and signaling -> quiet
          }
          exponent = USR_ALL_ONES_EXPO;

        } else { // normals

          if (exponent > USR_MAX_EXPO) { // round to inf
            exponent = USR_ALL_ONES_EXPO;
            mantissa = 0;
          } else if (exponent < USR_MIN_EXPO) { // round to 0
            exponent = 0;
            mantissa = 0;
          } else {
            exponent -= ADJUST_BIAS; // rebias to usr def bias
          }

        }

        // truncate mantissa
        mantissa = mantissa >> MANT_SHIFT;

        // pack the bits together
        const U packed = (sign << USR_SIGN_SHIFT) | (exponent << USR_BITS_MANT) | mantissa;

        // compute positions, shifts, masks
        const size_t bit_pos = idx * USR_BITS_TOTAL;
        const size_t arr_pos = bit_pos / TB;

        const unsigned int shift_lo = bit_pos % TB;
        const U mask_lo = ~(USR_BITS_MASK << shift_lo);

        #pragma omp atomic update
        data[arr_pos] &= mask_lo;
        #pragma omp atomic update
        data[arr_pos] |= (packed << shift_lo);

        const unsigned int shift_hi = TB - shift_lo;

        if (shift_hi < USR_BITS_TOTAL) 
        {
          const U mask_hi = ~(USR_BITS_MASK >> shift_hi);

          #pragma omp atomic update
          data[arr_pos + 1] &= mask_hi;
          #pragma omp atomic update
          data[arr_pos + 1] |= (packed >> shift_hi);
        }

        return *this;
      }

      // read
      operator T() const
      {
        const size_t bit_pos = idx * USR_BITS_TOTAL;
        const size_t arr_pos = bit_pos / TB;

        const U lo = data[arr_pos];
        const U hi = data[arr_pos + 1];

        const unsigned int shift = bit_pos % TB;

        U concat;

        concat = lo >> shift;
        if (TB - USR_BITS_TOTAL < shift) {
          concat |= hi << (TB - shift);
        }

        const U mantissa = concat & USR_MASK_FOR_MANT;
        U exponent = (concat >> USR_BITS_MANT) & USR_MASK_FOR_EXPO;
        const U sign = concat >> USR_SIGN_SHIFT;  // also contains garbage bits but that's okay

        // exponent special cases
        if (exponent == USR_ALL_ONES_EXPO) {
          exponent = ALL_ONES_EXPO;
        } else if (exponent != 0) {
          exponent += ADJUST_BIAS; // rebias to IEEE bias
        }

        U reconstructed = (sign << (BITS_EXPO + BITS_MANT)) | (exponent << BITS_MANT) | (mantissa << MANT_SHIFT);

        return reinterpret_cast<T&>(reconstructed);
      }
    };

    public:
    void quick_write(const size_t& idx, const T& valp)
    {
      // float, double -> int, long long
      const U val_as_integer = reinterpret_cast<const U&>(valp);

      // shift and mask the sign, exponent, mantissa
      const U sign = val_as_integer >> SHIFT_FOR_SIGN;
      U exponent = (val_as_integer >> SHIFT_FOR_EXPO) & MASK_FOR_EXPO;
      U mantissa = val_as_integer & MASK_FOR_MANT;

      // exponent special cases

      if (exponent == ALL_ONES_EXPO) { // NaN, inf

        if (mantissa != 0) {
          mantissa = QUIET_NAN_MANT; // quiet and signaling -> quiet
        }
        exponent = USR_ALL_ONES_EXPO;

      } else { // normals

        if (exponent > USR_MAX_EXPO) { // round to inf
          exponent = USR_ALL_ONES_EXPO;
          mantissa = 0;
        } else if (exponent < USR_MIN_EXPO) { // round to 0
          exponent = 0;
          mantissa = 0;
        } else {
          exponent -= ADJUST_BIAS; // rebias to usr def bias
        }

      }

      // truncate mantissa
      mantissa = mantissa >> MANT_SHIFT;

      // pack the bits together
      const U packed = (sign << USR_SIGN_SHIFT) | (exponent << USR_BITS_MANT) | mantissa;

      // compute positions, shifts, masks
      const size_t bit_pos = idx * USR_BITS_TOTAL;
      const size_t arr_pos = bit_pos / TB;

      const unsigned int shift_lo = bit_pos % TB;
      const U mask_lo = ~(USR_BITS_MASK << shift_lo);

      data[arr_pos] &= mask_lo;
      data[arr_pos] |= (packed << shift_lo);

      const unsigned int shift_hi = TB - shift_lo;

      if (shift_hi < USR_BITS_TOTAL) 
      {
        const U mask_hi = ~(USR_BITS_MASK >> shift_hi);

        data[arr_pos + 1] &= mask_hi;
        data[arr_pos + 1] |= (packed >> shift_hi);
      }
    }

    HostFitFloatArray(size_t sizep) // constructor
    : given_size(sizep),
      is_copy(false)
    {
      const size_t rounded_size = (given_size + 31) / 32 * 32;
      const size_t bits_per_byte = 8;
      const size_t size_in_bytes = (USR_BITS_TOTAL * rounded_size + bits_per_byte - 1) / bits_per_byte;

      backing_size = (size_in_bytes + sizeof(U) - 1) / sizeof(U) + 1;

      data = new U [backing_size];

      #pragma omp parallel for
      for (size_t idx = 0; idx < backing_size; idx++) {
        data[idx] = 0;
      }
    }

    HostFitFloatArray(HostFitFloatArray& original) // copy constructor
    : given_size(original.given_size),
      backing_size(original.backing_size),
      data(original.data), is_copy(true) {}

    ~HostFitFloatArray() // destructor
    {
      if (!is_copy) {
        delete [] data;
      }
    }

    Proxy operator[](size_t idx)
    {
      return Proxy(idx, data);
    }

    size_t size() const noexcept { return given_size; }

    size_t actual_size() const noexcept { return backing_size; }

    U* data_pointer() const noexcept { return data; }
};


typedef FitFloatArray<float, BITS_FOR_EXPONENT_32, BITS_FOR_MANTISSA_32> FFArr32;
typedef FitFloatArray<double, BITS_FOR_EXPONENT_64, BITS_FOR_MANTISSA_64> FFArr64;
typedef HostFitFloatArray<float, BITS_FOR_EXPONENT_32, BITS_FOR_MANTISSA_32> Host_FFArr32;
typedef HostFitFloatArray<double, BITS_FOR_EXPONENT_64, BITS_FOR_MANTISSA_64> Host_FFArr64;


// *************************************
// ** Host to Device Helper Functions **
// *************************************


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


void FloatH2FFDmemcpy(FFArr32 dest, const float* src, const size_t num_elements) {
  Host_FFArr32 h_ff(num_elements);

  const unsigned int nt = omp_get_max_threads();
  const size_t elements_per_sub_block = 32;
  const size_t num_elements_rounded = (num_elements + elements_per_sub_block - 1) / elements_per_sub_block * elements_per_sub_block;
  const size_t elements_per_thread = (num_elements_rounded / nt + elements_per_sub_block - 1) / elements_per_sub_block * elements_per_sub_block;

  #pragma omp parallel for
  for (size_t s_idx = 0; s_idx <= num_elements_rounded; s_idx += elements_per_thread) {
    const size_t limit = std::min(num_elements, s_idx + elements_per_thread);
    for (size_t idx = s_idx; idx < limit; idx++) {
      h_ff.quick_write(idx, src[idx]);
    }
  }

  cudaMemcpy(dest.data_pointer(), h_ff.data_pointer(), sizeof(float) * h_ff.actual_size(), cudaMemcpyHostToHost);
  CheckCuda(__LINE__);
}

void FFD2FloatHmemcpy(float* dest, FFArr32 src, const size_t num_elements) {
  Host_FFArr32 h_ff(num_elements);

  cudaMemcpy(h_ff.data_pointer(), src.data_pointer(), sizeof(float) * h_ff.actual_size(), cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);

  #pragma omp parallel for
  for (size_t idx = 0; idx < num_elements; idx++) {
    dest[idx] = h_ff[idx];
  }
}

void DoubleH2FFDmemcpy(FFArr64 dest, const double* src, const size_t num_elements) {
  Host_FFArr64 h_ff(num_elements);

  const unsigned int nt = omp_get_max_threads();
  const size_t elements_per_sub_block = 64;
  const size_t num_elements_rounded = (num_elements + elements_per_sub_block - 1) / elements_per_sub_block * elements_per_sub_block;
  const size_t elements_per_thread = (num_elements_rounded / nt + elements_per_sub_block - 1) / elements_per_sub_block * elements_per_sub_block;

  #pragma omp parallel for
  for (size_t s_idx = 0; s_idx <= num_elements_rounded; s_idx += elements_per_thread) {
    const size_t limit = std::min(num_elements, s_idx + elements_per_thread);
    for (size_t idx = s_idx; idx < limit; idx++) {
      h_ff.quick_write(idx, src[idx]);
    }
  }

  cudaMemcpy(dest.data_pointer(), h_ff.data_pointer(), sizeof(double) * h_ff.actual_size(), cudaMemcpyHostToHost);
  CheckCuda(__LINE__);
}

void FFD2DoubleHmemcpy(double* dest, FFArr64 src, const size_t num_elements) {
  Host_FFArr64 h_ff(num_elements);

  cudaMemcpy(h_ff.data_pointer(), src.data_pointer(), sizeof(double) * h_ff.actual_size(), cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);

  #pragma omp parallel for
  for (size_t idx = 0; idx < num_elements; idx++) {
    dest[idx] = h_ff[idx];
  }
}
