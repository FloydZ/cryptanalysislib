#ifndef CRYPTANALYSISLIB_SIMD_H
#define CRYPTANALYSISLIB_SIMD_H

#include <cinttypes>
#include <cmath>
#include <cstdint>

#include "helper.h"
#include "algorithm/bits/popcount.h"
#include "print/print.h"
#include "random.h"

using cryptanalysislib::print_binary;
using namespace cryptanalysislib;

#define bit_shuffle_const(b0, b1, b2, b3, b4, b5, b6, b7) \
	((uint64_t(uint8_t(1 << b0)) << (7 * 8)) |            \
	 (uint64_t(uint8_t(1 << b1)) << (6 * 8)) |            \
	 (uint64_t(uint8_t(1 << b2)) << (5 * 8)) |            \
	 (uint64_t(uint8_t(1 << b3)) << (4 * 8)) |            \
	 (uint64_t(uint8_t(1 << b4)) << (3 * 8)) |            \
	 (uint64_t(uint8_t(1 << b5)) << (2 * 8)) |            \
	 (uint64_t(uint8_t(1 << b6)) << (1 * 8)) |            \
	 (uint64_t(uint8_t(1 << b7)) << (0 * 8)))

#if defined(USE_AVX2)

#include "simd/avx2.h"
#include "simd/float/avx2.h"
#if defined(USE_AVX512F)
#include "simd/avx512.h"
#endif

#elif defined(USE_ARM)
#include "simd/neon.h"
// #include "simd/float/neon.h"
#include "simd/float/simd.h"
#elif defined(USE_RISCV)

#include "simd/riscv.h"

#else

namespace cryptanalysislib {
    template<const bool __unsigned=true>
	struct _Xint8x16_t;
    template<const bool __unsigned=true>
	struct _Xint16x8_t;
    template<const bool __unsigned=true>
	struct _Xint32x4_t;
    template<const bool __unsigned=true>
	struct _Xint64x2_t;

    using _uint8x16_t = _Xint8x16_t<true>;
    using  _int8x16_t = _Xint8x16_t<false>;
    using _uint16x8_t = _Xint16x8_t<true>;
    using  _int16x8_t = _Xint16x8_t<false>;
    using _uint32x4_t = _Xint32x4_t<true>;
    using  _int32x4_t = _Xint32x4_t<false>;
    using _uint64x2_t = _Xint64x2_t<true>;
    using  _int64x2_t = _Xint64x2_t<false>;

    template<const bool __unsigned>
	struct _Xint8x16_t {
		constexpr static uint32_t LIMBS = 16;
		using limb_type = uint8_t;
		using S = _Xint8x16_t<__unsigned>;

        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

		union {
			// compatibility to `TxN_t`
			T8   d[16];
			T8  v8[16];
			T16 v16[8];
			T32 v32[4];
			T64 v64[2];
		};
		
	    [[nodiscard]] constexpr inline static size_t size() noexcept { 
            return LIMBS; 
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
            return __unsigned; 
        }

        constexpr inline _Xint8x16_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint8x16_t operator=(const _Xint32x4_t<> &b) noexcept;
		constexpr inline _Xint8x16_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint8x16_t() noexcept {}
		constexpr _Xint8x16_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint8x16_t(const _Xint32x4_t<> &b) noexcept;
		constexpr _Xint8x16_t(const _Xint64x2_t<> &b) noexcept;

        /// \param i[in]: position of the limb to return
        /// \return __m256i[i]
		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

        /// \param i[in]: position of the limb to return
        /// \return __m256i[i]
		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		/// \return
		static inline _Xint8x16_t random() noexcept {
			_Xint8x16_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

	    /// \param binary[in]:
	    /// \param hex[in]:
	    constexpr inline void print(bool binary = false,
	                                bool hex = false) const;

        
        /// \return [a, a, ..., a]
		[[nodiscard]] constexpr static inline _Xint8x16_t set1(const limb_type a) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret[i] = a;
			}
			return ret;
		}

        /// \return [a, b, ..., p]
		[[nodiscard]] constexpr static inline _Xint8x16_t set(
		        const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		        const limb_type e, const limb_type f, const limb_type g, const limb_type h,
		        const limb_type i, const limb_type j, const limb_type k, const limb_type l,
		        const limb_type m, const limb_type n, const limb_type o, const limb_type p) noexcept {
			_Xint8x16_t ret;
			ret.v8[0] = p;
			ret.v8[1] = o;
			ret.v8[2] = n;
			ret.v8[3] = m;
			ret.v8[4] = l;
			ret.v8[5] = k;
			ret.v8[6] = j;
			ret.v8[7] = i;
			ret.v8[8] = h;
			ret.v8[9] = g;
			ret.v8[10] = f;
			ret.v8[11] = e;
			ret.v8[12] = d;
			ret.v8[13] = c;
			ret.v8[14] = b;
			ret.v8[15] = a;
			return ret;
		}

        /// \return [p, o, ..., a]
		[[nodiscard]] constexpr static inline _Xint8x16_t setr(
		        const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		        const limb_type e, const limb_type f, const limb_type g, const limb_type h,
		        const limb_type i, const limb_type j, const limb_type k, const limb_type l,
		        const limb_type m, const limb_type n, const limb_type o, const limb_type p) noexcept {
			_Xint8x16_t ret;
			ret.v8[0] = a;
			ret.v8[1] = b;
			ret.v8[2] = c;
			ret.v8[3] = d;
			ret.v8[4] = e;
			ret.v8[5] = f;
			ret.v8[6] = g;
			ret.v8[7] = h;
			ret.v8[8] = i;
			ret.v8[9] = j;
			ret.v8[10] = k;
			ret.v8[11] = l;
			ret.v8[12] = m;
			ret.v8[13] = n;
			ret.v8[14] = o;
			ret.v8[15] = p;
			return ret;
		}

		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}

		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(limb_type *ptr,
                                           const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(limb_type *ptr,
                                                   const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}

		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(limb_type *ptr,
                                                     const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] ^ in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] & in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] | in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (~in1.d[i]) & in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = ~in1.d[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] + in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] - in2[i];
            }
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return 
	    [[nodiscard]] constexpr static inline S div(const S in1,
	                                                const limb_type in2) noexcept {
            S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out[i] = in1[i] / in2;
            }
            return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 << in2
	    [[nodiscard]] constexpr static inline S slli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] >> in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] << in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const uint8_t in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] >> in2) ^ (in1.d[i] << ((sizeof(limb_type)*8) - in2));
            }
		    return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> ((sizeof(limb_type)*8) - in2));
            }
	    	return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
		constexpr static inline uint32_t gt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] > in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		constexpr static inline uint32_t gt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] > in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 < in2 uncompressed
		constexpr static inline uint32_t lt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] < in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 < in2 compressed
		constexpr static inline uint32_t lt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] < in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 == in2 uncompressed
		constexpr static inline uint32_t cmp_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] == in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S &in1,
		                                                   const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] == in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = __builtin_popcountll(in.d[i]); 
            }
	    	return ret;
	    }

	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            for (uint32_t i = 1; i < LIMBS; i++) {
                if (in.d[0] != in.d[i]) {
                    return false;
                }
            }
	    	return true;
        }
        
        // just shuffle the 16 u8 elements 
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = in.d[LIMBS - i - 1];
            }
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
            uint32_t ret = 0;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret ^= in.d[i] >> (sizeof(limb_type)*8 - i - 1);
            }
	    	return 0;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const limb_type *ptr,
	    											   const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
	    	S ret;
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = *(p + data.d[i]*scale);
            }
	    	return ret;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param offset[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    constexpr static inline void scatter(const void *ptr,
	    									 const S offset,
	    									 const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                *(p + offset.d[i]*scale) = data.d[i];
            }
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::min(a.d[i], b.d[i]);
            }
            return c;
        }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::max(a.d[i], b.d[i]);
            }
            return c;
        }
	};

    template<const bool __unsigned>
	struct _Xint16x8_t {
		constexpr static uint32_t LIMBS = 8;
		using limb_type = uint16_t;
		using S = _Xint16x8_t;

        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

		union {
			// compatibility to `TxN_t`
			T16 d[8];

			T8 v8[16];
			T16 v16[8];
			T32 v32[4];
			T64 v64[2];
		};

	    
        [[nodiscard]] constexpr inline static size_t size() noexcept { 
            return LIMBS; 
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
            return __unsigned; 
        }

		constexpr inline _Xint16x8_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint16x8_t operator=(const _Xint32x4_t<> &b) noexcept;
		constexpr inline _Xint16x8_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint16x8_t() noexcept {}
		constexpr _Xint16x8_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint32x4_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint64x2_t<> &b) noexcept;

		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _Xint16x8_t random() noexcept {
			_Xint16x8_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}


		[[nodiscard]] constexpr static inline _Xint16x8_t set(
		        const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		        const limb_type e, const limb_type f, const limb_type g, const limb_type h) noexcept {
			_Xint16x8_t ret;
			ret.v16[0] = h;
			ret.v16[1] = g;
			ret.v16[2] = f;
			ret.v16[3] = e;
			ret.v16[4] = d;
			ret.v16[5] = c;
			ret.v16[6] = b;
			ret.v16[7] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint16x8_t setr(
		        const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		        const limb_type e, const limb_type f, const limb_type g, const limb_type h) noexcept {
			_Xint16x8_t ret;
			ret.v16[0] = a;
			ret.v16[1] = b;
			ret.v16[2] = c;
			ret.v16[3] = d;
			ret.v16[4] = e;
			ret.v16[5] = f;
			ret.v16[6] = g;
			ret.v16[7] = h;
			return ret;
		}

		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline S load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S aligned_load(const void *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const void *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}

		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(limb_type *ptr,
                                           const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(limb_type *ptr, 
                                                   const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}

		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(limb_type *ptr, 
                                                     const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}


	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] ^ in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] & in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] | in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (~in1.d[i]) & in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = ~in1.d[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] + in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] - in2[i];
            }
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return 
	    [[nodiscard]] constexpr static inline S div(const S in1,
	                                                const limb_type in2) noexcept {
            S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out[i] = in1[i] / in2;
            }
            return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 << in2
	    [[nodiscard]] constexpr static inline S slli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] >> in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] << in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const uint8_t in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] >> in2) ^ (in1.d[i] << ((sizeof(limb_type)*8) - in2));
            }
		    return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> ((sizeof(limb_type)*8) - in2));
            }
	    	return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
		constexpr static inline uint32_t gt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] > in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		constexpr static inline uint32_t gt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] > in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 < in2 uncompressed
		constexpr static inline uint32_t lt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] < in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 < in2 compressed
		constexpr static inline uint32_t lt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] < in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 == in2 uncompressed
		constexpr static inline uint32_t cmp_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] == in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S &in1,
		                                                   const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] == in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = __builtin_popcountll(in.d[i]); 
            }
	    	return ret;
	    }

	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            for (uint32_t i = 1; i < LIMBS; i++) {
                if (in.d[0] != in.d[i]) {
                    return false;
                }
            }
	    	return true;
        }
        
        // just shuffle the 16 u8 elements 
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = in.d[LIMBS - i - 1];
            }
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
            uint32_t ret = 0;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret ^= in.d[i] >> (sizeof(limb_type)*8 - i - 1);
            }
	    	return 0;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const limb_type *ptr,
	    											   const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
	    	S ret;
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = *(p + data.d[i]*scale);
            }
	    	return ret;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param offset[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    constexpr static inline void scatter(const void *ptr,
	    									 const S offset,
	    									 const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                *(p + offset.d[i]*scale) = data.d[i];
            }
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::min(a.d[i], b.d[i]);
            }
            return c;
        }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::max(a.d[i], b.d[i]);
            }
            return c;
        }
	};

    template<const bool __unsigned>
	struct _Xint32x4_t {
		constexpr static uint32_t LIMBS = 4;
		using limb_type = uint32_t;
		using S = _Xint32x4_t;

        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

		union {
			// compatibility to `TxN_t`
			T32   d[4];

			T8   v8[16];
			T16 v16[8];
			T32 v32[4];
			T64 v64[2];
		};
		
        [[nodiscard]] constexpr inline static size_t size() noexcept { 
            return LIMBS; 
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
            return __unsigned; 
        }

        constexpr inline _Xint32x4_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint32x4_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint32x4_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint32x4_t() noexcept {}
		constexpr _Xint32x4_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint32x4_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint32x4_t(const _Xint64x2_t<> &b) noexcept;

		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _Xint32x4_t random() noexcept {
			_Xint32x4_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint32x4_t set(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint32x4_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint32x4_t setr(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint32x4_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}

		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(limb_type *ptr,
                                           const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(limb_type *ptr,
                                                   const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}

		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(limb_type *ptr,
                                                     const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] ^ in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] & in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] | in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (~in1.d[i]) & in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = ~in1.d[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] + in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] - in2[i];
            }
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return 
	    [[nodiscard]] constexpr static inline S div(const S in1,
	                                                const limb_type in2) noexcept {
            S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out[i] = in1[i] / in2;
            }
            return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 << in2
	    [[nodiscard]] constexpr static inline S slli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] >> in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] << in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const uint8_t in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] >> in2) ^ (in1.d[i] << ((sizeof(limb_type)*8) - in2));
            }
		    return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> ((sizeof(limb_type)*8) - in2));
            }
	    	return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
		constexpr static inline uint32_t gt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] > in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		constexpr static inline uint32_t gt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] > in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 < in2 uncompressed
		constexpr static inline uint32_t lt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] < in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 < in2 compressed
		constexpr static inline uint32_t lt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] < in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 == in2 uncompressed
		constexpr static inline uint32_t cmp_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] == in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S &in1,
		                                                   const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] == in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = __builtin_popcountll(in.d[i]); 
            }
	    	return ret;
	    }

	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            for (uint32_t i = 1; i < LIMBS; i++) {
                if (in.d[0] != in.d[i]) {
                    return false;
                }
            }
	    	return true;
        }
        
        // just shuffle the 16 u8 elements 
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = in.d[LIMBS - i - 1];
            }
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
            uint32_t ret = 0;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret ^= in.d[i] >> (sizeof(limb_type)*8 - i - 1);
            }
	    	return 0;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const limb_type *ptr,
	    											   const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
	    	S ret;
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = *(p + data.d[i]*scale);
            }
	    	return ret;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param offset[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    constexpr static inline void scatter(const void *ptr,
	    									 const S offset,
	    									 const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                *(p + offset.d[i]*scale) = data.d[i];
            }
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::min(a.d[i], b.d[i]);
            }
            return c;
        }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::max(a.d[i], b.d[i]);
            }
            return c;
        }

	};

    template<const bool __unsigned>
	struct _Xint64x2_t {
		constexpr static uint32_t LIMBS = 2;
		using limb_type = uint64_t;
		using S = _Xint64x2_t;

        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;
		
        union {
			T64   d[2];

			T8   v8[16];
			T16 v16[8];
			T32 v32[4];
			T64 v64[2];
		};

        [[nodiscard]] constexpr inline static size_t size() noexcept { 
            return LIMBS; 
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
            return __unsigned; 
        }

		constexpr inline _Xint64x2_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint32x4_t<> &b) noexcept;

		constexpr _Xint64x2_t() noexcept {}
		constexpr _Xint64x2_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint32x4_t<> &b) noexcept;

        /// \param i[in]: position of the limb to return
        /// \return __m256i[i]
		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

        /// \param i[in]: position of the limb to return
        /// \return __m256i[i]
		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _Xint64x2_t random() noexcept {
			_Xint64x2_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint64x2_t set(
		        const limb_type a, const limb_type b) noexcept {
			_Xint64x2_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint64x2_t setr(
		        const limb_type a, const limb_type b) noexcept {
			_Xint64x2_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
			auto *ptrd = (limb_type *) ptr;
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out[i] = ptrd[i];
			}
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(limb_type *ptr,
                                           const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(limb_type *ptr,
                                                   const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(limb_type *ptr,
                                                     const S in) noexcept {
			auto *ptrd = (limb_type *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ptrd[i] = in[i];
			}
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] ^ in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] & in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] | in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (~in1.d[i]) & in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = ~in1.d[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] + in2[i];
            }
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] - in2[i];
            }
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2[i];
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] * in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return 
	    [[nodiscard]] constexpr static inline S div(const S in1,
	                                                const limb_type in2) noexcept {
            S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out[i] = in1[i] / in2;
            }
            return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 << in2
	    [[nodiscard]] constexpr static inline S slli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] >> in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = in1.d[i] << in2;
            }
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const uint8_t in2) noexcept {
		    S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] >> in2) ^ (in1.d[i] << ((sizeof(limb_type)*8) - in2));
            }
		    return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> ((sizeof(limb_type)*8) - in2));
            }
	    	return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
		constexpr static inline uint32_t gt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] > in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		constexpr static inline uint32_t gt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] > in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 < in2 uncompressed
		constexpr static inline uint32_t lt_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] < in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 < in2 compressed
		constexpr static inline uint32_t lt(const S &in1,
		                                    const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] < in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 == in2 uncompressed
		constexpr static inline uint32_t cmp_(const S &in1,
		                                     const S &in2) noexcept {
			S ret;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret.d[i] = in1.d[i] == in2.d[i];
			}
			return ret;
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S &in1,
		                                                   const S &in2) noexcept {
			uint32_t ret = 0;
			for (uint32_t i = 0; i < LIMBS; i++) {
				ret ^= (in1.d[i] == in2.d[i]) << i;
			}
			return ret;
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = __builtin_popcountll(in.d[i]); 
            }
	    	return ret;
	    }

	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            for (uint32_t i = 1; i < LIMBS; i++) {
                if (in.d[0] != in.d[i]) {
                    return false;
                }
            }
	    	return true;
        }
        
        // just shuffle the 16 u8 elements 
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = in.d[LIMBS - i - 1];
            }
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
            uint32_t ret = 0;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret ^= in.d[i] >> (sizeof(limb_type)*8 - i - 1);
            }
	    	return 0;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const limb_type *ptr,
	    											   const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
	    	S ret;
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                ret.d[i] = *(p + data.d[i]*scale);
            }
	    	return ret;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param offset[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    constexpr static inline void scatter(const void *ptr,
	    									 const S offset,
	    									 const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
            uint8_t *p = (uint8_t *)ptr;
            for (uint32_t i = 0; i < LIMBS; i++) {
                *(p + offset.d[i]*scale) = data.d[i];
            }
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::min(a.d[i], b.d[i]);
            }
            return c;
        }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
                                                    const S b) noexcept {
            S c;
            for (uint32_t i = 0; i < LIMBS; i++) {
                c.d[i] = std::max(a.d[i], b.d[i]);
            }
            return c;
        }
    };
};// namespace cryptanalysislib

using namespace cryptanalysislib;

template<const bool __unsigned=true>
struct Xint8x32_t {
	constexpr static uint32_t LIMBS = 32;
	using limb_type = uint8_t;
	using S = Xint8x32_t;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		T8  d  [32];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[ 8];
		T64 v64[ 4];
	};
	
    [[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }
	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
        return __unsigned; 
    }

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	static inline S random() noexcept {
		S ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = rng();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \param __q31
	/// \param __q30
	/// \param __q29
	/// \param __q28
	/// \param __q27
	/// \param __q26
	/// \param __q25
	/// \param __q24
	/// \param __q23
	/// \param __q22
	/// \param __q21
	/// \param __q20
	/// \param __q19
	/// \param __q18
	/// \param __q17
	/// \param __q16
	/// \param __q15
	/// \param __q14
	/// \param __q13
	/// \param __q12
	/// \param __q11
	/// \param __q10
	/// \param __q09
	/// \param __q08
	/// \param __q07
	/// \param __q06
	/// \param __q05
	/// \param __q04
	/// \param __q03
	/// \param __q02
	/// \param __q01
	/// \param __q00
	/// \return
	[[nodiscard]] constexpr static inline S setr(const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	                                             const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	                                             const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	                                             const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	                                             const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	                                             const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	                                             const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	                                             const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		S out;
		out.d[0] = __q31;
		out.d[1] = __q30;
		out.d[2] = __q29;
		out.d[3] = __q28;
		out.d[4] = __q27;
		out.d[5] = __q26;
		out.d[6] = __q25;
		out.d[7] = __q24;
		out.d[8] = __q23;
		out.d[9] = __q22;
		out.d[10] = __q21;
		out.d[11] = __q20;
		out.d[12] = __q19;
		out.d[13] = __q18;
		out.d[14] = __q17;
		out.d[15] = __q16;
		out.d[16] = __q15;
		out.d[17] = __q14;
		out.d[18] = __q13;
		out.d[19] = __q12;
		out.d[20] = __q11;
		out.d[21] = __q10;
		out.d[22] = __q09;
		out.d[23] = __q08;
		out.d[24] = __q07;
		out.d[25] = __q06;
		out.d[26] = __q05;
		out.d[27] = __q04;
		out.d[28] = __q03;
		out.d[29] = __q02;
		out.d[30] = __q01;
		out.d[31] = __q00;
		return out;
	}

	[[nodiscard]] constexpr static inline S set(const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	                                            const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	                                            const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	                                            const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	                                            const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	                                            const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	                                            const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	                                            const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		S out;
		out.d[31] = __q31;
		out.d[30] = __q30;
		out.d[29] = __q29;
		out.d[28] = __q28;
		out.d[27] = __q27;
		out.d[26] = __q26;
		out.d[25] = __q25;
		out.d[24] = __q24;
		out.d[23] = __q23;
		out.d[22] = __q22;
		out.d[21] = __q21;
		out.d[20] = __q20;
		out.d[19] = __q19;
		out.d[18] = __q18;
		out.d[17] = __q17;
		out.d[16] = __q16;
		out.d[15] = __q15;
		out.d[14] = __q14;
		out.d[13] = __q13;
		out.d[12] = __q12;
		out.d[11] = __q11;
		out.d[10] = __q10;
		out.d[9] = __q09;
		out.d[8] = __q08;
		out.d[7] = __q07;
		out.d[6] = __q06;
		out.d[5] = __q05;
		out.d[4] = __q04;
		out.d[3] = __q03;
		out.d[2] = __q02;
		out.d[1] = __q01;
		out.d[0] = __q00;
		return out;
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		S out;
		out = S::set(a, a, a, a, a, a, a, a,
		             a, a, a, a, a, a, a, a,
		             a, a, a, a, a, a, a, a,
		             a, a, a, a, a, a, a, a);
		return out;
	}

	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}

	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}

	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	static inline void store(limb_type *ptr,
                             const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	/// \param ptr
	/// \param in
	static inline void aligned_store(limb_type *ptr,
                                     const S in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	/// \param ptr
	/// \param in
	static inline void unaligned_store(limb_type *ptr,
                                       const S in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] ^ in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] & in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] | in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~(in1.d[i] & in2.d[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~in1.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] + in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] - in2.d[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] * in2.d[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out[i] = in1[i] / in2;
        }
        return out;
    }

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 8);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 8);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] >> in2;
		}
		return out;
	}


	constexpr static inline uint32_t gt(const S &in1,
	                                    const S &in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			ret ^= (in1.d[i] > in2.d[i]) << i;
		}
		return ret;
	}


	[[nodiscard]] constexpr static inline S gt_(const S in1,
											    const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] > in2.d[i]) * -1ull;
		}
		return ret;
	}

	constexpr static inline uint32_t lt(const S &in1,
	                                    const S &in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			ret ^= (in1.d[i] < in2.d[i]) << i;
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] < in2.d[i]) * -1ull;
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline S cmp_(const S in1,
											     const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = in1.d[i] == in2.d[i];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline int cmp(const S in1,
	                                              const S in2) noexcept {
		int ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] == in2.d[i]) << i;
		}

		return ret;
	}


	[[nodiscard]] constexpr static inline S popcnt(const S in) {
		S ret;

		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = cryptanalysislib::popcount::popcount(in.d[i]);
		}
		return ret;
	}

	/// \param in1 vector register
	/// \param in2 vector register
	/// \return true if all elements are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in1) noexcept {
		for (uint32_t i = 1; i < S::LIMBS; i++) {
			if (in1.d[i-1] != in1.d[i]) {
				return false;
			}
		}
		return true;
	}


	/// \param in1 vector register
	/// \param in2 vector register
	/// \return reverses the order of the 32 8bit e limbs. not the order within the limbs
	[[nodiscard]] constexpr static inline S reverse(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[31 - i] = in1.d[i];
		}

		return ret;
	}

	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[perm.d[i]] = in.d[i];
		}
        return ret;
    }

	[[nodiscard]] constexpr static inline uint32_t move(const S in1) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] >> 7u) << i;
		}

		return ret;
	}
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::min(a.d[i], b.d[i]);
		}

        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::max(a.d[i], b.d[i]);
		}

        return c;
    }
};

/// 
using uint8x32_t = Xint8x32_t<true>;
using  int8x32_t = Xint8x32_t<false>;

template<const bool __unsigned=true>
struct Xint16x16_t {
	constexpr static uint32_t LIMBS = 16;
	using limb_type = uint16_t;
	using S = Xint16x16_t;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		T16 d  [16];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[8];
		T64 v64[4];
	};

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
        return __unsigned; 
    }

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline S random() noexcept {
		S ret;
		for (uint32_t i = 0; i < LIMBS; i++) {
			ret.d[i] = rng();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline S setr(
	        const limb_type a0, const limb_type a1, const limb_type a2, const limb_type a3,
	        const limb_type a4, const limb_type a5, const limb_type a6, const limb_type a7,
	        const limb_type a8, const limb_type a9, const limb_type a10, const limb_type a11,
	        const limb_type a12, const limb_type a13, const limb_type a14, const limb_type a15) {
		S out;
		out.d[0] = a0;
		out.d[1] = a1;
		out.d[2] = a2;
		out.d[3] = a3;
		out.d[4] = a4;
		out.d[5] = a5;
		out.d[6] = a6;
		out.d[7] = a7;
		out.d[8] = a8;
		out.d[9] = a9;
		out.d[10] = a10;
		out.d[11] = a11;
		out.d[12] = a12;
		out.d[13] = a13;
		out.d[14] = a14;
		out.d[15] = a15;
		return out;
	}

	[[nodiscard]] constexpr static inline S set(
	        const limb_type a0, const limb_type a1, const limb_type a2, const limb_type a3,
	        const limb_type a4, const limb_type a5, const limb_type a6, const limb_type a7,
	        const limb_type a8, const limb_type a9, const limb_type a10, const limb_type a11,
	        const limb_type a12, const limb_type a13, const limb_type a14, const limb_type a15) noexcept {
		return S::setr(a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0);
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		S out;
		out = S::set(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a);
		return out;
	}

	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}

	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(limb_type *ptr,
                                       const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	/// \param ptr
	/// \param in
	static inline void aligned_store(limb_type *ptr, 
                                     const S in) noexcept {
		auto *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.d[i];
		}
	}

	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const S in) noexcept {
		auto *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.d[i];
		}
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] ^ in2.d[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] & in2.d[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] | in2.d[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~(in1.d[i] & in2.d[i]);
		}
		return out;
	}

	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~in1.d[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] + in2.d[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] - in2.d[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] * in2.d[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const uint8_t in2) noexcept {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out[i] = in1[i] / in2;
        }
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 16);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] << in2;
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 8);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] >> in2;
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> (sizeof(limb_type) * 8 - in2));
        }
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> (sizeof(limb_type) * 8 - in2));
        }
		return out;
	}

	[[nodiscard]] constexpr static inline S gt_(const S in1,
	                                            const S in2) noexcept{
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] > in2.d[i]) * -1ull;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] > in2.d[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept{
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] < in2.d[i]) * -1ull;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] < in2.d[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
	                                                   const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] == in2.d[i]) << i;
		}

		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
                                                 const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = in1.d[i] == in2.d[i];
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret;

		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = cryptanalysislib::popcount::popcount(in.d[i]);
		}
		return ret;
	}

	/// \param in1 vector register
	/// \param in2 vector register
	/// \return true if all elements are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in1) noexcept {
		for (uint32_t i = 1; i < S::LIMBS; i++) {
			if (in1.d[i-1] != in1.d[i]) {
				return false;
			}
		}
		return true;
	}

	/// \param in1 vector register
	/// \param in2 vector register
	/// \return reverses the order of the 16 16bit e limbs. not the order within the limbs
	[[nodiscard]] constexpr static inline S reverse(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[S::LIMBS - 1 - i] = in1.d[i];
		}

		return ret;
	}
	
    /// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[perm.d[i]] = in.d[i];
		}
        return ret;
    }
	
	/// extracts the sign bit of each limb
	[[nodiscard]] constexpr static inline limb_type move(const S in1) noexcept {
		uint16_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] >> 15u) << i;
		}

		return ret;
	}

    /// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::min(a.d[i], b.d[i]);
		}

        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::max(a.d[i], b.d[i]);
		}

        return c;
    }
};

/// 
using uint16x16_t = Xint16x16_t<true>;
using  int16x16_t = Xint16x16_t<false>;

template<const bool __unsigned=true>
struct Xint32x8_t {
	constexpr static uint32_t LIMBS = 8;
	using limb_type = uint32_t;
	using S = Xint32x8_t;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		T32 d  [ 8];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[ 8];
		T64 v64[ 4];
	};
	
    [[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }
	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
        return __unsigned; 
    }

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline S random() noexcept {
		S ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = rng();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline S setr(
	        const limb_type a0, const limb_type a1, const limb_type a2, const limb_type a3,
	        const limb_type a4, const limb_type a5, const limb_type a6, const limb_type a7) noexcept {
		S out;
		out.d[0] = a0;
		out.d[1] = a1;
		out.d[2] = a2;
		out.d[3] = a3;
		out.d[4] = a4;
		out.d[5] = a5;
		out.d[6] = a6;
		out.d[7] = a7;
		return out;
	}

	[[nodiscard]] constexpr static inline S set(
	        const limb_type a0, const limb_type a1, const limb_type a2, const limb_type a3,
	        const limb_type a4, const limb_type a5, const limb_type a6, const limb_type a7) noexcept {
		return S::setr(a7, a6, a5, a4, a3, a2, a1, a0);
	}

	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) {
		S out;
		out = S::set(a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(limb_type *ptr,
                                       const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(limb_type *ptr,
                                               const S in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const S in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] ^ in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] & in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] | in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~(in1.d[i] & in2.d[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~in1.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(
	        const S in1,
	        const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] + in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] - in2.d[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] * in2.d[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out[i] = in1[i] / in2;
        }
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const uint32_t in2) noexcept {
		assert(in2 <= 32);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const uint16_t in2) noexcept {
		assert(in2 <= 8);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] >> in2;
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> (sizeof(limb_type) * 8 - in2));
        }
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> (sizeof(limb_type) * 8 - in2));
        }
		return out;
	}

	[[nodiscard]] constexpr static inline S gt_(const S in1,
												const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] > in2.d[i]) * -1ull;
		}

		return ret;
	}


	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
													  const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] > in2.d[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                                     const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] < in2.d[i]) * -1ull;
		}

		return ret;
	}


	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] < in2.d[i]) << i;
		}

		return ret;
	}
	
    [[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
													   const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] == in2.d[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                             const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = in1.d[i] == in2.d[i];
		}

		return ret;
	}

    ///
	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret;

		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = cryptanalysislib::popcount::popcount(in.d[i]);
		}
		return ret;
	}

	/// \param in1 vector register
	/// \param in2 vector register
	/// \return true if all elements are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in1) noexcept {
		for (uint32_t i = 1; i < S::LIMBS; i++) {
			if (in1.d[i-1] != in1.d[i]) {
				return false;
			}
		}
		return true;
	}

	/// \param in1 vector register
	/// \param in2 vector register
	/// \return reverses the order of the 8 32bit e limbs. not the order within the limbs
	[[nodiscard]] constexpr static inline S reverse(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[S::LIMBS - 1 - i] = in1.d[i];
		}

		return ret;
	}
	
    ///
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = in.d[perm.d[i] & 0x7];
		}
		return ret;
	}


	[[nodiscard]] static inline uint32_t move(const S in1) noexcept {
		uint8_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] >> 31) << i;
		}

		return ret;
	}


	/// input:
	/// 	mask: 0b010101010
	/// output: a permutation mask s.t, applied on in =  [ x0, x1, x2, x3, x4, x5, x6, x7 ],
	/// 			S::permute(in, permutation_mask) will result int
	///  	[x1, x3, x5, x7, 0, 0, 0, 0]
	[[nodiscard]] static inline S pack(uint32_t mask) noexcept {
		S ret = S::set1(0);
		for (uint32_t i = 0; (i < 8) && (mask != 0); i++ ) {
			const uint32_t pos = __builtin_ctz(mask);
			ret[i] = pos;
			mask ^= 1u << pos;
		}
		return ret;
	}


	[[nodiscard]] static inline S cvtepu8(const _uint8x16_t in) noexcept {
		S ret;
		for (uint32_t i = 0; i < 16; i++) {
			ret.d[i] = in.d[i];
		}

		return ret;
	}

    /// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::min(a.d[i], b.d[i]);
		}

        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::max(a.d[i], b.d[i]);
		}

        return c;
    }

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 4>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
	                                               const S data) noexcept {
		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = *(uint32_t *) (ptr8 + (data.d[i]*scale));
		}

		return ret;
	}
};

///
using uint32x8_t = Xint32x8_t<true>;
using  int32x8_t = Xint32x8_t<false>;


template<const bool __unsigned=true>
struct Xint64x4_t {
	constexpr static uint32_t LIMBS = 4;
	using limb_type = uint64_t;
	using S = Xint64x4_t;
    
    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		T64  d [ 4];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[ 8];
		T64 v64[ 4];
	};

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
        return __unsigned; 
    }

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline S random() noexcept {
		S ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.d[i] = rng();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline S setr(
	        const limb_type a0, const limb_type a1,
	        const limb_type a2, const limb_type a3) noexcept {
		S out;
		out.d[0] = a0;
		out.d[1] = a1;
		out.d[2] = a2;
		out.d[3] = a3;
		return out;
	}

	[[nodiscard]] constexpr static inline S set(
	        const limb_type a0, const limb_type a1,
	        const limb_type a2, const limb_type a3) noexcept {
		return S::setr(a3, a2, a1, a0);
	}

	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		S out;
		out = S::set(a, a, a, a);
		return out;
	}

	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ptr[i];
		}
		return out;
	}

	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(limb_type *ptr,
                                       const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(limb_type *ptr,
                                               const S in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.d[i];
		}
	}

	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const S in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.d[i];
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] ^ in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] & in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] | in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out{};
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~(in1.d[i] & in2.d[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = ~in1.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] + in2.d[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] - in2.d[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] * in2.d[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}
	
    /// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out[i] = in1[i] / in2;
        }
        return out;
    }

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 64);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 64);
		S out;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			out.d[i] = in1.d[i] >> in2;
		}
		return out;
	
    }
	
    /// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> (sizeof(limb_type) * 8 - in2));
        }
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out.d[i] = (in1.d[i] << in2) ^ (in1.d[i] >> (sizeof(limb_type) * 8 - in2));
        }
		return out;
	}


	[[nodiscard]] constexpr static inline S gt_(const S in1,
											    const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] > in2.d[i]) * -1ull;
		}

		return ret;
	}


	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
													  const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] > in2.d[i]) << i;
		}

		return ret;
	}


	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = (in1.d[i] < in2.d[i]) * -1ull;
		}

		return ret;
	}


    ///
	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] < in2.d[i]) << i;
		}

		return ret;
	}
	
    [[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
													   const S in2) noexcept {
		uint32_t ret = 0;
		for (uint8_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] == in2.d[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                             const S in2) noexcept {
		S ret;
		for (uint8_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = in1.d[i] == in2.d[i];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret;

		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = cryptanalysislib::popcount::popcount(in.d[i]);
		}
		return ret;
	}

	/// \param in1 vector register
	/// \param in2 vector register
	/// \return true if all elements are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in1) noexcept {
		for (uint32_t i = 1; i < S::LIMBS; i++) {
			if (in1.d[i-1] != in1.d[i]) {
				return false;
			}
		}
		return true;
	}

	/// \param in1 vector register
	/// \param in2 vector register
	/// \return reverses the order of the 4 64bit e limbs. not the order within the limbs
	[[nodiscard]] constexpr static inline S reverse(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[S::LIMBS - 1 - i] = in1.d[i];
		}

		return ret;
	}

	///
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = in.d[perm.d[i]];
		}
		return ret;
	}

	///
	/// \tparam in2
	/// \param in1
	/// \return
	template<const uint32_t in2>
	[[nodiscard]] constexpr static inline S permute(const S in1) noexcept {
		S ret;

		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = in1.d[(in2 >> (2 * i)) & 0b11];
		}
		return ret;
	}

	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint8_t move(const S in1) noexcept {
		uint8_t ret = 0;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= (in1.d[i] >> 63u) << i;
		}

		return ret;
	}
    
    /// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::min(a.d[i], b.d[i]);
		}

        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::max(a.d[i], b.d[i]);
		}

        return c;
    }

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
	                                               const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);

		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = *(uint64_t *) (ptr8 + data.d[i] * scale);
		}

		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
	                                               const cryptanalysislib::_uint32x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = *(uint64_t *) (ptr8 + data.v32[i] * scale);
		}
		return ret;
	}

};

///
using uint64x4_t = Xint64x4_t<true>;
using  int64x4_t = Xint64x4_t<false>;


#include "simd/float/simd.h"
#endif// no SIMD unit available


template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator*(const _Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::mullo(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator*(const _Xint8x16_t<_unsigned> &lhs, const uint8_t &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::mullo(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator*(const uint8_t &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::mullo(rhs, lhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator+(const _Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::add(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator-(const _Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::sub(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator&(const _Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::and_(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator^(const _Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::xor_(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator|(const _Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	return _Xint8x16_t<_unsigned>::or_(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator~(const _Xint8x16_t<_unsigned> &lhs) noexcept {
	return _Xint8x16_t<_unsigned>::not_(lhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator>>(const _Xint8x16_t<_unsigned> &lhs, const uint32_t rhs) noexcept {
	return _Xint8x16_t<_unsigned>::srli(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator<<(const _Xint8x16_t<_unsigned> &lhs, const uint32_t rhs) noexcept {
	return _Xint8x16_t<_unsigned>::slli(lhs, rhs);
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator^=(_Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	lhs = _Xint8x16_t<_unsigned>::xor_(lhs, rhs);
	return lhs;
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator&=(_Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	lhs = _Xint8x16_t<_unsigned>::and_(lhs, rhs);
	return lhs;
}
template<const bool _unsigned>
constexpr inline _Xint8x16_t<_unsigned> operator|=(_Xint8x16_t<_unsigned> &lhs, const _Xint8x16_t<_unsigned> &rhs) noexcept {
	lhs = _Xint8x16_t<_unsigned>::or_(lhs, rhs);
	return lhs;
}

///
constexpr inline uint8x32_t operator*(const uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	return uint8x32_t::mullo(lhs, rhs);
}
constexpr inline uint8x32_t operator*(const uint8x32_t &lhs, const uint8_t &rhs) noexcept {
	return uint8x32_t::mullo(lhs, rhs);
}
constexpr inline uint8x32_t operator*(const uint8_t &lhs, const uint8x32_t &rhs) noexcept {
	return uint8x32_t::mullo(rhs, lhs);
}
constexpr inline uint8x32_t operator+(const uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	return uint8x32_t::add(lhs, rhs);
}
constexpr inline uint8x32_t operator-(const uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	return uint8x32_t::sub(lhs, rhs);
}
constexpr inline uint8x32_t operator&(const uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	return uint8x32_t::and_(lhs, rhs);
}
constexpr inline uint8x32_t operator^(const uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	return uint8x32_t::xor_(lhs, rhs);
}
constexpr inline uint8x32_t operator|(const uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	return uint8x32_t::or_(lhs, rhs);
}
constexpr inline uint8x32_t operator~(const uint8x32_t &lhs) noexcept {
	return uint8x32_t::not_(lhs);
}
constexpr inline uint8x32_t operator>>(const uint8x32_t &lhs, const uint32_t rhs) noexcept {
	return uint8x32_t::srli(lhs, rhs);
}
constexpr inline uint8x32_t operator<<(const uint8x32_t &lhs, const uint32_t rhs) noexcept {
	return uint8x32_t::slli(lhs, rhs);
}
constexpr inline uint8x32_t operator^=(uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	lhs = uint8x32_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint8x32_t operator&=(uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	lhs = uint8x32_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint8x32_t operator|=(uint8x32_t &lhs, const uint8x32_t &rhs) noexcept {
	lhs = uint8x32_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint16x16_t operator*(const uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	return uint16x16_t::mullo(lhs, rhs);
}
constexpr inline uint16x16_t operator*(const uint16x16_t &lhs, const uint8_t &rhs) noexcept {
	return uint16x16_t::mullo(lhs, rhs);
}
constexpr inline uint16x16_t operator*(const uint8_t &lhs, const uint16x16_t &rhs) noexcept {
	return uint16x16_t::mullo(rhs, lhs);
}
constexpr inline uint16x16_t operator+(const uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	return uint16x16_t::add(lhs, rhs);
}
constexpr inline uint16x16_t operator-(const uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	return uint16x16_t::sub(lhs, rhs);
}
constexpr inline uint16x16_t operator&(const uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	return uint16x16_t::and_(lhs, rhs);
}
constexpr inline uint16x16_t operator^(const uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	return uint16x16_t::xor_(lhs, rhs);
}
constexpr inline uint16x16_t operator|(const uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	return uint16x16_t::or_(lhs, rhs);
}
constexpr inline uint16x16_t operator~(const uint16x16_t &lhs) noexcept {
	return uint16x16_t::not_(lhs);
}
constexpr inline uint16x16_t operator>>(const uint16x16_t &lhs, const uint32_t rhs) noexcept {
	return uint16x16_t::srli(lhs, rhs);
}
constexpr inline uint16x16_t operator<<(const uint16x16_t &lhs, const uint32_t rhs) noexcept {
	return uint16x16_t::slli(lhs, rhs);
}
constexpr inline uint16x16_t operator^=(uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	lhs = uint16x16_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint16x16_t operator&=(uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	lhs = uint16x16_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint16x16_t operator|=(uint16x16_t &lhs, const uint16x16_t &rhs) noexcept {
	lhs = uint16x16_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint32x8_t operator*(const uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	return uint32x8_t::mullo(lhs, rhs);
}
constexpr inline uint32x8_t operator*(const uint32x8_t &lhs, const uint8_t &rhs) noexcept {
	return uint32x8_t::mullo(lhs, rhs);
}
constexpr inline uint32x8_t operator*(const uint8_t &lhs, const uint32x8_t &rhs) noexcept {
	return uint32x8_t::mullo(rhs, lhs);
}
constexpr inline uint32x8_t operator+(const uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	return uint32x8_t::add(lhs, rhs);
}
constexpr inline uint32x8_t operator-(const uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	return uint32x8_t::sub(lhs, rhs);
}
constexpr inline uint32x8_t operator&(const uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	return uint32x8_t::and_(lhs, rhs);
}
constexpr inline uint32x8_t operator^(const uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	return uint32x8_t::xor_(lhs, rhs);
}
constexpr inline uint32x8_t operator|(const uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	return uint32x8_t::or_(lhs, rhs);
}
constexpr inline uint32x8_t operator~(const uint32x8_t &lhs) noexcept {
	return uint32x8_t::not_(lhs);
}
constexpr inline uint32x8_t operator>>(const uint32x8_t &lhs, const uint32_t rhs) noexcept {
	return uint32x8_t::srli(lhs, rhs);
}
constexpr inline uint32x8_t operator<<(const uint32x8_t &lhs, const uint32_t rhs) noexcept {
	return uint32x8_t::slli(lhs, rhs);
}
constexpr inline uint32x8_t operator^=(uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	lhs = uint32x8_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint32x8_t operator&=(uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	lhs = uint32x8_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint32x8_t operator|=(uint32x8_t &lhs, const uint32x8_t &rhs) noexcept {
	lhs = uint32x8_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint64x4_t operator*(const uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	return uint64x4_t::mullo(lhs, rhs);
}
constexpr inline uint64x4_t operator*(const uint64x4_t &lhs, const uint64_t &rhs) noexcept {
	return uint64x4_t::mullo(lhs, rhs);
}
constexpr inline uint64x4_t operator*(const uint8_t &lhs, const uint64x4_t &rhs) noexcept {
	return uint64x4_t::mullo(rhs, lhs);
}
constexpr inline uint64x4_t operator+(const uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	return uint64x4_t::add(lhs, rhs);
}
constexpr inline uint64x4_t operator-(const uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	return uint64x4_t::sub(lhs, rhs);
}
constexpr inline uint64x4_t operator&(const uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	return uint64x4_t::and_(lhs, rhs);
}
constexpr inline uint64x4_t operator^(const uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	return uint64x4_t::xor_(lhs, rhs);
}
constexpr inline uint64x4_t operator|(const uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	return uint64x4_t::or_(lhs, rhs);
}
constexpr inline uint64x4_t operator~(const uint64x4_t &lhs) noexcept {
	return uint64x4_t::not_(lhs);
}
constexpr inline uint64x4_t operator>>(const uint64x4_t &lhs, const uint32_t rhs) noexcept {
	return uint64x4_t::srli(lhs, rhs);
}
constexpr inline uint64x4_t operator<<(const uint64x4_t &lhs, const uint32_t rhs) noexcept {
	return uint64x4_t::slli(lhs, rhs);
}
constexpr inline uint64x4_t operator^=(uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	lhs = uint64x4_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint64x4_t operator&=(uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	lhs = uint64x4_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint64x4_t operator|=(uint64x4_t &lhs, const uint64x4_t &rhs) noexcept {
	lhs = uint64x4_t::or_(lhs, rhs);
	return lhs;
}


/* 					 comparison									*/
constexpr inline int operator==(const uint8x32_t &a, const uint8x32_t &b) noexcept {
	return uint8x32_t::cmp(a, b);
}
constexpr inline int operator!=(const uint8x32_t &a, const uint8x32_t &b) noexcept {
	return 0xffffffff ^ uint8x32_t::cmp(a, b);
}
constexpr inline int operator<(const uint8x32_t &a, const uint8x32_t &b) noexcept {
	return uint8x32_t::gt(b, a);
}
constexpr inline int operator>(const uint8x32_t &a, const uint8x32_t &b) noexcept {
	return uint8x32_t::gt(a, b);
}


///
constexpr inline int operator==(const uint16x16_t &a, const uint16x16_t &b) noexcept {
	return (int) uint16x16_t::cmp(a, b);
}
constexpr inline int operator!=(const uint16x16_t &a, const uint16x16_t &b) noexcept {
	return 0xffff ^ uint16x16_t::cmp(a, b);
}
constexpr inline int operator<(const uint16x16_t &a, const uint16x16_t &b) noexcept {
	return (int) uint16x16_t::gt(b, a);
}
constexpr inline int operator>(const uint16x16_t &a, const uint16x16_t &b) noexcept {
	return (int) uint16x16_t::gt(a, b);
}


///
constexpr inline uint32_t operator==(const uint32x8_t &a, const uint32x8_t &b) noexcept {
	return (int) uint32x8_t::cmp(a, b);
}
constexpr inline int operator!=(const uint32x8_t &a, const uint32x8_t &b) noexcept {
	return 0xff ^ uint32x8_t::cmp(a, b);
}
constexpr inline int operator<(const uint32x8_t &a, const uint32x8_t &b) noexcept {
	return (int) uint32x8_t::gt(b, a);
}
constexpr inline int operator>(const uint32x8_t &a, const uint32x8_t &b) noexcept {
	return (int) uint32x8_t::gt(a, b);
}

///
constexpr inline int operator==(const uint64x4_t &a, const uint64x4_t &b) noexcept {
	return (int) uint64x4_t::cmp(a, b);
}
constexpr inline int operator!=(const uint64x4_t &a, const uint64x4_t &b) noexcept {
	return 0xf ^ uint64x4_t::cmp(a, b);
}
constexpr inline int operator<(const uint64x4_t &a, const uint64x4_t &b) noexcept {
	return (int) uint64x4_t::gt(b, a);
}
constexpr inline int operator>(const uint64x4_t &a, const uint64x4_t &b) {
	return (int) uint64x4_t::gt(a, b);
}

/* sub types */
constexpr inline uint32_t operator==(const cryptanalysislib::_uint8x16_t &a, const cryptanalysislib::_uint8x16_t &b) noexcept {
	return cryptanalysislib::_uint8x16_t::cmp(a, b);
}
constexpr inline uint32_t operator!=(const cryptanalysislib::_uint8x16_t &a, const cryptanalysislib::_uint8x16_t &b) noexcept {
	return 0xffffffff ^ cryptanalysislib::_uint8x16_t::cmp(a, b);
}
constexpr inline uint32_t operator<(const cryptanalysislib::_uint8x16_t &a, const cryptanalysislib::_uint8x16_t &b) noexcept {
	return cryptanalysislib::_uint8x16_t::gt(b, a);
}
constexpr inline int operator>(const cryptanalysislib::_uint8x16_t &a, const cryptanalysislib::_uint8x16_t &b) noexcept {
	return cryptanalysislib::_uint8x16_t::gt(a, b);
}

////////////////////////////////////////////////////////////


/// functions which are shared among all implementations.
template<const bool __unsigned>
constexpr inline void Xint8x32_t<__unsigned>::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	assert(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 32; i++) {
			print_binary(this->v8[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 32; i++) {
			printf("%hhx ", this->v8[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 32; i++) {
		printf("%u ", this->v8[i]);
	}
	printf("\n");
}

template<const bool __unsigned>
constexpr inline void Xint16x16_t<__unsigned>::print(bool binary,
                                                     bool hex) const {
	/// make sure that only one is defined
	assert(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 16; i++) {
			print_binary(this->v16[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 16; i++) {
			printf("%hx ", this->v16[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 16; i++) {
		printf("%u ", this->v16[i]);
	}
	printf("\n");
}

template<const bool __unsigned>
constexpr inline void Xint32x8_t<__unsigned>::print(bool binary,
                                                    bool hex) const {
	/// make sure that only one is defined
	assert(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 8; i++) {
			print_binary(this->v32[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 8; i++) {
			printf("%x ", this->v32[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 8; i++) {
		printf("%u ", this->v32[i]);
	}
	printf("\n");
}

template<const bool __unsigned>
constexpr inline void Xint64x4_t<__unsigned>::print(bool binary, 
                                                    bool hex) const {
	/// make sure that only one is defined
	assert(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 4; i++) {
			print_binary(this->v64[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 4; i++) {
			printf("%" PRIu64 " ", this->v64[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 4; i++) {
		printf("%" PRIu64 " ", this->v64[i]);
	}
	printf("\n");
}

////////////////////////////////////////////////////////////////////////

namespace cryptanalysislib {
    template<>
	constexpr inline _uint8x16_t _uint8x16_t::operator=(const _uint16x8_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}

		return *this;
	}
    template<>
	constexpr inline _uint8x16_t _uint8x16_t::operator=(const _uint32x4_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}

		return *this;
	}
    template<>
	constexpr inline _uint8x16_t _uint8x16_t::operator=(const _uint64x2_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}

		return *this;
	}

    template<>
	constexpr inline _uint16x8_t _uint16x8_t::operator=(const _uint8x16_t &b) noexcept {
		_uint16x8_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}
    template<>
	constexpr inline _uint16x8_t _uint16x8_t::operator=(const _uint32x4_t &b) noexcept {
		_uint16x8_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}
    template<>
	constexpr inline _uint16x8_t _uint16x8_t::operator=(const _uint64x2_t &b) noexcept {
		_uint16x8_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}

    template<>
	constexpr inline _uint32x4_t _uint32x4_t::operator=(const _uint16x8_t &b) noexcept {
		_uint32x4_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}
    template<>
	constexpr inline _uint32x4_t _uint32x4_t::operator=(const _uint8x16_t &b) noexcept {
		_uint32x4_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}
    template<>
	constexpr inline _uint32x4_t _uint32x4_t::operator=(const _uint64x2_t &b) noexcept {
		_uint32x4_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}

    template<>
	constexpr inline _uint64x2_t _uint64x2_t::operator=(const _uint16x8_t &b) noexcept {
		_uint64x2_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}
    template<>
	constexpr inline _uint64x2_t _uint64x2_t::operator=(const _uint32x4_t &b) noexcept {
		_uint64x2_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}
    template<>
	constexpr inline _uint64x2_t _uint64x2_t::operator=(const _uint8x16_t &b) noexcept {
		_uint64x2_t ret;
		for (uint32_t i = 0; i < 2; ++i) {
			ret.v64[i] = b.v64[i];
		}

		return ret;
	}
}


// TODO signed type
namespace cryptanalysislib {
    template<>
	constexpr _uint8x16_t::_Xint8x16_t(const _uint16x8_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint8x16_t::_Xint8x16_t(const _uint32x4_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint8x16_t::_Xint8x16_t(const _uint64x2_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}

    template<>
	constexpr _uint16x8_t::_Xint16x8_t(const _uint8x16_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint16x8_t::_Xint16x8_t(const _uint32x4_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint16x8_t::_Xint16x8_t(const _uint64x2_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}

    template<>
	constexpr _uint32x4_t::_Xint32x4_t(const _uint16x8_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint32x4_t::_Xint32x4_t(const _uint8x16_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint32x4_t::_Xint32x4_t(const _uint64x2_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}

    template<>
	constexpr _uint64x2_t::_Xint64x2_t(const _uint16x8_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint64x2_t::_Xint64x2_t(const _uint32x4_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
    template<>
	constexpr _uint64x2_t::_Xint64x2_t(const _uint8x16_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
}// namespace cryptanalysislib


#if __cplusplus > 201709L
/// describes the needed function to be row
template<class S>
concept SIMDAble = requires(S s) {
	typename S::limb_type;
	typename S::S;

	S::LIMBS;

	requires requires (
	        const bool b,
	        const uint32_t u32,
	        typename S::limb_type l,
	        typename S::limb_type *pl) {
		{ S::is_unsigned() } -> std::convertible_to<bool>;

		{ S::random() } -> std::convertible_to<S>;
		{ S::set(pl) } -> std::convertible_to<S>;
		{ S::setr(pl) } -> std::convertible_to<S>;
		{ S::set1(l) } -> std::convertible_to<S>;
		{ S::load(pl) } -> std::convertible_to<S>;
		{ S::aligned_load(pl) } -> std::convertible_to<S>;
		{ S::unaligned_load(pl) } -> std::convertible_to<S>;

		{ S::xor_(s, s) } -> std::convertible_to<S>;
		{ S::and_(s, s) } -> std::convertible_to<S>;
		{ S::or_(s, s) } -> std::convertible_to<S>;
		{ S::andnot(s, s) } -> std::convertible_to<S>;
		{ S::not_(s) } -> std::convertible_to<S>;
		{ S::add(s, s) } -> std::convertible_to<S>;
		{ S::sub(s, s) } -> std::convertible_to<S>;
		{ S::mullo(s, s) } -> std::convertible_to<S>;
		{ S::mulhi(s, s) } -> std::convertible_to<S>;
		{ S::mul(s, s) } -> std::convertible_to<S>;
		{ S::slli(s, l) } -> std::convertible_to<S>;
		{ S::srli(s, l) } -> std::convertible_to<S>;
		{ S::ror(s, l) } -> std::convertible_to<S>;
		{ S::rol(s, l) } -> std::convertible_to<S>;

		{ S::gt_(s, s) } -> std::convertible_to<S>;
		{ S::gt(s, s) } -> std::convertible_to<typename S::limb_type>;
		{ S::lt_(s, s) } -> std::convertible_to<S>;
		{ S::lt(s, s) } -> std::convertible_to<typename S::limb_type>;
		//TODO { S::eq_(s, s) } -> std::convertible_to<S>;
		//TODO { S::eq(s, s) } -> std::convertible_to<typename S::limb_type>;
		{ S::cmp_(s, s) } -> std::convertible_to<S>;
		{ S::cmp(s, s) } -> std::convertible_to<typename S::limb_type>;

		{ S::popcnt(s) } -> std::convertible_to<S>;
		{ S::all_equal(s) } -> std::convertible_to<bool>;
		{ S::reverse(s) } -> std::convertible_to<S>;

		{ S::gather(pl, s) } -> std::convertible_to<S>;
		S::scatter(pl, s, s);
		{ S::permute(s, s) } -> std::convertible_to<S>;
		{ S::move(s) } -> std::convertible_to<typename S::limb_type>;
		// { S::mask(u32) } -> std::convertible_to<S>;

		s.print(b, b);
		// needed for sorting
		s.size();
	};

};
#endif

#include "simd/bits/bits.h"
#include "simd/generic.h"
#include "simd/bits/generic.h"

template<typename T>
#ifdef USE_AVX512F
using SIMDSelector = TxN_t<T, 64/sizeof(T), std::is_unsigned_v<T>>;
#else
using SIMDSelector = TxN_t<T, 32/sizeof(T), std::is_unsigned_v<T>>;
#endif

#endif//CRYPTANALYSISLIB_SIMD_H
