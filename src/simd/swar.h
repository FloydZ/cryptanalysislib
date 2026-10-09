#pragma once
#include <stdint.h>

#include "helper.h"

// based on: https://github.com/qlibs/swar/blob/main/swar
/// TODO copy tests from  https://github.com/qlibs/swar/blob/main/swar and add them

namespace cryptanalysislib::swar {
    /// replicates `value` every `N` bits across a `T`
    /// e.g. broadcast<uint16_t, 4>(0b111) == 0b0111'0111'0111'0111
    template<class T, const size_t N>
    [[nodiscard]] constexpr auto broadcast(const T value) noexcept -> T {
        T result{};
        for (size_t i = 0; i < (sizeof(T) * __CHAR_BIT__) / N; ++i) {
            result = T(result | T(T(value) << (i * N)));
        }
        return result;
    }

    namespace internal {
        // bit helpers on the register type (at most 64 bits wide)
        template<class A>
        [[nodiscard]] constexpr size_t ctz(const A value) noexcept {
            return __builtin_ctzll(uint64_t(value));
        }

        template<class A>
        [[nodiscard]] constexpr size_t clz(const A value) noexcept {
            // count only the leading zeros within the width of A
            return __builtin_clzll(uint64_t(value)) - (64u - sizeof(A) * __CHAR_BIT__);
        }

        template<class A>
        [[nodiscard]] constexpr size_t popcount(const A value) noexcept {
            return __builtin_popcountll(uint64_t(value));
        }
    } // end namespace internal

    /// TODO
    template<class T,
             const size_t Width = sizeof(uint64_t) / sizeof(T),
             typename TAbi = LogTypeTemplate<uint32_t(sizeof(T) * __CHAR_BIT__ * Width)>>
        requires((sizeof(T) * Width) <= sizeof(TAbi))
    struct swar_mask {
    	using value_type = bool;/// predefined
    	using abi_type = TAbi;
    
    	static constexpr size_t nbits = sizeof(T) * __CHAR_BIT__;
    	static constexpr abi_type lsb = broadcast<abi_type, nbits>(1u);
    	static constexpr abi_type msb = lsb << (nbits - 1u);
    
    	constexpr swar_mask() noexcept = default;
    	constexpr swar_mask(const swar_mask &) noexcept = default;
    	constexpr swar_mask(swar_mask &&) noexcept = default;
    	constexpr explicit swar_mask(const abi_type value) noexcept : value{value} {}
    
        /// \param index
        /// \return
    	[[nodiscard]] constexpr 
        auto operator[](const size_t index) const noexcept -> bool {
    		return value & abi_type(1u) << ((1u + index) * nbits - 1u);
    	}
    
        /// \return
    	[[nodiscard]] static constexpr 
        auto size() noexcept -> size_t { return Width; }
    
    	abi_type value{};
    };
    
    /// TODO
    template<typename T, 
             const size_t Width = sizeof(uint64_t) / sizeof(T),
             typename TAbi = LogTypeTemplate<uint32_t(sizeof(T) * __CHAR_BIT__ * Width)>>
      requires ((sizeof(T) * Width) <= sizeof(TAbi))
    struct swar {
        using value_type = T;
        using abi_type = TAbi;
        
        static constexpr size_t nbits = sizeof(T) * __CHAR_BIT__;
        static constexpr abi_type lsb = broadcast<abi_type, nbits>(1u);
        static constexpr abi_type msb = lsb << (nbits - 1u);
        
        constexpr swar() noexcept = default;
        constexpr swar(const swar&) noexcept = default;
        constexpr swar(swar&&) noexcept = default;
        constexpr explicit swar(const auto value) noexcept requires requires { abi_type(value); }
          : value{abi_type(value) * lsb} /// broadcast
        { }
    
        constexpr explicit swar(const auto* mem) noexcept {
            for (auto i = 0u; i < Width; ++i) {
                value |= abi_type(mem[i]) << (nbits * i);
            }
        }
    
        /// TODO doc
        constexpr explicit swar(const auto& gen) noexcept requires requires(size_t i) { gen(i); } {
            for (auto i = 0u; i < Width; ++i) {
                value |= abi_type(gen(i)) << (nbits * i);
            }
        }
        /// TODO doc
        [[nodiscard]] constexpr explicit operator abi_type() const noexcept { 
            return value; 
        }
        /// TODO doc
        [[nodiscard]] constexpr auto operator[](const size_t index) const noexcept -> T {
            // nbits is the full width of T, so the cast truncates to the lane
            return T(value >> (index * nbits));
        }
        /// TODO doc
        [[nodiscard]] static constexpr auto size() noexcept -> size_t {
            return Width; 
        }
        /// TODO doc
        [[nodiscard]] friend constexpr 
        auto operator==(const swar& lhs,
                        const swar& rhs) noexcept -> swar_mask<T, Width, TAbi> {
            return swar_mask<T, Width, TAbi>{~(((abi_type(lhs) ^ abi_type(rhs)) | msb) - lsb) & msb};
        }
        
        abi_type value{};
    };
    
    
    // TODO use cryptanalysislib functions
    // [[nodiscard]] constexpr auto ctz(const auto value) noexcept -> size_t {
    //       if constexpr (sizeof(value) <= sizeof(u32)) { return __builtin_ctz(value); }
    //  else if constexpr (sizeof(value) <= sizeof(u64)) { return __builtin_ctzl(value); }
    // }
    // 
    // [[nodiscard]] constexpr auto clz(const auto value) noexcept -> size_t {
    //       if constexpr (sizeof(value) <= sizeof(u32)) { return __builtin_clz(value); }
    //  else if constexpr (sizeof(value) <= sizeof(u64)) { return __builtin_clzl(value); }
    // }
    // 
    // [[nodiscard]] constexpr auto popcount(const auto value) noexcept -> size_t {
    //       if constexpr (sizeof(value) <= sizeof(u32)) { return __builtin_popcount(value); }
    //  else if constexpr (sizeof(value) <= sizeof(u64)) { return __builtin_popcountl(value); }
    // }
    
    template<class T, size_t Width, class TAbi>
    [[nodiscard]] constexpr auto all_of(const swar_mask<T, Width, TAbi>& s) noexcept -> bool {
      return s.value == s.msb;
    }
    
    template<class T, size_t Width, class TAbi>
    [[nodiscard]] constexpr auto any_of(const swar_mask<T, Width, TAbi>& s) noexcept -> bool {
      return s.value;
    }
    
    template<class T, size_t Width, class TAbi>
    [[nodiscard]] constexpr auto some_of(const swar_mask<T, Width, TAbi>& s) noexcept -> bool {
      return any_of(s) and not all_of(s);
    }
    
    template<class T, size_t Width, class TAbi>
    [[nodiscard]] constexpr auto none_of(const swar_mask<T, Width, TAbi>& s) noexcept -> bool {
      return not s.value;
    }
    
    template<class T, size_t Width, class TAbi>
    [[nodiscard]] constexpr auto find_first_set(const swar_mask<T, Width, TAbi>& s) noexcept -> size_t {
      return internal::ctz(s.value) / s.nbits;
    }
    
    template<class T, size_t Width, class TAbi>
    [[nodiscard]] constexpr auto find_last_set(const swar_mask<T, Width, TAbi>& s) noexcept -> size_t {
      return s.size() - (internal::clz(s.value) / s.nbits) - 1u;
    }
    
    template<class T, size_t Width, class TAbi>
    [[nodiscard]] constexpr auto popcount(const swar_mask<T, Width, TAbi>& s) noexcept -> size_t {
      return internal::popcount(s.value);
    }
    
    template<class> inline constexpr auto is_swar_v = false;
    template<class T, size_t Width, class TAbi>
    inline constexpr auto is_swar_v<swar<T, Width, TAbi>> = true;
    
    template<class> inline constexpr auto is_swar_mask_v = false;
    template<class T, size_t Width, class TAbi>
    inline constexpr auto is_swar_mask_v<swar_mask<T, Width, TAbi>> = true;
}; // end namespace cryptanalysislib::swar
