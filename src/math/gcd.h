#ifndef CRYPTANALYSISLIB_ALGORITHM_GCD_H
#define CRYPTANALYSISLIB_ALGORITHM_GCD_H

#include <type_traits>
#include <algorithm>

namespace cryptanalysislib {
    namespace internal {
        ///
        template<typename T>
        #if __cplusplus > 201709L
            requires std::is_arithmetic_v<T>
        #endif
        constexpr static T gcd_recursive_v0(const T a,
                                            const T b) noexcept {
            if (b == 0) { return a; }
            if (a == 0) { return b; }

            // Base case
            if (a == b)
                return a;

            // a is greater
            if (a > b) {
                return gcd_recursive_v0<T>(a - b, b);
            }

            return gcd_recursive_v0<T>(a, b - a);
        }

        /// \tparam T
        /// \param a
        /// \param b
        /// \return
        template<typename T>
        #if __cplusplus > 201709L
            requires std::is_arithmetic_v<T>
        #endif
        constexpr static T gcd_recursive_v1(const T a,
                                            const T b) noexcept {
            if  (b == 0) {
                return a;
            }

            return gcd_recursive_v1<T>(b, a % b);
        }

        /// tparam T
        /// param a
        /// param b
        /// return
        template<typename T>
        #if __cplusplus > 201709L
            requires std::is_arithmetic_v<T>
        #endif
        constexpr static T gcd_recursive_v2(const T a,
                                            const T b) noexcept {
            return b ? gcd_recursive_v2<T>(b, a % b) : a;
        }

        /// \tparam T
        /// \param a
        /// \param b
        /// \return
        template<typename T>
        #if __cplusplus > 201709L
            requires std::is_arithmetic_v<T>
        #endif
        constexpr static T gcd_non_recursive_v1(T a,
                                                T b) noexcept {
            while (b > 0) {
                a %= b;
                std::swap(a, b);
            }
            return a;
        }

        /// \tparam T
        /// \param a
        /// \param b
        /// \return
        template<typename T>
        #if __cplusplus > 201709L
            requires std::is_arithmetic_v<T>
        #endif
        constexpr static T gcd_non_recursive_v2(T a,
                                                T b) noexcept {
            while (b) b ^= a ^= b ^= a %= b;
            return a;
        }

        /// \tparam T
        /// \param a
        /// \param b
        /// \return
        template<typename T>
        #if __cplusplus > 201709L
            requires std::is_integral_v<T>
        #endif
        constexpr static T gcd_binary(const T a_,
                                      const T b_) noexcept {
            // NOTE: computed on the absolute values in the unsigned type of
            // the same width, so it works for every integral type
            using U = std::make_unsigned_t<T>;
            U a = U(a_), b = U(b_);
            if constexpr (std::is_signed_v<T>) {
                a = (a_ < 0) ? U(U(0) - a) : a;
                b = (b_ < 0) ? U(U(0) - b) : b;
            }

            if (a == 0) return T(b);
            if (b == 0) return T(a);

            auto ctz = [](const U x) constexpr noexcept -> uint32_t {
                if constexpr (sizeof(U) <= 8) {
                    return __builtin_ctzll(uint64_t(x));
                } else {
                    const uint64_t lo = uint64_t(x);
                    return lo ? __builtin_ctzll(lo) : 64u + __builtin_ctzll(uint64_t(x >> 64u));
                }
            };

            // Stein's algorithm
            const uint32_t shift = ctz(a | b);
            a >>= ctz(a);
            while (b != 0) {
                b >>= ctz(b);
                if (a > b) {
                    const U t = a; a = b; b = t;
                }
                b -= a;
            }

            return T(a << shift);
        }
    } // end namespace internal


	/// \tparam T
    /// \param a
    /// \param b
    /// \return
template<typename T>
    #if __cplusplus > 201709L
        requires std::is_integral_v<T>
    #endif
    constexpr static T gcd(T a, T b) noexcept {
        return internal::gcd_binary(a, b);
    }

} // end namespace cryptanalysislib
#endif
