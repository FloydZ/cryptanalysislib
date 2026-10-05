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
            if (b == 0) { return b; }
            if (a == 0) { return a; }

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
            requires std::is_arithmetic_v<T>
        #endif
        constexpr static T gcd_binary(T a,
                                      T b) noexcept {
            if (a == 0) return b;
            if (b == 0) return a;

            int az = __builtin_ctz(a);
            int bz = __builtin_ctz(b);
            auto shift = std::min(az, bz);
            b >>= bz;

            while (a != 0) {
                a >>= az;
                int diff = b - a;
                az = __builtin_ctz(diff);
                b = std::min(a, b);
                a = std::abs(diff);
            }

            return b << shift;
        }
    } // end namespace internal


	/// \tparam T
    /// \param a
    /// \param b
    /// \return
template<typename T>
    #if __cplusplus > 201709L
        requires std::is_arithmetic_v<T>
    #endif
    constexpr static T gcd(T a, T b) noexcept {
        return internal::gcd_binary(a, b);
    }

} // end namespace cryptanalysislib
#endif
