#ifndef CRYPTANALYSISLIB_GROWTH_POLICY_H
#define CRYPTANALYSISLIB_GROWTH_POLICY_H

#if !defined(CRYPTANALYSISLIB_HASHMAP_H)
#error "Do not include this file directly. Use: `#include <container/hashmap.h>`"
#endif

// SOURCE: https://github.com/Tessil/hopscotch-map/blob/master/include/tsl/hopscotch_growth_policy.h

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ratio>
#include <stdexcept>

#include "helper.h"
#include "binary.h"
#include "popcount/popcount.h"
#include "math/math.h"

namespace cryptanalysislib::hh {
	/**
	 * Grow the hash table by a factor of GrowthFactor keeping the bucket count to a
	 * power of two. It allows the table to use a mask operation instead of a modulo
	 * operation to map a hash to a bucket.
	 *
	 * GrowthFactor must be a power of two >= 2.
	 */
	template<std::size_t GrowthFactor>
	class power_of_two_growth_policy {
	public:

	/**
     * Called on the hash table creation and on rehash. The number of buckets for
     * the table is passed in parameter. This number is a minimum, the policy may
     * update this value with a higher value if needed (but not lower).
     *
     * If 0 is given, min_bucket_count_in_out must still be 0 after the policy
     * creation and bucket_for_hash must always return 0 in this case.
     */
	constexpr explicit power_of_two_growth_policy(std::size_t &min_bucket_count_in_out) noexcept {
		if (min_bucket_count_in_out > max_bucket_count()) {
			// the hashmap has reached its max size
			ASSERT(false);
		}

		if (min_bucket_count_in_out > 0) {
			min_bucket_count_in_out = round_up_to_power_of_two(min_bucket_count_in_out);
			m_mask = min_bucket_count_in_out - 1;
		} else {
			m_mask = 0;
		}
	}

	/**
	* Return the bucket [0, bucket_count()) to which the hash belongs.
	* If bucket_count() is 0, it must always return 0.
	*/
	[[nodiscard]] constexpr inline std::size_t bucket_for_hash(std::size_t hash) const noexcept {
		return hash & m_mask;
	}

	/**
	* Return the bucket count to use when the bucket array grows on rehash.
	*/
	[[nodiscard]] constexpr inline std::size_t next_bucket_count() const noexcept {
		if ((m_mask + 1) > max_bucket_count() / GrowthFactor) {
			// the hashmap has reached its max size
			ASSERT(false);
		}

		return (m_mask + 1) * GrowthFactor;
	}

	/**
    * Return the maximum number of buckets supported by the policy.
    */
	[[nodiscard]] constexpr inline std::size_t max_bucket_count() const noexcept {
		// Largest power of two.
		return (std::numeric_limits<std::size_t>::max() / 2) + 1;
	}

	/**
    * Reset the growth policy as if it was created with a bucket count of 0.
    * After a clear, the policy must always return 0 when bucket_for_hash is
    * called.
    */
	void clear() noexcept { m_mask = 0; }

	private:
		static_assert((GrowthFactor >= 2) &&
		              (cryptanalysislib::popcount::popcount(GrowthFactor) == 1),
		              "GrowthFactor must be a power of two >= 2.");

		std::size_t m_mask;
	};

	/**
     * Grow the hash table by GrowthFactor::num / GrowthFactor::den and use a modulo
     * to map a hash to a bucket. Slower but it can be useful if you want a slower
     * growth.
     */
	template<class GrowthFactor = std::ratio<3, 2>>
	class mod_growth_policy {
	public:
		constexpr explicit mod_growth_policy(const std::size_t &min_bucket_count_in_out) noexcept {
			if (min_bucket_count_in_out > max_bucket_count()) {
				ASSERT(false);
			}

			if (min_bucket_count_in_out > 0) {
				m_mod = min_bucket_count_in_out;
			} else {
				m_mod = 1;
			}
		}

		[[nodiscard]] constexpr inline std::size_t bucket_for_hash(std::size_t hash) const noexcept {
			return hash % m_mod;
		}

		[[nodiscard]] constexpr inline std::size_t next_bucket_count() const noexcept {
			if (m_mod == max_bucket_count()) {
				ASSERT(false);
			}

			const double next_bucket_count =
			        std::ceil(double(m_mod) * REHASH_SIZE_MULTIPLICATION_FACTOR);
			if (!std::isnormal(next_bucket_count)) {
				ASSERT(false);
			}

			if (next_bucket_count > double(max_bucket_count())) {
				return max_bucket_count();
			} else {
				return std::size_t(next_bucket_count);
			}
		}

		[[nodiscard]] constexpr inline std::size_t max_bucket_count() const { return MAX_BUCKET_COUNT; }

		constexpr inline void clear() noexcept { m_mod = 1; }

	private:
		static constexpr double REHASH_SIZE_MULTIPLICATION_FACTOR =
		        1.0 * GrowthFactor::num / GrowthFactor::den;
		static constexpr std::size_t MAX_BUCKET_COUNT =
		        std::size_t(static_cast<double>(std::numeric_limits<std::size_t>::max()) /
		                           REHASH_SIZE_MULTIPLICATION_FACTOR);

		static_assert(REHASH_SIZE_MULTIPLICATION_FACTOR >= 1.1,
		              "Growth factor should be >= 1.1.");

		std::size_t m_mod;
	};

	namespace detail {

#if SIZE_MAX >= ULLONG_MAX
#define TSL_HH_NB_PRIMES 51
#elif SIZE_MAX >= ULONG_MAX
#define TSL_HH_NB_PRIMES 40
#else
#define TSL_HH_NB_PRIMES 23
#endif

		static constexpr const std::array<std::size_t, TSL_HH_NB_PRIMES> PRIMES = {{
				1u, 5u, 17u, 29u, 37u, 53u, 67u, 79u, 97u, 131u, 193u, 257u,
				389u, 521u, 769u, 1031u, 1543u, 2053u, 3079u, 6151u, 12289u, 24593u, 49157u,
#if SIZE_MAX >= ULONG_MAX
		        98317ul, 196613ul, 393241ul, 786433ul, 1572869ul, 3145739ul, 6291469ul, 12582917ul,
		        25165843ul, 50331653ul, 100663319ul, 201326611ul, 402653189ul, 805306457ul, 1610612741ul, 3221225473ul, 4294967291ul,
#endif
#if SIZE_MAX >= ULLONG_MAX
		        6442450939ull, 12884901893ull, 25769803751ull, 51539607551ull, 103079215111ull, 206158430209ull,
		        412316860441ull, 824633720831ull, 1649267441651ull, 3298534883309ull, 6597069766657ull,
#endif
		}};

		template<unsigned int IPrime>
		static constexpr std::size_t mod(std::size_t hash) {
			return hash % PRIMES[IPrime];
		}

		// MOD_PRIME[iprime](hash) returns hash % PRIMES[iprime]. This table allows for
		// faster modulo as the compiler can optimize the modulo code better with a
		// constant known at the compilation.
		static constexpr const std::array<std::size_t (*)(std::size_t),
		                                  TSL_HH_NB_PRIMES>
		        MOD_PRIME = {{
		                &mod<0>,  &mod<1>,  &mod<2>,  &mod<3>,  &mod<4>,  &mod<5>,
		                &mod<6>,  &mod<7>,  &mod<8>,  &mod<9>,  &mod<10>, &mod<11>,
		                &mod<12>, &mod<13>, &mod<14>, &mod<15>, &mod<16>, &mod<17>,
		                &mod<18>, &mod<19>, &mod<20>, &mod<21>, &mod<22>,
#if SIZE_MAX >= ULONG_MAX
		                &mod<23>, &mod<24>, &mod<25>, &mod<26>, &mod<27>, &mod<28>,
		                &mod<29>, &mod<30>, &mod<31>, &mod<32>, &mod<33>, &mod<34>,
		                &mod<35>, &mod<36>, &mod<37>, &mod<38>, &mod<39>,
#endif
#if SIZE_MAX >= ULLONG_MAX
		                &mod<40>, &mod<41>, &mod<42>, &mod<43>, &mod<44>, &mod<45>,
		                &mod<46>, &mod<47>, &mod<48>, &mod<49>, &mod<50>,
#endif
		        }};

	}// namespace detail

	/**
	* Grow the hash table by using prime numbers as bucket count. Slower than
	* tsl::hh::power_of_two_growth_policy in general but will probably distribute
	* the values around better in the buckets with a poor hash function.
	*
	* To allow the compiler to optimize the modulo operation, a lookup table is
	* used with constant primes numbers.
	*
	* With a switch the code would look like:
	* \code
	* switch(iprime) { // iprime is the current prime of the hash table
	*     case 0: hash % 5ul;
	*             break;
	*     case 1: hash % 17ul;
	*             break;
	*     case 2: hash % 29ul;
	*             break;
	*     ...
	* }
	* \endcode
	*
	* Due to the constant variable in the modulo the compiler is able to optimize
	* the operation by a series of multiplications, substractions and shifts.
	*
	* The 'hash % 5' could become something like 'hash - (hash * 0xCCCCCCCD) >> 34)
	* * 5' in a 64 bits environment.
	*/
	class prime_growth_policy {
	public:
		constexpr explicit prime_growth_policy(std::size_t &min_bucket_count_in_out) noexcept {
			auto it_prime = std::lower_bound(
			        detail::PRIMES.begin(), detail::PRIMES.end(), min_bucket_count_in_out);
			if (it_prime == detail::PRIMES.end()) {
				ASSERT(false);
			}

			m_iprime = static_cast<unsigned int>(
			        std::distance(detail::PRIMES.begin(), it_prime));
			if (min_bucket_count_in_out > 0) {
				min_bucket_count_in_out = *it_prime;
			} else {
				min_bucket_count_in_out = 0;
			}
		}

		[[nodiscard]] constexpr inline std::size_t bucket_for_hash(std::size_t hash) const noexcept {
			return detail::MOD_PRIME[m_iprime](hash);
		}

		[[nodiscard]] constexpr inline std::size_t next_bucket_count() const noexcept {
			if (m_iprime + 1 >= detail::PRIMES.size()) {
				ASSERT(false);
			}

			return detail::PRIMES[m_iprime + 1];
		}

		[[nodiscard]] constexpr inline std::size_t max_bucket_count() const noexcept {
			return detail::PRIMES.back();
		}

		constexpr void clear() noexcept { m_iprime = 0; }

	private:
		uint32_t m_iprime;

		static_assert(std::numeric_limits<decltype(m_iprime)>::max() >=
		                      detail::PRIMES.size(),
		              "The type of m_iprime is not big enough.");
	};

	///
	class fibonacci_growth_policy {
	public:
		constexpr fibonacci_growth_policy() = default;

		constexpr explicit fibonacci_growth_policy(std::size_t &min_bucket_count_in_out) noexcept {
			/// TODO
			(void)min_bucket_count_in_out;

		}

		[[nodiscard]] constexpr inline size_t bucket_for_hash(const size_t hash) const noexcept{
			return (11400714819323198485ull * hash) >> shift;
		}

		[[nodiscard]] constexpr inline size_t keep_in_range(const size_t index,
		                                             const size_t num_slots_minus_one) const noexcept{
			return index & num_slots_minus_one;
		}

		[[nodiscard]] constexpr inline int8_t next_size_over(size_t & size) const noexcept {
			size = std::max(size_t(2), round_up_to_power_of_two(size));
			return 64u - bits_log2(size);
		}

		constexpr inline void commit(int8_t shift) noexcept {
			this->shift = shift;
		}

		constexpr inline void reset() noexcept {
			shift = 63;
		}

	private:
		int8_t shift = 63;
	};
}

#endif
