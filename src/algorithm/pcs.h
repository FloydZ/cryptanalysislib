#ifndef SMALLSECRETLWE_ALGORITHM_PCS_H
#define SMALLSECRETLWE_ALGORITHM_PCS_H

#include <atomic>
#include <cstddef>
#include <future>
#include <map>
#include <mutex>
#include <vector>

#include "random.h"
#include "thread/thread.h"
#include "algorithm/countr.h"

/// NOTE: README + usage example: `src/algorithm/README.md` (Pollard rho section)


static size_t walk_len = 0;
#if __cplusplus > 201709L
#include <concepts>


/// describes a valid compare function in the 
/// context of Pollard Rho
template<class C, class T>
concept RhoCompareAble = requires(C c) {
	requires requires (const T a,
	                   const T &aa){
		{ c(aa, aa, aa, aa) } -> std::convertible_to<bool>;
	};
};

/// 
template<class C, class T>
concept PCSDistinguishAble = requires(C c) {
	requires requires (const T a,
	                   const T &aa){
		{ c(aa) } -> std::convertible_to<bool>;
	};
};
#endif


/// Configuration of `PollardRho`
struct PollardRhoConfig {
	/// cycle finding of the flavoured `run`: Brent (true) or Floyd (false)
	const bool brent = true;

	/// if true: Gosper's cycle finding (HAKMEM 132) is used by the flavoured
	/// `run`, regardless of `brent`
	const bool gosper = false;
};
constexpr static PollardRhoConfig pollardRhoConfig{};

/// \tparam Compare compare object, which needs to take 4 argumens
///     So the following ()operator must be implemented:
///         operator()(const T &a1, const T &a2, const T &b1, const T &b2)
///     where:
///         a2 = f(a1), b2 = f(b1)
///     so a1 and b1 are always the predecessors of a2 and b2. This additional
///     information is needed for algorithms which do flavouring.
/// \tparam T base element to compare
/// \tparam config see `PollardRhoConfig`
template<class Compare,
		 class T,
		 const PollardRhoConfig &config = pollardRhoConfig>
#if __cplusplus > 201709L
	requires RhoCompareAble<Compare, T>
#endif
class PollardRho {
private:

	constexpr static bool brent = config.brent;
	constexpr static bool gosper = config.gosper;

public:
	constexpr PollardRho() noexcept {};

	/// NOTE: the inputs col1 and col2 are also outputs
	/// \tparam F
	/// \param f lambda function to execute a single step.
	/// \param col1 input/output the starting value will be set onto this value,
	/// 		and if a collision is found in will be written into this value
	///			NOTE: the element before the collision will be return
	/// \param col2 input/output
	/// \param max_iters: maximal iterations until to exit the algorithm
	/// \return true/false if a solution/collision was found
	template<class F>
#if __cplusplus > 201709L
		requires std::regular_invocable<F, const T&>
#endif
	[[nodiscard]] constexpr static bool run(F &&f,
											T &col1, T &col2,
											const size_t max_iters = size_t(-1ull)) noexcept {
		Compare cmp;

		size_t i = 0;
		T a1=col1, b1=col2;
		while (i < max_iters) {
			const T a2 = f(a1);
			const T b2_= f(b1);
			const T b2 = f(b2_);

			if (cmp(a1, a2, b2_, b2)) {
				col1 = a1;
				col2 = b2_;
				return true;
			}

			a1 = a2;
			b1 = b2;
			i += 1u;
		}

		std::cout << "no solution found" << std::endl;
		return false;
	}

	/// NOTE: returns in col1/col2 the preimages of the collision
	/// \tparam F
	/// \tparam Flavour
	/// \param f
	/// \param flavour
	/// \param x1
	/// \param y1
	/// \param x2
	/// \param y2
	/// \param max_iters
	/// \return
	template<class F,
			 class Flavour>
#if __cplusplus > 201709L
		requires std::regular_invocable<F, const T&> &&
				 std::regular_invocable<Flavour, const T&>
#endif
	[[nodiscard]] static bool run(F &&f,
								  Flavour &&flavour,
								  T &x1, T &y1,
								  T &x2, T &y2,
								  const size_t max_iters=-1ull) noexcept __ATTRIBUTE__(always_inline) {
		Compare cmp;
		bool ret = false;

		const auto sp = x1;
		size_t i = 0;
		if constexpr (gosper) {
			// Gosper (HAKMEM 132): the table slot `k` holds the last element
			// x_j (and j) with ctz(j) == k, i.e. only O(log(n)) elements. Every
			// new element is compared against all of them. A match x_n = x_j
			// means that n-j is a multiple of the cycle length.
			T tv[64];
			size_t tj[64];
			uint64_t valid = 1;
			tv[0] = sp; tj[0] = 0;

			T prev = sp, cur = sp;
			size_t n = 0, d = 0;
			while ((i < max_iters) && (d == 0)) {
				prev = cur;
				cur = f(flavour(prev));
				n += 1; i += 1; walk_len += 1;
				for (uint32_t k = 0; k < 64; ++k) {
					if (((valid >> k) & 1u) && cmp(sp, tv[k], prev, cur)) {
						d = n - tj[k];
						break;
					}
				}

				const uint32_t k = cryptanalysislib::countr_zero(n);
				tv[k] = cur; tj[k] = n;
				valid |= 1ull << k;
			}

			// early exit, no collision found
			if (d == 0) { return false; }

			// phase 2 (as Floyd): `sp` and `f^d(sp)` step together, they meet
			// at the cycle entry, their predecessors are the collision. At most
			// `n` steps: else `sp` is on the cycle.
			T a = sp, b = sp;
			for (size_t k = 0; k < d; ++k) { b = f(flavour(b)); }
			for (size_t k = 0; k <= n; ++k) {
				walk_len += 1;
				const T an = f(flavour(a));
				const T bn = f(flavour(b));
				if (cmp(a, an, b, bn)) {
					x1 = a; y1 = b;
					x2 = an; y2 = bn;
					return true;
				}
				a = an; b = bn;
			}
			return false;
		} else if constexpr (brent) {
			x2 = x1;
			y2 = f(flavour(x1));
			uint32_t power = 1, lam = 1;
			
			while (!cmp(x1, x2, y1, y2) && (i < max_iters)) {
				// time to start a new power of two?
				if (power == lam) {
					x2 = y2;
					power *= 2;
					lam = 0;
				}

				y1 = y2;
				y2 = f(flavour(y1));
				i += 1;
				lam += 1;
				walk_len += 1;
			}

			// early exit, no collision found
			if (!cmp(x1, x2, y1, y2)) { return false; }

			x2 = sp;
			y2 = sp;
			for (uint32_t j = 0; j < lam; j++) {
				y2 = f(flavour(y2));
			}
			
			while (!cmp(x1, x2, y1, y2)) {
				x1 = x2;
				x2 = f(flavour(x1));
				y1 = y2;
				y2 = f(flavour(y1));
			}
			return true;
		} else {
			// Floyd: phase 1, tortoise `x` and hare `y` meet within the cycle.
			// NOTE: at the meeting both have the same predecessor, so (as in
			// 	Brent's variant) `sp` is the dummy first preimage of `cmp`.
			// 	Before, the meeting was compared with the two predecessors,
			// 	which never differ, so this loop never terminated.
			T x = sp, y = sp;
			while (i < max_iters) {
				i += 1;
				walk_len += 1;
				const T xn = f(flavour(x));
				const T yh = f(flavour(y));
				const T yn = f(flavour(yh));
				x = xn; y = yn;
				if (cmp(sp, xn, yh, yn)) [[unlikely]] {
					ret = true;
					break;
				}
			}

			if (!ret) { return false; }

			// phase 2: from the start and from the meeting point, one step
			// at a time: they meet at the cycle entry, their predecessors
			// are the collision. At most `i` steps: else `sp` is on the
			// cycle and there are no distinct preimages.
			T a = sp, b = y;
			for (size_t k = 0; k <= i; ++k) {
				walk_len += 1;
				const T an = f(flavour(a));
				const T bn = f(flavour(b));
				if (cmp(a, an, b, bn)) {
					x1 = a; y1 = b;
					x2 = an; y2 = bn;
					return true;
				}
				a = an; b = bn;
			}
			return false;
		}

		return ret;
	}
};


/// Parallel collision search (van Oorschot, Wiener): every thread walks
/// `x -> f(flavour(x))` from random starting points until a distinguished
/// point is reached. All distinguished points are stored together with the
/// start and the length of their walk. If two walks end in the same
/// distinguished point, they are reconstructed to find the collision
/// `a != b, f(flavour(a)) == f(flavour(b))`.
/// \tparam Compare golden collision check `cmp(a, f(a), b, f(b))`, see `PollardRho`.
/// 	Collisions for which it returns false are ignored.
/// \tparam Distinguished `d(x)` returns true if x is a distinguished point
/// \tparam T element type. NOTE: must be constructible from an `uint64_t`
/// 	(random starting points), comparable with `==` and `<`.
template<class Compare,
         class Distinguished,
		 class T>
#if __cplusplus > 201709L
	requires RhoCompareAble<Compare, T> &&
             PCSDistinguishAble<Distinguished, T>
#endif
class PCS {
private:

public:
	constexpr PCS() noexcept {};

	/// \tparam F step function
	/// \tparam Flavour flavour function, applied before every step
	/// \tparam ExecPolicy threading policy
	/// \param policy if parallel: all threads of its pool are walking
	/// \param f step function
	/// \param flavour
	/// \param col1 output: first preimage of the collision
	/// \param col2 output: second preimage of the collision
	/// \param max_walk_len walks without a distinguished point after
	/// 		`max_walk_len` steps are discarded (they are probably in a cycle)
	/// \param max_walks maximal number of walks (all threads together)
	/// \return true/false if a (golden) collision was found
	template<class F,
             class Flavour,
             class ExecPolicy>
	[[nodiscard]] static bool run(ExecPolicy &&policy,
								  F &&f,
								  Flavour &&flavour,
								  T &col1, T &col2,
								  const size_t max_walk_len,
								  const size_t max_walks = size_t(-1ull)) noexcept {
		// distinguished point -> (start of the walk, length of the walk)
		std::map<T, std::pair<T, size_t>> table;
		std::mutex m;
		std::atomic<bool> found = false;
		size_t walks = 0;

		auto walk_f = [&]() -> bool {
			Compare cmp;
			Distinguished d;
			while (!found.load(std::memory_order_relaxed)) {
				T start;
				{
					// NOTE: the rng is not thread safe
					std::lock_guard<std::mutex> guard(m);
					if (walks >= max_walks) { return false; }
					walks += 1;
					start = T(cryptanalysislib::rng());
				}

				T v = start;
				size_t len = 0;
				while (!d(v) && (len < max_walk_len)) {
					v = f(flavour(v));
					len += 1;
				}

				if (!d(v)) { continue; }

				T start2;
				size_t len2;
				{
					std::lock_guard<std::mutex> guard(m);
					const auto it = table.find(v);
					if (it == table.end()) {
						table.emplace(v, std::pair<T, size_t>{start, len});
						continue;
					}
					start2 = it->second.first;
					len2 = it->second.second;
				}

				// reconstruction: the longer walk goes ahead, then both step
				// together until they merge
				T a = start, b = start2;
				for (; len > len2; --len) { a = f(flavour(a)); }
				for (; len2 > len; --len2) { b = f(flavour(b)); }
				// one start lies on the other walk: no collision
				if (a == b) { continue; }

				T an = f(flavour(a)), bn = f(flavour(b));
				while (!(an == bn)) {
					a = an; b = bn;
					an = f(flavour(a));
					bn = f(flavour(b));
				}

				if (!cmp(a, an, b, bn)) { continue; }

				std::lock_guard<std::mutex> guard(m);
				if (!found.load()) {
					col1 = a; col2 = b;
					found.store(true);
				}
				return true;
			}
			return false;
		};

		if (is_seq<ExecPolicy>(policy)) {
			return walk_f();
		}

		auto &task_pool = *policy.pool();
		const uint32_t nthreads = task_pool.get_num_threads();
		std::vector<std::future<bool>> futures;
		for (uint32_t i = 0; i < nthreads; i++) {
			futures.emplace_back(task_pool.submit(walk_f));
		}

		for (uint32_t i = 0; i < nthreads; i++) {
			futures[i].wait();
		}

		return found.load();
	}

	/// same as above, without flavour
	template<class F,
             class ExecPolicy>
	[[nodiscard]] static bool run(ExecPolicy &&policy,
								  F &&f,
								  T &col1, T &col2,
								  const size_t max_walk_len,
								  const size_t max_walks = size_t(-1ull)) noexcept {
		return PCS::run(policy, f, [](const T &x) { return x; },
		                col1, col2, max_walk_len, max_walks);
	}

	/// same as above, with all threads of the default pool
	template<class F>
	[[nodiscard]] static bool run(F &&f,
						          T &col1, T &col2,
						          const size_t max_walk_len,
						          const size_t max_walks = size_t(-1ull)) noexcept {
        return PCS::run(cryptanalysislib::par_if(true), f, col1, col2, max_walk_len, max_walks);
    }
};
#endif
