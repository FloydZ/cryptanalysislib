#ifndef SMALLSECRETLWE_ALGORITHM_PCS_H
#define SMALLSECRETLWE_ALGORITHM_PCS_H

#include <cstddef>

#include "random.h"
#include "thread/thread.h"

/// NOTE: README + usage example: `src/algorithm/README.md` (Pollard rho section)
/// TODO multithreading and then pcs
/// TODO gospers cycle finding


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
		if constexpr (brent) {
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


	/// NOTE: the inputs col1 and col2 are also outputs
	/// \tparam F
	/// \tparam C type of the check function, which checks if a global solution 
    ///         is found
	/// \tparam ExecPolicy threading policy
    /// \param policy
	/// \param f lambda function to execute a single step.
    /// \param c
	/// \param col1 input/output the starting value will be set onto this value,
	/// 		and if a collision is found in will be written into this value
	///			NOTE: the element before the collision will be return
	/// \param col2 input/output
	/// \param walk_len: maximal iterations until to exit the algorithm
	/// \return true/false if a solution/collision was found
	template<class F,
             class C,
             class ExecPolicy>
	[[nodiscard]] static bool run(ExecPolicy &&policy,
								  F &&f,
								  C &&c,
								  T &col1, T &col2,
								  const size_t walk_len) noexcept {
		Compare cmp;
        Distinguished d;
        
        // TODO probaly also safe the length of the walk within the distinguished point data set
        std::vector<T> distinguished_points;

        bool found = false;
        auto walk_f = [&]() __ATTRIBUTE__(always_inline) noexcept {
            while (!found) {
                // TODO: flavour
                T v = cryptanalysislib::rng();
                for (size_t i = 0; i < walk_len; i++) {
                    if (d(v)) {
                        if (c(distinguished_points, v) && !found) {
                            found = true;
							col1 = v;
							col2 = v;
                            goto finish;
                        }

                        distinguished_points.push_back(v);
                    }

                    v = f(v);
                }
            }

        finish:
            return true;
        };
        
        if (policy ){
        }

        constexpr static uint32_t nthreads = 2; // TODO
        std::vector<std::future<bool>> futures;
        for (uint32_t i = 0; i < nthreads; i++) {
            futures.emplace_back(walk_f);
        }

        for (uint32_t i = 0; i < nthreads; i++) {
            futures[i].wait();
        }

        // TODO reconstruct or whatever
        return true;
	}

	/// \tparam F
    /// \tparam C
    /// \param f
    /// \param c
    /// \param col1
    /// \param col2
    /// \param walk_len
    /// \return
	template<class F,
             class C>
	[[nodiscard]] static bool run(F &&f,
                                  C &&c,
						          T &col1, T &col2,
						          const size_t walk_len) noexcept {
        return PCS::run(cryptanalysislib::par_if(true), f, c, col1, col2, walk_len);
    }
};
#endif
