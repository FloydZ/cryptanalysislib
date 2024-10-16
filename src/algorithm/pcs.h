#ifndef SMALLSECRETLWE_ALGORITHM_PCS_H
#define SMALLSECRETLWE_ALGORITHM_PCS_H

#include <cstddef>
#include "thread/thread.h"

/// TODO multithreading and then pcs
/// TODO Rho without Floyds cycle finding

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


/// \tparam Compare compare object, which needs to take 4 argumens
///     So the following ()operator must be implemented:
///         operator()(const T &a1, const T &a2, const T &b1, const T &b2)
///     where:
///         a2 = f(a1), b2 = f(b1)
///     so a1 and b1 are always the predecessors of a2 and b2. This additional
///     information is needed for algorithms which do flavouring.
/// \tparam T base element to compare
template<class Compare,
		 class T>
#if __cplusplus > 201709L
	requires RhoCompareAble<Compare, T>
#endif
class PollardRho {
private:

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
	[[nodiscard]] constexpr static bool run(F &&f,
						T &col1, T &col2,
						const size_t max_iters = size_t(-1ull)) noexcept {
		Compare cmp;

		size_t i = 0;
		T a1=col1, b1=col2;
		while (i < max_iters) {
			const T a2 = f(a1);
			const T b2_ =f(b1);
			const T b2 = f(b2_);

			if (cmp(a1, a2, b2_, b2)) {
				col1 = a1;
				col2 = b2_;
				return true;
			}

			a1 = a2;
			b1 = b2;
			i += 1;
		}
		return false;
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
	/// \param max_iters: maximal iterations until to exit the algorithm
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
        auto walk_f = [&]() __attribute__((always_inline)) noexcept {
            while (!found) {
                // TODO: flavour 
                T v = rng<T>();
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
	}

    ///
	template<class F,
             class C>
	[[nodiscard]] static bool run(F &&f,
                                  C &&c,
						          T &col1, T &col2,
						          const size_t walk_len) noexcept {
        return PCS::run(cryptanalysislib::par, f, c, col1, col2, walk_len);
    }
};
#endif
