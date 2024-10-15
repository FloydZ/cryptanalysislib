#ifndef CRYPTANALYSISLIB_ALGORITHM_SUBSETSUM_H
#define CRYPTANALYSISLIB_ALGORITHM_SUBSETSUM_H

#include <cstdint>
#include "algorithm/pcs.h"
#include "algorithm/random_index.h"
#include "algorithm/int2weight.h"
#include "tree.h"

/// TODO wrapper function schreiben die einfach nur die TRee funktionen wrapped => die dann bcj und hgj nennen


/// generates a rng subset sum instance
/// NOTE:
/// 	- nr of indices which are generated is = n/2
/// 	- max index = n
/// \tparam Label
/// \tparam List, std:vector, std::array
/// \tparam Matrix
/// \param target return value
/// \param weights return value
/// \param AT transposed matrix, actually vector in this case
/// \param n number of bits of the label
/// \param mitm if true: will make sure that the solution
/// 	evenly splits between each half
/// \param debug if true: will print the solution
template<typename Label,
		typename List,
		typename Matrix>
constexpr static void generate_subsetsum_instance(Label &target,
												  List &weights,
												  const Matrix &AT,
												  const uint32_t n,
												  const bool mitm = true,
												  const bool debug = true) noexcept {
	if (!IsStdArray<List>()) { weights.reserve(n/2);}
	target.zero();
	if (mitm) { generate_random_mitm_indices(weights, n);
	} else { generate_random_indices(weights, n); }

	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, AT[0][weights[i]]);
	}

	if (debug) {
		std::cout << target << " , subset sum target" << std::endl;
		for (const auto &w : weights) {
			std::cout << w << " ";
		}
		std::cout << std::endl;
	}
}



struct SSS {
    const uint32_t n = 32;
    const uint64_t q = 1ull << n;
    const uint64_t walk_len = 1u << 4u;
};

template<class Element>
struct SubSetSumCmp {
    using Label = Element::LabelType;
    using C = Label::ContainerType::LimbType;

    constexpr static uint32_t bit_pos = 0;
    constexpr static C mask = ((C)1ull) << bit_pos;

	/// simple comparison struct
	/// only a2 and b2 are compared for equality
	/// \param a1 predecessor of a2
	/// \param a2 value to be
	/// \param b1 predecessor of b2
	/// \param b2 value to be compared
	/// \return true if a2==b2, and a1!=b1;
	auto operator()(const Element &a1,
	                const Element &a2,
	                const Element &b1,
	                const Element &b2) const noexcept __attribute__((always_inline)) {
		// get the lowest bit
        const C la = a1.label.value();
        const C lb = b1.label.value();

		const C alb = la & mask;
		const C blb = lb & mask;

		// and make sure, that they are different
		if (alb == blb) {
			return false;
		}

        // if they are the same we found a collision
		return a2.label == b2.label;
	}
};

template<const SSS &instance>
class sss {
public:
    constexpr static uint32_t n = instance.n;
    constexpr static uint64_t q = instance.q;

	using T 		= uint64_t;
	using Value     = BinaryVector<n>;
    using Label     = kAry_Type_T<q>;
	using Matrix 	= FqVector<T, n, q, true>;
	using Element	= Element_T<Value, Label, Matrix>;
	using List		= List_T<Element>;
	using Tree		= Tree_T<List>;
	using L 		= Label::LimbType;

	Matrix A;
	Label target;
	constexpr sss(Matrix &A, Label &target) noexcept
	    : A(A), target(target) {
	}

	bool run() noexcept {
		// enumerate weight n//4 on n//2 coordinates.
		// NOTE: elements are randomly chosen, so no chase sequence or whatsoever
		using Enumerator = BinaryRandomEnumerator<List, n / 2, n / 4>;
		Enumerator en{A};

		using rho = PollardRho<SubSetSumCmp<Element>, Element>;

		constexpr static size_t size = 1ull << (n / 4u);
		List L1{size}, L2{size}, out{size};

		// write random elements in the lists in a mitm split. The elements will have
		// weight n//4 on each halve.
		en.template run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n / 4);

		Tree t{1, A, 0};


		Element a, b;
		a.random();
		b.random();

		while (true) {
			/// restart every X runs
			const bool val = rho::run(
			        [&](const Element &in) {
				        // merge the two base lists on the full length ([0, n)) on the target
				        // t.join2lists(out, L1, L2, target, 0, n, false);

				        Element ret;
				        for (size_t i = 0; i < out.load(); ++i) {
					        // TODO do something
					        // const auto el = out[i];

					        // remap each solution back to a binary string of weight n //4
					        static uint32_t weights[n / 4];
					        int2weights<L>(weights, in.label.value(), n, n / 4, n / 4);

					        // super simple distinguish function
					        if (in.label.value() & 1u) {
						        ret.label += target;
					        }

					        for (uint32_t j = 0; j < n / 4; j++) {
						        ret.label += A[0][weights[i]];
					        }
				        }

				        // TODO return something
				        return ret;
			        },
			        a, b, 1u << 6);

			if (val) {
				break;
			}

			// resample the flavour
			a.random(A);
			b.random(A);
		}
	}
};


#endif
