#ifndef CRYPTANALYSISLIB_ALGORITHM_SUBSETSUM_H
#define CRYPTANALYSISLIB_ALGORITHM_SUBSETSUM_H

#include <cstdint>
#include "algorithm/pcs.h"
#include "algorithm/int2weight.h"
#include "tree.h"

struct SSS {
    constexpr static uint32_t n = 32;
    constexpr static uint64_t q = 1ull << n;
    constexpr static uint64_t walk_len = 1u << 4u;
};

template<class Element>
struct SubSetSumCmp {
    using Label = Element::Label;
    using C = Label::ContainerType;

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
void sss() noexcept {
    constexpr static uint32_t n = instance.n;
    constexpr static uint64_t q = instance.q;

	using T 			= uint64_t;
	using Value     	= BinaryVector<n>;
    using Label         = kAry_Type_T<q>;
	using Matrix 		= FqMatrix<T, n, q, true>;
	using Element		= Element_T<Value, Label, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	// represent the SubSetSum Instance a 1xn row matrix= row vector
	// this makes a lot of things alot easier
	Matrix A; A.random();
	Label target;
	std::vector<uint32_t> sol;
	generate_subsetsum_instance(target, sol, A, n);

	// enumerate weight n//4 on n//2 coordinates.
	// NOTE: elements are randomly chosen, so no chase sequence or whatsoever
	using Enumerator = BinaryRandomEnumerator<List, n/2, n/4>;
	Enumerator en{A};

	constexpr static size_t size = 1ull << (n / 4u);
	List L1{size}, L2{size}, out{size};
	Tree t{1, A, 0};


	Element a,b;
	a.random(); b.random();

	while(true) {
		/// restart every X runs
		const bool val = PollardRho<SubSetSumCmp<Element>, T>::run(
		        [&A, &target, &L1, &L2, &out, &en, &t](const Label &in) {
			// reset the lists
			L1.set_load(0); L2.set_load(0), out.set_load(0);

			// write random elements in the lists in a mitm split. The elements will have
			// weight n//4 on each halve.
			en.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
					  (&L1, &L2, n/4);

			// merge the two base lists on the full length ([0, n)) on the target
			t.join2lists(out, L1, L2, target, 0, n, false);

			T ret = T(0);
			for (size_t i = 0; i < out.load(); ++i) {
				// TODO do something
				// const auto el = out[i];

				// remap each solution back to a binary string of weight n //4
				static uint32_t weights[n/4];
				int2weights<Label::LimbType>(weights, in.value(), n, n/4, n/4);

				// super simple distinguish function
				if (in.value() & 1u) {
					ret += target;
				}

				for (uint32_t j = 0; j < n/4; j++) {
					ret += A[0][weights[i]];
				}

			}

			// TODO return something
			return ret;
		}, a, b, 1u << 6);

		if (val) {
			break;
		}

		// resample the flavour
		a = rng<Label::LimbType>(q);
		b = rng<Label::LimbType>(q);
	}
}


#endif
