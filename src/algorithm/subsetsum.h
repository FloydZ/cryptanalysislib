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
/// \tparam List, vector, array or iteratable
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


/// TODO image and explanation of config parameters
struct SSS {
	/// these are just fake numbers. Enter your own correct ones.
    const uint32_t n = 32;
    const uint64_t q = 1ull << n;
	const uint32_t bp = 1;
	const uint32_t l1 = 9;
	const uint32_t l2 = 11;
    const uint64_t walk_len = 1u << 4u;
};

///
/// @tparam Element 
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

/// TREE(x, y):
///                   out
///                 ┌───┐                  level 2
///                 └───┘ match on x
///                l_1│l_2
///         ┌─────────┴─────┐
///      ┌──┴───┐           │              level 1
///      └┐    ┌┘           │
///       └┐  ┌┘HMiL        │
///        └┬─┘match on x-y │
///        0│l_1            │ match on y
///     ┌───┴──┐           0│l_1
///     │      │        ┌───┴───┐
///   ┌─┴─┐ ┌──┴───┐ ┌──┼──┐ ┌──┼───┐      level 0
///   │   │ └┐    ┌┘ │     │ └┐    ┌┘
///   │   │  └┐  ┌┘  │     │  └┐  ┌┘
///   └───┘   └──┘   └─────┘   └──┘
///    L1     HML2      L1      HML2
///
/// instance to solve: <a, e> = t
/// flavor values: b_1,b_2
///
/// // collision function
/// f_i(s, iT) = {
///		i = lsb(s)
///		// NOTE: iT++ if no solution found
///		o = (i == 0) ? TREE(s, iT) : TREE(t-s, iT)
///		return o
/// }
///
/// // flavour function
/// P(x) {
///		return b_1 * x + b_2
/// }
///
/// rho() = {
///		//
///		x1,y1 = rng(0, 2**n), f_i(x1)
///		x2,y2 = 0,0
///
///		iT= rng(0, 2**(l_1))
///		s = rng(0, 2**(l_2+l_1))
///
///		// NOTE: the loop also ends if a max length is reached
///		while(x1 != y1 &&
///			x2,y2 = x1,y1
///			x1 = f_i(x2)
///			y1 = f_i(f_i(y2))
///		}
///
///		if (lsb(x2) != lsb(y2)) {
///			return found
///		}
///
///		x1 = P(x2)
///		y1 = P(y2)
///		goto restart
/// }
///
template<const SSS &instance>
class sss_d2 {
public:
    constexpr static uint32_t n = instance.n;
    constexpr static uint64_t q = instance.q;

	using T 		= uint64_t;
	using Value     = BinaryVector<n>;
    using Label = kAry_Type_T<q>;
	using Matrix 	= FqVector<T, n, q>;
	using Element	= Element_T<Value, Label, Matrix>;
	using List		= List_T<Element>;
	using Tree		= Tree_T<List>;
	using L 		= Label::LimbType;
	using V 		= Value::LimbType;

	// instance to solve: <A, e> = target
	const Matrix A;
	const Label target;

	/// \param A
	/// \param target
	constexpr sss_d2(const Matrix &A,
					 const Label &target) noexcept
	    : A(A), target(target) {
	}

	///
	bool run() noexcept {
		constexpr static uint32_t k_lower1 = 0,
								  k_upper1 = instance.l1,
								  k_lower2 = instance.l1,
								  k_upper2 = instance.l1+instance.l2;

		using rho = PollardRho<SubSetSumCmp<Element>, Element>;

		/// allocate the enumerator and the base lists
		// using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, instance.bp>;
		using Enumerator = BinaryLexicographicEnumerator<List, n/2, instance.bp>;
		constexpr static size_t size = Enumerator::max_list_size;
		List L1{size}, L2{size}, out{50};

		Enumerator en{A};
		en.template run
			<std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&L1, &L2, n/2);

		using D = typename Label::DataType;
		using E = std::pair<size_t, size_t>;

		// constexpr static size_t factor = 2;
		constexpr static size_t L1_bucketsize = 100; // factor * (Enumerator::max_list_size >> (instance.l1));
		constexpr static size_t iL_bucketsize = 100; // factor * (Enumerator::max_list_size * Enumerator::max_list_size >> (instance.l2 + instance.l1));

		constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
				L1_bucketsize, 1ull<<(k_upper1-k_lower1), 1
		};
		constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
				iL_bucketsize, 1ull<<(k_upper2-k_lower2), 1
		};

		using HML2 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower1, k_upper1, 2>>;
		using HMiL = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_lower2, k_upper2, 2>>;
		HML2 *hmL2 = new HML2{};
		HMiL *hmiL = new HMiL{};

		/// prepare the hashmaps
		for (size_t i = 0; i < L2.load(); ++i) {
			hmL2->insert(L2[i].label.value(), i);
		}

		/// dummy object
		Tree t{1, A, 0};

		Label s, tree_iT, one; one.set(1, 0);
		s.random(0, 1ull << k_upper2);
		tree_iT.random(0, 1ull << k_upper1);
		Element x, y, x_old, y_old;

		//flavout values:
		constexpr static L a = 2, b = 2;

		// \return value=int2weight(b_2 * flavor(e) + b_2))
		auto flavour = [&](const Element &e) __attribute__((always_inline)){
			Element ret;
			const L c = (a * e.label.value() + b) % instance.q;
			// TODO not correct
			int2weight_bits<V, L>(ret.value.ptr(), c, n, n, instance.n / 4);
			ret.recalculate_label(A);
			return ret;
		};

		/// pollard rho
		auto f =  [&](const Element &in) __attribute__((always_inline)) {
			// reset a few things
			out.set_load(0);
			Label tree_target, tmp_iT;

			// depending on the lowest bit
			if (in.label.value() & 1u) {
				Label::sub(tree_target, target, s);
			} else {
				tree_target = s;
			}

			// restart the tree, as long as we do not have any outputs
			size_t iters = 0;
			while (out.load() == 0) {
				hmiL->clear();
				Label::add(tree_iT, tree_iT, one);
				Label::sub(tmp_iT, tree_target, tree_iT);

				// join to intermediate list (hashmap)
				// NOTE: `prepare==false`, because its already done
				t.template join2lists_on_iT_hashmap_v2
					<k_lower1, k_upper1>
					(*hmiL, L1, L2, *hmL2, tree_iT, false);

				// join to output list
				t.template twolevel_streamjoin_on_iT_hashmap_v2
					<k_lower1, k_upper1, k_lower2, k_upper2>
					(out, *hmiL, L1, L2, *hmL2, target, tmp_iT);
				// TODO: optimize the filtering, use a lambda to directly exit upon the first match

				iters += 1;
			}

			// std::cout << target << std::endl;
			// std::cout << out << std::endl;
			ASSERT(out.load() > 0);
			for (const auto &o :out) {
				ASSERT(o.is_correct(A));
			}

			std::cout << "iters:" << iters << std::endl;
			std::cout << target << std::endl;
			std::cout << out << std::endl;

			Element ret = out[0];
			ASSERT(ret.label.is_equal(target, 0, k_upper2));
			return ret;
		};

		// init
		x_old.random(A);
		y_old = f(x_old);

		// start loop
		while (true) {
			x = flavour(x_old);
			y = flavour(y_old);
			s.random(0, 1ull << k_upper2);
			tree_iT.random(0, 1ull << k_upper1);

			std::cout << x << " x" << std::endl;
			std::cout << y << " y" << std::endl;
			std::cout << s << " s" << std::endl;
			std::cout << tree_iT << std::endl << std::endl;

			/// restart every X runs
			if (rho::run(f, x, y, instance.walk_len)) {
				break;
			}

			x_old = x;
			y_old = y;
		}

		Element sol;
		Element::add(sol, x, y);

		std::cout << x << std::endl;
		std::cout << y << std::endl;
		std::cout << sol << std::endl;
		std::cout << target << std::endl;

		// memory cleanup
		delete hmL2;
		delete hmiL;

		return true;
	}
};


#endif
