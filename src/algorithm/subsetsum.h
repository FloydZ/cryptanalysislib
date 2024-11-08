#ifndef CRYPTANALYSISLIB_ALGORITHM_SUBSETSUM_H
#define CRYPTANALYSISLIB_ALGORITHM_SUBSETSUM_H

#include <cstdint>
#include "algorithm/pcs.h"
#include "algorithm/random_index.h"
#include "algorithm/int2weight.h"
#include "tree.h"

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
	const uint32_t d = 2;
    const uint32_t n = 32;
    const uint64_t q = 1ull << n;
	// base p
	const uint32_t bp = 1;
	const uint32_t l1 = 9;
	const uint32_t l2 = 11;
	const uint32_t l3 = 0;
    const uint64_t walk_len = 1u << 10u;

	// flavouring prime
    const uint64_t flavour_q = 509;

	// after how many iterations should the
	// algorithm print some runtime information
	const size_t print_iterations = 512;

	/// print some basic informaton about this
	/// data container
	constexpr void info() const noexcept {
		std::cout << "{ \"name\": \"SubSetSumConfig\""
		          << ", \"d\": " << d
		          << ", \"n\": " << n
		          << ", \"q\": " << q
		          << ", \"bp\": " << bp
		          << ", \"l1\": " << l1
		          << ", \"l2\": " << l2
		          << ", \"l3\": " << l3
		          << ", \"walk_len\": " << walk_len
		          << ", \"flavour_q\": " << flavour_q
		          << ", \"print_iterations\": " << print_iterations
		          << " }" << std::endl;
	}
};

/// very simple comparison class.
/// Simply checks if two elements are equal
/// between [l1+l2, l1+l2+l1)
/// \tparm Element base element type
/// \tparm SSS subset sum config class
template<class Element,
		 const SSS &instance>
struct SubSetSumCmp {
    using Label = Element::LabelType;
    using Value = Element::ValueType;
    using C = Label::ContainerType::LimbType;

	constexpr static uint32_t k_lower = instance.l1 + instance.l2;
	constexpr static uint32_t k_upper = k_lower + instance.l1;

	constexpr static uint32_t weight = instance.n/2;

	/// simple comparison struct
	/// only a2 and b2 are compared for equality
	/// \param a1 predecessor of a2
	/// \param a2 value to be
	/// \param b1 predecessor of b2
	/// \param b2 value to be compared
	/// \return true if a2.label==b2.label, weight is correct, and a1!=b1;
	auto operator()(const Element &a1,
	                const Element &a2,
	                const Element &b1,
	                const Element &b2) const noexcept __attribute__((always_inline)) {
		(void)a1;
		(void)b1;
		return a2.template is_equal<k_lower, k_upper>(b2);
	}
};

/// TREE(t, iT):
///		do this outside of the rho
///                   out
///                 ┌───┐                  level 2
///                 └───┘ match on x
///                l_1│l_2			e1+e2 = t-e3-e4
///         ┌─────────┴─────┐
///      ┌──┴───┐           │              level 1
///      └┐    ┌┘           │
///       └┐  ┌┘HMiL        │
///        └┬─┘match on iT  │
///        0│l_1            │ match on t-iT
///     ┌───┴──┐           0│l_1    e4 = t-iT-e3 mod q
///     │      │        ┌───┴───┐
///   ┌─┴─┐ ┌──┴───┐ ┌──┼──┐ ┌──┴───┐      level 0
///   │   │ └┐    ┌┘ │     │ └┐    ┌┘
///   │   │  └┐  ┌┘  │     │  └┐  ┌┘
///   └───┘   └──┘   └─────┘   └──┘
///    L1     HML2      L1      HML2
///     e1     e2      e3        e4
///
/// instance to solve: <a, e> = t
/// flavor values: b_1,b_2
///
/// // global typedefs
///		using Element = [value, label], s.t. label = <a, value>
///
/// // collision function
/// f_i(iT) = {
///		// NOTE:
///			- s is passes as a lambda reference
///		i = lsb(input)
///		// NOTE: iT++ if no solution found
///		o = (i == 0) ? TREE(s, iT) : TREE(t-s, iT)
///		return o
/// }
///
/// // flavour function
/// P(x: Element) {
///		return b_1 * x.label[l, l+l1] + b_2 mod p circa 2**l1
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
///		// NOTE: the weight check and the function output check are done outside of the rho
///		while((x1&LSB == y1&LSB) || (x2 !=[l,..., l+l1] y2) {
///			x2,y2 = x1,y1
///			x1 = f_i(P(x2))
///			y1 = f_i(P(f_i(P(y2))))
///		}
///
///		if (lsb(x2) != lsb(y2)) {
///			return found
///		}
///
///		goto restart
/// }
template<const SSS &instance>
class sss_d2 {
public:
    constexpr static uint32_t n = instance.n;
    constexpr static uint64_t q = instance.q;

	using T 		= uint64_t;
	using Value     = BinaryVector<n>;
    using Label		= kAry_Type_T<q>;
	using Matrix 	= FqVector<T, n, q>;
	using Element	= Element_T<Value, Label, Matrix>;
	using List		= List_T<Element>;
	using Tree		= Tree_T<List>;
	using L 		= Label::LimbType;
	using V 		= Value::LimbType;

	// needed config for the rho collision search
	constexpr static uint32_t rho_k_lower = instance.l1 + instance.l2;
	constexpr static uint32_t rho_k_upper = rho_k_lower + instance.l1;
	constexpr static uint32_t rho_weight = instance.n/2;

	static_assert(q > 1);
	static_assert(n > 0);
	static_assert(instance.bp < (n/2));
	static_assert((instance.l1 + instance.l2 + instance.l3) <= n);
	static_assert(is_prime(instance.flavour_q));

	// instance to solve: <A, e> = target
	const Matrix A;
	const Label global_target;

	/// \param A
	/// \param target
	constexpr sss_d2(const Matrix &A,
					 const Label &target) noexcept
	    : A(A), global_target(target) {
	}

	///
	bool run() noexcept {
		constexpr static uint32_t k_lower1 = 0,
								  k_upper1 = instance.l1,
								  k_lower2 = instance.l1,
								  k_upper2 = instance.l1+instance.l2;

		using rho = PollardRho<SubSetSumCmp<Element, instance>, Element>;
		instance.info();

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

		Label s, one; one.set(1, 0);
		s.random(0, 1ull << k_upper2);
		Element x1,x2,y1,y2;
        L z; z = rng(q);

		//flavout values:
		L b_1 = rng<L>(instance.flavour_q), b_2 = rng<L>(instance.flavour_q);

        /// \return <e,z> & 1
		auto function_selector = [&](const Element &e) __attribute__((always_inline)) {
            const L tmp1 = z ^ e.label.value();
            const uint32_t tmp2 = cryptanalysislib::popcount::popcount(tmp1);
            return tmp2 & 1u;
        };

		/// \return value=(b_2 * flavor(e) + b_2))
		///			label = A*value
		auto flavour = [&](const Element &e) __attribute__((always_inline)) {
			Element ret;
			// ASSERT(e.is_correct(A));
			const L c = (b_1 * (e.label.value() >> (instance.l1+instance.l2)) + b_2) % instance.flavour_q;
			*ret.value.ptr() = c;
			ret.recalculate_label(A);
			return ret;
		};

		/// pollard rho f function
		size_t tree_iters = 0;
		size_t f_calls = 0;
		auto f =  [&](const Element &c1) __attribute__((always_inline)) {
			f_calls += 1;

			// reset a few things
			out.set_load(0);
			Label tree_target, tmp_iT, tree_iT;
			tree_iT = c1.label;

			// depending on the lowest bit
			const uint32_t bit = function_selector(c1);
			if (bit) {
				Label::sub(tree_target, global_target, s);
			} else {
				tree_target = s;
			}

			// restart the tree, as long as we do not have any outputs
			while (out.load() == 0) {
				// reset the intermediate hashmap
				hmiL->clear();

				// prepare the itermediate target for the next round
				Label::add(tree_iT, tree_iT, one);
				Label::sub(tmp_iT, tree_target, tree_iT);

				// join to intermediate list (hashmap)
				// NOTE: `prepare==false`, because its already done
				t.template join2lists_on_iT_v2
					<k_lower1, k_upper1>
					(*hmiL, L1, L2, *hmL2, tree_iT, false);

				// join to output list
				t.template twolevel_streamjoin_on_iT_hashmap_v2
					<k_lower1, k_upper1, k_lower2, k_upper2, 4*instance.bp>
					(out, *hmiL, L1, L2, *hmL2, tree_target, tmp_iT);

				tree_iters += 1;
			}

			// std::cout << target << std::endl;
			// std::cout << out << std::endl;
			// ASSERT(out.load() > 0);
			// ASSERT(iters < 100);
			// size_t wrong = 0;
			// for (size_t it = 0; it < out.load(); it++) {
			// 	ASSERT(out[it].is_correct(A));
			// 	if (!out[it].label.is_equal(tree_target, 0, k_upper2)) {
			// 		wrong += 1;
			// 	}
			// }
			Element ret = out[0];
			if (bit) {
				Label::sub(ret.label, global_target, out[0].label);
			}
			// ASSERT(ret.label.is_equal(tree_target, 0, k_upper2));
			// ASSERT(wrong == 0);

			// debug information
			// std::cout << "iters:" << iters << std::endl;
			// std::cout << "wrong:" << wrong << std::endl;
			// std::cout << tree_target << ", tree_target" << std::endl;
			// std::cout << out << std::endl;
			// std::cout << ret << ", ret" << std::endl;

			return ret;
		};

		const auto start = std::chrono::high_resolution_clock::now();
		size_t rho_calls=0, collisions=0, pass_rho=0, pass_function_selector=0, pass_weight_check=0;

		// start loop
		restart:
		while (true) {
			rho_calls += 1;
			x1.random(A);
			y1 = f(x1);
			s.random(0, 1ull << (k_upper2 + k_upper1));
			b_1 = rng<L>(instance.flavour_q);
			b_2 = rng<L>(instance.flavour_q);
			z = rng<L>(instance.q);

			// if ((iters % instance.print_iterations) == 0) {
			// 	std::cout << "iters: " << iters << std::endl;
			// }

			// NOTE: restart every `instance.walk_len` runs
			// NOTE: the weight check and the check if the collision is between
			//		 two different functions is done outside of the rho function,
			//		 to assure that we do not run into useless cycles.
			if (rho::run(f, flavour, x1, y1, x2, y2, instance.walk_len)) {
                pass_rho += 1;
				const L alb = function_selector(x1);
				const L blb = function_selector(y1);

				// ... and make sure, that they are different
				if (alb == blb) { continue; }
                pass_function_selector += 1;


				// debugging
				// Element sol;
				// Element::add(sol, x2, y2);
				// Label ss;
				// Label::sub(ss, global_target, s);
				// std::cout << x2 << ", x" << std::endl;
				// std::cout << y2 << ", y" << std::endl;
				// std::cout << sol << ", sum" << std::endl;
				// sol.recalculate_label(A);
				// std::cout << sol << ", sum" << std::endl;
				// std::cout << global_target << ", global_target" << std::endl;
				// std::cout << s << ", s" << std::endl;
				// std::cout << ss << ", ss" << std::endl;

				// weight check:
				Value tmp;
				Value::add(tmp, x2.value, y2.value);
				if (tmp.popcnt() == rho_weight) {
					pass_weight_check += 1;
					break;
				}
			}
		}

		const auto duration  = std::chrono::high_resolution_clock::now() - start;
		const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);

		Element sol, sol2;
		Element::add(sol, x2, y2);
		sol2 = sol;
        sol2.recalculate_label(A);
		collisions += 1u;

		// debugging
		// std::cout << x2 << ", x" << std::endl;
		// std::cout << y2 << ", y" << std::endl;
		// std::cout << sol2 << ", sol2" << std::endl;
		// std::cout << sol << ", sol" << std::endl;
		// std::cout << sol2.label << ", sol2" << std::endl;
		// std::cout << sol.label << ", sol" << std::endl;
		// std::cout << global_target << ", global_target" << std::endl;
		// ASSERT(sol2.label.is_equal(sol.label));
		if (!global_target.is_equal(sol2.label)) {
			goto restart;
		}

		// memory cleanup
		delete hmL2;
		delete hmiL;

		// print some cool stats
		std::cout << "{ "
				  << "\"rho_calls\": " << rho_calls
				  << ", \"collisions\": " << collisions
				  << ", \"f_calls\": " << f_calls
				  << ", \"pass_rho\": " << pass_rho
				  << ", \"pass_function_selector\": " << pass_function_selector
				  << ", \"pass_weight_check\": " << pass_weight_check
				  << ", \"avg_tree_iters\": " << (double)tree_iters/(double)f_calls
				  << ", \"avg_walk_len\": " << (((double)(f_calls - rho_calls))/3.0)/(double)rho_calls
				  << ", \"seconds\": " << seconds.count()
				  << " }" << std::endl;
		return true;
	}
};

template<const SSS &instance>
class HGJ {
    constexpr static uint32_t n = instance.n;
    constexpr static uint64_t q = instance.q;

	constexpr static uint32_t k_l1 = 0;
	constexpr static uint32_t k_h1 = instance.l1;
	constexpr static uint32_t k_l2 = k_h1;
	constexpr static uint32_t k_h2 = k_l2 + instance.l2;
	constexpr static uint32_t k_l3 = k_h2;
	constexpr static uint32_t k_h3 = k_l2 + instance.l3;

	constexpr static uint32_t filter_weight = n/2;

	static_assert(instance.d >= 2);
	static_assert(instance.d <= 3);

public:
	using T 		= uint64_t;
	using Value     = BinaryVector<n>;
    using Label		= kAry_Type_T<q>;
	using Matrix 	= FqVector<T, n, q>;
	using Element	= Element_T<Value, Label, Matrix>;
	using List		= List_T<Element>;
	using Tree		= Tree_T<List>;
	using L 		= Label::LimbType;
	using V 		= Value::LimbType;

	// instance to solve: <A, e> = target
	const Matrix A;
	const Label global_target;

	/// \param A
	/// \param target
	constexpr HGJ(const Matrix &A,
				  const Label &target) noexcept
	    : A(A), global_target(target) {
	}

	size_t run() noexcept {
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
				L1_bucketsize, 1ull<<(k_h1-k_l1), 1
		};
		constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
				iL_bucketsize, 1ull<<(k_h2-k_l2), 1
		};

		using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_l1, k_h1, 2>>;
		using HML1 = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_l2, k_h2, 2>>;
		HML0 *hmL0 = new HML0{};
		HML1 *hmL1 = new HML1{};

		/// prepare the hashmaps
		for (size_t i = 0; i < L2.load(); ++i) {
			hmL0->insert(L2[i].label.value(), i);
		}

		/// dummy object
		Tree t{1, A, 0};

		if constexpr (instance.d == 2) {
			t.template join4lists_twolists_on_iT_hashmap_v2
				<k_l1, k_h1, k_l2, k_h2>
				(out, L1, L2, *hmL0, *hmL1, global_target, false);
		}

		if constexpr (instance.d == 3) {
			using F = std::pair<E, E>;
			using HML2 = SimpleHashMap<D, F, simpleHashMapConfigL1, Hash<D, k_l3, k_h3, 2>>;
			HML2 *hmL2 = new HML2{};

			t.template join8lists_twolists_on_iT_v2
				<k_l1, k_h1, k_l2, k_h2, k_l3, k_h3, 0, filter_weight>
				(out, L1, L2, *hmL0, *hmL1, *hmL2, global_target);
			delete hmL2;
		}

		std::cout << out;

		delete hmL0;
		delete hmL1;
	}
};


template<const SSS &instance>
class BCJ {
    constexpr static uint32_t n = instance.n;
    constexpr static uint64_t q = instance.q;

	constexpr static uint32_t k_l1 = 0;
	constexpr static uint32_t k_h1 = instance.l1;
	constexpr static uint32_t k_l2 = k_h1;
	constexpr static uint32_t k_h2 = k_l2 + instance.l2;
	constexpr static uint32_t k_l3 = k_h2;
	constexpr static uint32_t k_h3 = k_l2 + instance.l3;

	constexpr static uint32_t filter_weight = n/2;

	static_assert(instance.d >= 2);
	static_assert(instance.d <= 3);

public:
	using T 		= uint64_t;
	using Value     = FqPackedVector<n, 3, T, true>;
    using Label		= kAry_Type_T<q>;
	using Matrix 	= FqVector<T, n, q>;
	using Element	= Element_T<Value, Label, Matrix>;
	using List		= List_T<Element>;
	using Tree		= Tree_T<List>;
	using L 		= Label::LimbType;
	using V 		= Value::LimbType;

	// instance to solve: <A, e> = target
	const Matrix A;
	const Label global_target;

	/// \param A
	/// \param target
	constexpr BCJ(const Matrix &A,
				  const Label &target) noexcept
	    : A(A), global_target(target) {
	}

	size_t run() noexcept {
		using Enumerator = ListEnumerateMultiFullLength<List, n/2, 3, instance.bp>;
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
				L1_bucketsize, 1ull<<(k_h1-k_l1), 1
		};
		constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
				iL_bucketsize, 1ull<<(k_h2-k_l2), 1
		};

		using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_l1, k_h1, 3>>;
		using HML1 = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_l2, k_h2, 3>>;
		HML0 *hmL0 = new HML0{};
		HML1 *hmL1 = new HML1{};

		/// prepare the hashmaps
		for (size_t i = 0; i < L2.load(); ++i) {
			hmL0->insert(L2[i].label.value(), i);
		}

		/// dummy object
		Tree t{1, A, 0};

		auto f1 = [](HML1 &out,const List &L1,const List &L2,const size_t i,const size_t j)
					__attribute__((always_inline)) {
			using HMOutValueType = HML1::data_type;
			Label e;
			Label::add(e, L1[i].label, L2[j].label);
			// NOTE: that is shifted
			out.insert(e.value(), HMOutValueType{i, j});
				return false;
		};

		if constexpr (instance.d == 2) {
			auto f2 = [](List &out, Element &a1, Element &a2, size_t,size_t,size_t,size_t) __attribute__((always_inline)) {
				Element t;
				Element::add(t, a1, a2);
				if ((t.value.popcnt(0b10) == 0) && (t.value.popcnt(0))) {
					out[0] = t;
					out.set_load(1);
					return true;
				}
				return false;
			};
			t.template join4lists_twolists_on_iT_v2
				<k_l1, k_h1, k_l2, k_h2>
				(out, L1, L2, *hmL0, *hmL1, global_target, false, f1, f2);
		}

		if constexpr (instance.d == 3) {
			using F = std::pair<E, E>;
			using HML2 = SimpleHashMap<D, F, simpleHashMapConfigL1, Hash<D, k_l3, k_h3, 2>>;
			HML2 *hmL2 = new HML2{};

			t.template join8lists_twolists_on_iT_v2
				<k_l1, k_h1, k_l2, k_h2, k_l3, k_h3, 0, filter_weight>
				(out, L1, L2, *hmL0, *hmL1, *hmL2, global_target);
			delete hmL2;
		}

		std::cout << out;

		delete hmL0;
		delete hmL1;

		return 1;
	}

};
#endif
