#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "algorithm/subsetsum.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

#include "params.h"

int main() {
	constexpr uint32_t n = PARAM_n;
	constexpr uint64_t q = (1ul << n);

	using T 			= uint64_t;
	using Value     	= BinaryVector<n>;
	using Label    		= kAry_Type_T<q>;
	using Matrix 		= FqVector<T, n, q, true>;
	using Element		= Element_T<Value, Label, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	Matrix A; A.random();
	constexpr uint32_t k_lower1=0, k_higher1=PARAM_l1, k_higher2=(PARAM_l1+PARAM_l2);
	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

#if PARAM_g != PARAM_n
	using Enumerator = BinarySinglePartialSingleEnumerator<List, n, 1, 2, n/2>;

	constexpr size_t baselist_size = Enumerator::max_list_size;
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size},l3{baselist_size},l4{baselist_size};

	Tree t{1, A, 0};
	t.info();
	t.template join4lists_on_iT_hashmap_v2
			<k_lower1, k_higher1, k_higher1, k_higher2, Enumerator>
			(out, l1, l2, l3, l4, target);
#else
	using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, PARAM_n1_3>;
	// using Enumerator = BinaryRandomEnumerator<List, n/2, PARAM_n1_3>;

	constexpr size_t baselist_size = Enumerator::max_list_size;
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	Tree t{1, A, 0};
	t.info();
	t.template join4lists_twolists_on_iT_hashmap_v2
			<k_lower1, k_higher1, k_higher1, k_higher2, Enumerator>
			(out, l1, l2, target);

#endif
}