#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"
#include "algorithm/random_index.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t n    = 16ul;
constexpr uint32_t q    = (1ul << n);

using T 			= uint64_t;
//using Value     	= kAryContainer_T<T, n, 2>;
using Value     	= BinaryContainer<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

// unused ignore
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};

// NOTE: random enumeration of the values
TEST(SubSetSum, Simple) {
	// even it says matrix. It is a simple row vector
	Matrix A;
	A.fill(0);

	// // log size
	constexpr size_t list_size = 10;
	static std::vector<uint64_t> tbl{{0, n}};
	Tree t{1, A, list_size, tbl, __level_filter_array};

	t[0].random(1u << list_size, A);
	t[1].random(1u << list_size, A);
	t.join_stream(0);

	EXPECT_EQ(1u << 20u, t[2].load());
}

TEST(SubSetSum, JoinForLevelTwo) {
	Matrix A;
	A.fill(0);

	static std::vector<uint64_t> tbl{{0, 5, 10, n}};
	Tree t{2, A, 10, tbl, __level_filter_array};

	t[0].random(1u << 2u, A);
	t[1].random(1u << 2u, A);
	t.join_stream(0);
	t.join_stream(1);
	EXPECT_EQ(1u << 8u, t[3].load());
}

TEST(SubSetSum, JoinForLevelThree) {
	Matrix A;
	A.fill(0);
	static std::vector<uint64_t> tbl{{0, 5, 10, n}};
	Tree t{3, A, 10, tbl, __level_filter_array};

	t[0].random(1u << 2u, A);
	t[1].random(1u << 2u, A);
	t.join_stream(0);
	t.join_stream(1);
	t.join_stream(2);
	EXPECT_EQ(1u << 16u, t[4].load());
}

TEST(SubSetSum, JoinRandomListsLevel0) {
	Matrix A;
	A.random();

	static std::vector<uint64_t> tbl{{0, n}};
	Tree t{2, A, 10u, tbl, __level_filter_array};

	t[0].random(1u << 8u, A);
	t[1].random(1u << 8u, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);

	uint64_t num = 0;
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				num++;
			}
		}
	}

	t.join_stream(0);
	EXPECT_NE(0, num);
	EXPECT_EQ(t[2].load(), num);
}

// NOTE: takes very long
TEST(SubSetSum, JoinRandomListsLevel1) {
	Matrix A;
	A.random();

	static std::vector<uint64_t> tbl{{0, n/2, n}};
	Tree t{2, A, 11, tbl, __level_filter_array};

	t[0].random(1u << 12u, A);
	t[1].random(1u << 12u, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(0);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << 12u, A);
	t[1].random(1u << 12u, A);

	t[2].sort_level(1, tbl);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);

	uint64_t num = 0;
	Element el{};
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				Element::add(el, t[0][i], t[1][j]);

				for (size_t o = 0; o < t[2].load(); ++o) {
					if (el.is_equal(t[2][o], tbl[1], tbl[2])) {
						num++;
					}
				}
			}
		}
	}

	t.join_stream(1);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[3].load(), num);
}

TEST(TreeTest, JoinRandomListsLevel2) {
	Matrix A;
	A.random();

	constexpr size_t base_size = 5;
	static std::vector<uint64_t> tbl{{0, n/3, 2*n/3, n}};
	Tree t{3, A, base_size, tbl, __level_filter_array};

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);

	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(0);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);

	t[2].sort_level(1, tbl);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(1);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t[3].sort_level(2, tbl);

	uint64_t num = 0;
	Element el{};
	Element el2{};
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				Element::add(el, t[0][i], t[1][j]);
				for (size_t o = 0; o < t[2].load(); ++o) {
					if (el.is_equal(t[2][o], tbl[1], tbl[2])) {
						Element::add(el2, el, t[2][o]);
						for (size_t r = 0; r < t[3].load(); ++r) {
							if (el2.is_equal(t[3][r], tbl[2], tbl[3])) {
								num++;
							}
						}
					}
				}
			}
		}
	}

	t.join_stream(2);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[4].load(), num);
}

TEST(SubSetSum, JoinRandomListsLevel3) {
	Matrix A;
	A.random();

	constexpr size_t base_size = 5;
	static std::vector<uint64_t> tbl{{0, n/4, n/2, 3*n/4, n}};
	Tree t{4, A, 10, tbl, __level_filter_array};

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(0);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);

	t[2].sort_level(1, tbl);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(1);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t[3].sort_level(2, tbl);
	t.join_stream(2);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t[4].sort_level(3, tbl);

	uint64_t num = 0;
	Element el{};
	Element el2{};
	Element el3{};
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				Element::add(el, t[0][i], t[1][j]);
				for (size_t o = 0; o < t[2].load(); ++o) {
					if (el.is_equal(t[2][o], tbl[1], tbl[2])) {
						Element::add(el2, el, t[2][o]);
						for (size_t r = 0; r < t[3].load(); ++r) {
							if (el2.is_equal(t[3][r], tbl[2], tbl[3])) {
								Element::add(el3, el2, t[3][r]);
								for (size_t w = 0; w < t[4].load(); ++w) {
									if (el3.is_equal(t[4][w], tbl[3], tbl[4])) {
										num++;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	t.join_stream(3);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[5].load(), num);
}

TEST(SubSetSum, dissection) {
	Label::info();
	Matrix::info();

	Matrix AT; AT.random();
	std::cout << AT;

	List out{1<<n};
	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, AT[0][weights[i]]);
	}

	Tree::dissection4(out, target, AT);

	EXPECT_GE(out.load(), 1);
	for (size_t i = 0; i < out.load(); ++i) {
		target.print_binary();
		out[i].label.print_binary();
		// std::cout << target << ":" << out[i].label << std::endl;
		Label tmp;
		AT.mul(tmp, out[i].value);

		EXPECT_EQ(target.is_equal(tmp), true);
	}
}

TEST(SubSetSum, join2lists) {
	Matrix A; A.random();
	const uint64_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, A[0][weights[i]]);
	}


	// compute the number of collisions via the simple quadratic algorithm
	Label el{};
	uint64_t num = 0;
	for (size_t i = 0; i < l1.load(); ++i) {
		for (size_t j = 0; j < l2.load(); ++j) {
			Label::add(el, l1[i].label, l2[j].label);
			if (el.is_equal(target, k_lower, k_higher)) {
				num += 1;
			}
		}
	}

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	Tree::join2lists(out, l1, l2, target, k_lower, k_higher, true);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);

		// revert the changes of the right list
		Element l2_tmp = l2[i];
		l2_tmp.label.neg();
		l2_tmp.label += target;
		EXPECT_EQ(l2_tmp.is_correct(A), true);
	}

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		EXPECT_EQ(out[i].label.is_zero(k_lower, k_higher), true);

		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		out[i].recalculate_label(A);
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower, k_higher));

		if (!(Label::cmp(out[i].label, target, k_lower, k_higher))) {
			right = false;
			wrong++;
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(),1u<<3);
	EXPECT_LT(out.load(),1u<<7);
	EXPECT_EQ(out.load(), num);
}

TEST(SubSetSum, constexpr_join2lists) {
	Matrix A; A.random();
	constexpr uint64_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, A[0][weights[i]]);
	}


	// compute the number of collisions via the simple quadratic algorithm
	Label el{};
	uint64_t num = 0;
	for (size_t i = 0; i < l1.load(); ++i) {
		for (size_t j = 0; j < l2.load(); ++j) {
			Label::add(el, l1[i].label, l2[j].label);
			if (el.is_equal(target, k_lower, k_higher)) {
				num += 1;
			}
		}
	}

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	Tree::template join2lists<k_lower, k_higher>(out, l1, l2, target, true);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);

		// revert the changes of the right list
		Element l2_tmp = l2[i];
		l2_tmp.label.neg();
		l2_tmp.label += target;
		EXPECT_EQ(l2_tmp.is_correct(A), true);
	}

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		EXPECT_EQ(out[i].label.is_zero(k_lower, k_higher), true);

		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		out[i].recalculate_label(A);
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower, k_higher));

		if (!(Label::cmp(out[i].label, target, k_lower, k_higher))) {
			right = false;
			wrong++;
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(),1u<<3);
	EXPECT_LT(out.load(),1u<<7);
	EXPECT_EQ(out.load(), num);
}

TEST(SubSetSum, join4lists) {
	Matrix A; A.random();
	constexpr uint64_t k_lower1=0, k_higher1=n/2;
	constexpr uint64_t k_lower2=n/2, k_higher2=n;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size}, l3{baselist_size}, l4{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l3, &l4, n/2);

	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, A[0][weights[i]]);
	}

	std::cout << A << std::endl;
	for (const auto &w : weights) {
		std::cout << w << ",";
	}
	std::cout << std::endl;
	std::cout << target << std::endl;

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
		EXPECT_EQ(l3[i].is_correct(A), true);
		EXPECT_EQ(l4[i].is_correct(A), true);
	}

	Tree::join4lists(out, l1, l2, l3, l4, target,
	                 k_lower1, k_higher1, k_lower2, k_higher2, true);

	uint32_t right=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		// just for debugging, we are not filtering
		if (out[i].value.popcnt() != n/2) {
			continue;
		}

		// first check thats is zero
		EXPECT_EQ(out[i].label.is_zero(k_lower1, k_higher2), true);

		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		out[i].recalculate_label(A);
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, 0, n));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, 0, n));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, 0, n));
		std::cout << out[i];

		// TODO not finished
		if (Label::cmp(out[i].label, target)) {
			right += 1;
		}
	}

	EXPECT_GT(right,0);
}

TEST(SubSetSum, join2lists_on_iT) {
	Matrix A; A.random();
	const uint64_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, A[0][weights[i]]);
	}

	l1.sort_level(k_lower, k_higher);
	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	Tree::join2lists_on_iT(out, l1, l2, target, k_lower, k_higher);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		// NOTE that we do not recalculate the label
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower, k_higher));

		if (!(Label::cmp(out[i].label, target, k_lower, k_higher))) {
			right = false;
			wrong++;
		}
	}


	Label el{};
	uint64_t num = 0;
	for (size_t i = 0; i < l1.load(); ++i) {
		for (size_t j = 0; j < l2.load(); ++j) {
			Label::add(el, l1[i].label, l2[j].label);
			if (el.is_equal(target, k_lower, k_higher)) {
				num += 1;
			}
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(), 1u<<3);
	EXPECT_LT(out.load(), 1u<<7);
	EXPECT_EQ(out.load(), num);
}


TEST(SubSetSum, twolevel_streamjoin) {
	Matrix A; A.random();
	const uint64_t k_lower1=0, k_higher1=8;
	const uint64_t k_lower2=8, k_higher2=16;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size}, iT{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&iT, nullptr, n/2);

	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, A[0][weights[i]]);
	}

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}
	// pre


	Tree::twolevel_streamjoin(out, iT, l1, l2,
	                          k_lower1, k_higher1, k_lower2, k_higher2, true);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);

		// revert the changes of the right list
		Element l2_tmp = l2[i];
		l2_tmp.label.neg();
		l2_tmp.label += target;
		EXPECT_EQ(l2_tmp.is_correct(A), true);
	}

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		out[i].recalculate_label(A);
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, k_lower1, k_higher1));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, k_lower1, k_higher1));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower1, k_higher1));

		if (!(Label::cmp(out[i].label, target, k_lower1, k_higher1))) {
			right = false;
			wrong++;
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(),1u<<3);
	EXPECT_LT(out.load(),1u<<7);
}



int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
