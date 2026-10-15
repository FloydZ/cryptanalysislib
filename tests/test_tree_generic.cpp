#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "element.h"
#include "list/list.h"
#include "matrix/matrix.h"
#include "tree.h"

using ::testing::InitGoogleTest;

// NOTE: every level matches on `w` coordinates and the base lists have
// 	q**w elements, so ~q**w elements are expected on every level
constexpr uint32_t nb = 36, wb = 9;
using BinaryValue   = BinaryVector<nb>;
using BinaryMatrix  = FqMatrix<uint64_t, nb, nb, 2>;
using BinaryList    = List_T<Element_T<BinaryValue, BinaryValue, BinaryMatrix>>;
using BinaryTree    = Tree_T<BinaryList>;

constexpr uint32_t n3 = 20, w3 = 5;
using F3Value       = FqNonPackedVector<n3, 3, uint8_t>;
using F3Matrix      = FqMatrix<uint8_t, n3, n3, 3>;
using F3List        = List_T<Element_T<F3Value, F3Value, F3Matrix>>;
using F3Tree        = Tree_T<F3List>;

/// \return {0, w, 2w, ..., d*w}
static std::vector<uint32_t> limits(const uint32_t d, const uint32_t w) {
	std::vector<uint32_t> lta(d + 1);
	for (uint32_t i = 0; i <= d; ++i) { lta[i] = i * w; }
	return lta;
}

/// checks, that every output is a solution on [0, lta.back()), and
/// that the number of outputs is in [expected/4, 4*expected)
template<typename List, typename Matrix, typename Label>
static void check(const List &out, const Matrix &A, const Label &target,
                  const uint32_t k_upper, const size_t expected) {
	uint32_t wrong = 0;
	for (size_t i = 0; i < out.load(); ++i) {
		// NOTE: the labels are the target
		if (!out[i].label.is_equal(target, 0, k_upper)) { wrong++; }
		auto e = out[i];
		e.recalculate_label(A);
		if (!e.label.is_equal(target, 0, k_upper)) { wrong++; }
	}

	EXPECT_EQ(wrong, 0u);
	EXPECT_GE(out.load(), expected / 4u);
	EXPECT_LT(out.load(), expected * 4u);
}

template<const uint32_t d, typename Tree, typename List, typename Matrix,
         const uint32_t w>
static void run_generic(const size_t basesize, const size_t expected) {
	Matrix A; A.identity();
	const auto lta = limits(d, w);

	List out{expected * 4u};
	std::vector<List> L(1u << d, List{0});
	for (auto &l : L) { l.random(basesize, A); }

	typename List::LabelType target; target.random();

	Tree t{1, A, 0};
	t.template join_stream_generic<d>(out, L, target, lta);
	check(out, A, target, d * w, expected);
}

TEST(TreeGeneric, binary_d1) { run_generic<1, BinaryTree, BinaryList, BinaryMatrix, wb>(1u << wb, 1u << wb); }
TEST(TreeGeneric, binary_d2) { run_generic<2, BinaryTree, BinaryList, BinaryMatrix, wb>(1u << wb, 1u << wb); }
TEST(TreeGeneric, binary_d3) { run_generic<3, BinaryTree, BinaryList, BinaryMatrix, wb>(1u << wb, 1u << wb); }
TEST(TreeGeneric, binary_d4) { run_generic<4, BinaryTree, BinaryList, BinaryMatrix, wb>(1u << wb, 1u << wb); }

// 3**5 = 243
TEST(TreeGeneric, f3_d1) { run_generic<1, F3Tree, F3List, F3Matrix, w3>(243, 243); }
TEST(TreeGeneric, f3_d2) { run_generic<2, F3Tree, F3List, F3Matrix, w3>(243, 243); }
TEST(TreeGeneric, f3_d3) { run_generic<3, F3Tree, F3List, F3Matrix, w3>(243, 243); }
TEST(TreeGeneric, f3_d4) { run_generic<4, F3Tree, F3List, F3Matrix, w3>(243, 243); }

template<typename Tree, typename List, typename Matrix, const uint32_t w>
static void run_join16lists(const size_t basesize, const size_t expected) {
	Matrix A; A.identity();
	const auto lta = limits(4, w);

	List out{expected * 4u};
	std::vector<List> L(16, List{0});
	for (auto &l : L) { l.random(basesize, A); }

	typename List::LabelType target; target.random();

	Tree t{1, A, 0};
	t.join16lists(out, L, target, lta);
	check(out, A, target, 4 * w, expected);
}

TEST(TreeD4, binary_join16lists) { run_join16lists<BinaryTree, BinaryList, BinaryMatrix, wb>(1u << wb, 1u << wb); }
TEST(TreeD4, f3_join16lists) { run_join16lists<F3Tree, F3List, F3Matrix, w3>(243, 243); }

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
