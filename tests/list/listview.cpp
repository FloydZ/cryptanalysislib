#include <gtest/gtest.h>

#include "list/list.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u<<8u;

constexpr static uint32_t n = 32;
using MatrixT = uint8_t;
using Matrix = FqMatrix<uint64_t, n, n, 2>;
using Value = FqPackedVector<n, 2, MatrixT>;
using Label = FqPackedVector<n, 2, MatrixT>;
using Element = Element_T<Value, Label, Matrix>;
using List = List_T<Element>;

static_assert(ListAble<List>);

TEST(Listview, mutable_view) {
	List L{LS, 1};
	Matrix m;
	m.random();
	L.random(LS, m);

	constexpr size_t start = 10, end = 42;
	Listview_t<List> v{&L, start, end};
	EXPECT_EQ(v.size(), end - start);
	for (size_t i = 0; i < v.size(); ++i) {
		EXPECT_EQ(&v[i], &L[start + i]);
	}

	// writes go through to the list
	v[0] = L[0];
	EXPECT_TRUE(L[start].is_equal(L[0]));

	// full and empty views
	Listview_t<List> full{&L, 0, L.size()};
	EXPECT_EQ(full.size(), L.size());
	Listview_t<List> empty{&L, 5, 5};
	EXPECT_EQ(empty.size(), 0);
}

TEST(Listview, const_view) {
	List L{LS, 1};
	Matrix m;
	m.random();
	L.random(LS, m);

	const List &cL = L;
	constexpr size_t start = 3, end = 17;
	const Listview_const_t<List> v{&cL, start, end};
	EXPECT_EQ(v.size(), end - start);
	for (size_t i = 0; i < v.size(); ++i) {
		EXPECT_EQ(&v[i], &cL[start + i]);
	}
}

#ifdef __cpp_explicit_this_parameter
TEST(Listview, view) {
	List L{LS, 1};
	Matrix m;
	m.random();
	L.random(LS, m);

	auto v = L.view(4, 20);
	static_assert(std::is_same_v<decltype(v), Listview_t<List>>);
	EXPECT_EQ(v.size(), 16);
	EXPECT_EQ(&v[0], &L[4]);

	const List &cL = L;
	const auto cv = cL.view(4, 20);
	static_assert(std::is_same_v<std::remove_const_t<decltype(cv)>, Listview_const_t<List>>);
	EXPECT_EQ(cv.size(), 16);
	EXPECT_EQ(&cv[15], &L[19]);
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
