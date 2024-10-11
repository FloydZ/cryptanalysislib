#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>

#include "list/list.h"
#include "container/hashmap.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u<<8u;

constexpr static uint32_t n = 32;
using MatrixT = uint8_t;
using Matrix= FqMatrix<uint64_t, n, n, 2>;
using Value = FqPackedVector<n, 2, MatrixT>;
using Label = FqPackedVector<n, 2, MatrixT>;
using Element= Element_T<Value, Label, Matrix>;
using List = List_T<Element>;

TEST(List, hashmap_simple) {
	List L{LS, 1};
	Matrix m;
	m.random();
	L.random(LS, m);
	const auto e = L[0];

	const uint32_t k_lower=0,k_upper=10;
	L.sort_level(k_lower, k_upper);
	for (auto f = L.begin(e,k_lower,k_upper); f != L.end(e,k_lower,k_upper); f++) {
		std::cout << *f << std::endl;
	}

	constexpr uint32_t bucketsize = 10;
	constexpr static Hash<uint32_t, k_lower, k_upper, 2> hashclass{};
	constexpr static SimpleHashMapConfig s = SimpleHashMapConfig{bucketsize, 1u << k_upper, 1};
	using HM = SimpleHashMap<uint32_t, uint32_t, s, Hash<uint32_t, k_lower, k_upper, 2>>;
	HM hm = HM{};

	for (size_t i = 0; i < L.size(); i++) {
		const uint32_t data = *(L[i].label.ptr());
		hm.insert(data, i);
	}

	const uint32_t ff = *(e.label.ptr());
	for (auto f = hm.begin(ff); f != hm.end(ff); f++) {
		std::cout << *f << std::endl;
	}


}
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
