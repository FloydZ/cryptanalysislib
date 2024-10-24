#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "combination/chase.h"
#include "container/hashmap.h"
#include "container/binary_packed_vector.h"
#include "hash/simple.h"
#include "helper.h"
#include "list/enumeration/enumeration.h"
#include "list/list.h"
#include "matrix/matrix.h"
#include "random.h"

#include "algorithm/bits/popcount.h"

using namespace cryptanalysislib;
using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t n = 10;
constexpr uint32_t l = 10;
constexpr uint32_t q = 2;
constexpr uint32_t w = 3;

constexpr size_t list_size = compute_combinations_fq_chase_list_size<n, q, w>();

using T = uint64_t;
using Value = BinaryVector<n, T>;
using Label = BinaryVector<n, T>;
using Matrix = FqMatrix<T, n, n, q, true>;
using Element = Element_T<Value, Label, Matrix>;
using List = List_T<Element>;

using K = uint16_t;
using V = size_t[1];

constexpr static SimpleHashMapConfig simple{10, 1u << l, 1};
using HMType = SimpleHashMap<K, V, simple, Hash<K, 0, l, q>>;
using load_type = HMType::load_type;

TEST(Enum, p2) {
	constexpr int nn = 30;
	constexpr int p = 2;
	enumerate_t<nn, p> c;

	uint32_t ctr = 0;

	c.enumerate([&](uint16_t *p1) {
		uint32_t x = 0;
		x ^= (1u << p1[0]);
		x ^= (1u << p1[1]);
		// print_binary(x, nn, "");
		// std::cout << ": " << p1[0] << " " << p1[1] << std::endl;

		EXPECT_EQ(__builtin_popcountll(x), p);
		ctr += 1;
	});

	EXPECT_EQ(ctr, bc(nn, p));
}

TEST(Enum, p3) {
	constexpr int nn = 30;
	constexpr int p = 3;
	enumerate_t<nn, p> c;

	uint32_t ctr = 0;

	c.enumerate([&](uint16_t *p1) {
		uint32_t x = 0;
		x ^= (1u << p1[0]);
		x ^= (1u << p1[1]);
		x ^= (1u << p1[2]);
		// print_binary(x, nn, "");
		// std::cout << ": " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
		EXPECT_EQ(__builtin_popcountll(x), p);
		ctr += 1;
	});

	EXPECT_EQ(ctr, bc(nn, p));
}

TEST(Chase, p1) {
	constexpr int nn = 30;
	constexpr int p = 1;
	chase_t<nn, p> c;

	uint32_t ctr = 1;
	uint32_t x = 1u;
	uint16_t rows[p] = {0};
	//print_binary(x, nn, "");

	c.enumerate([&](uint16_t p1, uint16_t p2) {
		c.biject(ctr, rows);
		x ^= (1u << p1);
		x ^= (1u << p2);
		print_binary(x, nn, "");
		std::cout << ": " << p1 << " " << p2 << " | " << rows[0] << std::endl;

		EXPECT_EQ(__builtin_ctz(x), rows[0]);
		ctr += 1;
	});

	EXPECT_EQ(ctr, bc(nn, p));
}

TEST(Chase, p2) {
	constexpr int nn = 30;
	constexpr int p = 2;
	chase_t<nn, p> c;

	uint32_t ctr = 1;
	uint32_t x = 3u;
	uint16_t rows[p] = {0};
	//print_binary(x, nn, "");

	c.enumerate([&](uint16_t p1, uint16_t p2) {
		c.biject(ctr, rows);
		x ^= (1u << p1);
		x ^= (1u << p2);
		//print_binary(x, nn, "");
		//std::cout << ": " << p1 << " " << p2 << " | " << rows[0] << ":" << rows[1] << std::endl;

		uint32_t y = x;
		for (uint32_t i = 0; i < p; ++i) {
			const uint32_t ctz = __builtin_ctz(y);
			EXPECT_EQ(true, (ctz == rows[0]) || (ctz == rows[1]));
			y ^= 1u << ctz;
		}
		ctr += 1;
	});

	EXPECT_EQ(ctr, bc(nn, p));
}

TEST(Chase, p3) {
	constexpr int nn = 10;
	constexpr int p = 3;
	chase_t<nn, p> c;
	uint32_t ctr = 1;
	uint32_t x = 7u;
	uint16_t rows[p] = {0};
	//print_binary(x, nn, "");

	c.enumerate([&](uint16_t p1, uint16_t p2) {
		c.biject(ctr, rows);
		x ^= (1u << p1);
		x ^= (1u << p2);
		print_binary(x, nn, "");
		std::cout << ": " << p1 << " " << p2 << " | " << rows[0] << ":" << rows[1] << ":" << rows[2] << ", " << ctr << std::endl;

		uint32_t y = x;
		for (uint32_t i = 0; i < p; ++i) {
			const uint32_t ctz = __builtin_ctz(y);
			//EXPECT_EQ(true, (ctz == rows[0]) || (ctz == rows[1]) || (ctz == rows[2]));
			y ^= 1u << ctz;
		}
		ctr += 1;
	});

	EXPECT_EQ(ctr, bc(nn, p));
}

TEST(Chase, first) {
	constexpr uint32_t element_limbs = (n + 63) / 64;
	uint16_t pos1=0, pos2=0;
	uint16_t epos1, epos2;
	Combinations_Binary_Chase<T, n, w> c;

	uint64_t *w1 = (uint64_t *) malloc(element_limbs * sizeof(uint64_t)),
	         *w2 = (uint64_t *) malloc(element_limbs * sizeof(uint64_t));

	c.left_step(w2, &epos1, &epos2);
	for (size_t i = 0; i < list_size; ++i) {
		cryptanalysislib::memcpy(w1, w2, element_limbs);
		bool ret = c.left_step(w2, &epos1, &epos2);
		if (i != (list_size - 1)) {
			EXPECT_EQ(true, ret);
		}

		Combinations_Binary_Chase<T, n, w>::__diff(w1, w2, element_limbs, &pos1, &pos2);

		int diff = (int) pos1 - int(pos2);
		EXPECT_LE(std::abs(diff), w);
		int dif2 = (int) epos1 - int(epos2);
		EXPECT_LE(std::abs(dif2), w);
		bool v1 = pos1 == epos1 or pos1 == epos2;
		bool v2 = pos2 == epos2 or pos2 == epos1;
		EXPECT_EQ(true, v1);
		if (i != (list_size - 1)) {
			EXPECT_EQ(true, v2);
		}

		// std ::cout << epos1 << ":" << epos2 << " " << pos1 << ":" << pos2 << std::endl;
		print_binary(w2, n);

		uint32_t popc = 0;
		for (uint32_t j = 0; j < element_limbs; ++j) {
			popc += popcount::popcount<T>(w2[j]);
		}

		if (i < list_size - 1) {
			EXPECT_EQ(popc, w);
		}
	}

	free(w1);
	free(w2);
}

TEST(F2, single_list) {
	List L(list_size);
	Matrix HT;
	HT.random();
	BinaryListEnumerateMultiFullLength<List, n, w> enumerator{HT};
	enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L, nullptr);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L.data_value(i).popcnt(), w);
	}
}

TEST(F2, single_hashmap) {
	HMType hm;
	List L(list_size);
	auto extractor = [](const Label l) {
		return l.ptr()[0];
	};

	Matrix HT;
	HT.random();

	BinaryListEnumerateMultiFullLength<List, n, w> enumerator{HT};
	enumerator.run<HMType, decltype(extractor), std::nullptr_t>(&L, nullptr, 0, 0, 0, &hm, &extractor, nullptr);

	for (size_t i = 0; i < list_size; ++i) {
		const auto data = extractor(L.data_label(i));
		load_type load = 0;
		const auto pos = hm.find(data, load);

		// make sure we found something
		ASSERT_NE(pos, size_t(-1));
	}
}

TEST(F2, two_lists) {
	constexpr size_t list_size = compute_combinations_fq_chase_list_size<n / 2, q, w>();
	List L1(list_size);
	List L2(list_size);

	Matrix HT;
	HT.random();

	BinaryListEnumerateMultiFullLength<List, n / 2, w> enumerator{HT};
	enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n / 2);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).popcnt(), w);
		EXPECT_EQ(L2.data_value(i).popcnt(), w);

		for (uint32_t j = 0; j < n / 2; j++) {
			EXPECT_EQ(L1.data_value(i).get(j + n / 2), 0);
			EXPECT_EQ(L2.data_value(i).get(j), 0);
		}

		EXPECT_EQ(L1.data_label(i).is_zero(), false);
		EXPECT_EQ(L2.data_label(i).is_zero(), false);
	}
}

int main(int argc, char **argv) {
	rng_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
