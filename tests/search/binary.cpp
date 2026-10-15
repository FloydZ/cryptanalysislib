#include <gtest/gtest.h>

#include "random.h"
#include "search/search.h"
#include "container/fq_vector.h"
#include "list/list.h"
#include "matrix/matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

using T = uint32_t;
constexpr static uint64_t SIZE = 1<<10;

// Where to search
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 22;
constexpr static T MASK = ((T(1) << k_higher) - 1) ^ ((T(1) << k_lower) -1);
constexpr static size_t nr_sols = 5;


constexpr uint32_t k = 100;
constexpr uint32_t n = 100;
constexpr uint32_t q = 5;
constexpr size_t list_size = 1u << 10;

using MatrixT = uint64_t;
using Matrix = FqMatrix<MatrixT, n, k, q>;
using Value = FqPackedVector<k, q, MatrixT>;
using Label = FqPackedVector<n, q, MatrixT>;
using Element = Element_T<Value, Label, Matrix>;
using List = List_T<Element>;

TEST(upper_bound_standard_binary_search, kAryList) {
	List data{list_size};
	Element dummy;
	size_t solution_index;
	const Element search = cryptanalysislib::random_data<List, Element>(data, solution_index, SIZE, 1, dummy);

	 auto a = cryptanalysislib::upper_bound_standard_binary_search(data.begin(), data.end(), search,
		[](const Element &e1) {
		  return e1.hash();
		}
	 );

	 EXPECT_EQ(solution_index, std::distance(data.begin(), a));
}

TEST(upper_bound_standard_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	auto a = cryptanalysislib::upper_bound_standard_binary_search(data.begin(), data.end(), search,
		[](const T &e1) -> T {
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index, std::distance(data.begin(), a));
}

TEST(upper_bound_standard_binary_search, multiple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = cryptanalysislib::random_data(data, solution_index, SIZE, nr_sols, MASK);

	auto a = cryptanalysislib::upper_bound_standard_binary_search(data.begin(), data.end(), search,
		[](const T &e1) -> T {
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index+ nr_sols - 1, std::distance(data.begin(), a));
}
TEST(lower_bound_standard_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	auto a = cryptanalysislib::lower_bound_standard_binary_search(data.begin(), data.end(), search,
		[](const T &e1) -> T {
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_standard_binary_search, multiple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = cryptanalysislib::random_data(data, solution_index, SIZE, nr_sols, MASK);

	auto a = cryptanalysislib::lower_bound_standard_binary_search(data.begin(), data.end(), search,
		[](const T &e1) -> T {
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(upper_bound_monobound_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	const auto b = cryptanalysislib::monobound_binary_search(data.data(), data.size(), search);
	auto a = cryptanalysislib::lower_bound_monobound_binary_search(data.begin(), data.end(), search,
		 [](const T &e1) -> T {
		   return e1 & MASK;
		 }
	);

	EXPECT_EQ(solution_index, b);
	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(upper_bound_monobound_binary_search, multiple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = cryptanalysislib::random_data(data, solution_index, SIZE, nr_sols, MASK);

	const auto b = cryptanalysislib::monobound_binary_search(data.data(), data.size(), search);
	auto a = cryptanalysislib::lower_bound_monobound_binary_search(data.begin(), data.end(), search,
		 [](const T &e1) -> T {
		   return e1 & MASK;
		 }
	);

	EXPECT_EQ(solution_index + nr_sols -1u, b);
	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_monobound_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	const auto b = cryptanalysislib::monobound_binary_search(data.data(), data.size(), search);
	auto a = cryptanalysislib::lower_bound_monobound_binary_search(data.begin(), data.end(), search,
	     [](const T &e1) -> T {
		     return e1 & MASK;
	     }
	);

	EXPECT_EQ(solution_index, b);
	EXPECT_EQ(solution_index, distance(data.begin(), a));
}


TEST(iterator_tripletapped_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	const auto b = cryptanalysislib::monobound_binary_search(data.data(), data.size(), search);
	auto a = cryptanalysislib::tripletapped_binary_search(
	        data.begin(), data.end(), search,
		[](const T &e1) -> T {
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index, b);
	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(tripletapped_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data<std::vector<T>, T>(data, solution_index, SIZE, 1);

	/// NOTE MASK not working
	size_t a = cryptanalysislib::tripletapped_binary_search(data.data(), SIZE, search);
	EXPECT_EQ(solution_index, a);
}

TEST(monobound_quaternary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data<std::vector<T>, T>(data, solution_index, SIZE, 1);

	/// note mask not working
	size_t a = cryptanalysislib::monobound_quaternary_search(data.data(), SIZE, search);
	EXPECT_EQ(solution_index, a);
}

TEST(branchless_lower_bound_cmp, karylist_simple) {
	List data{list_size};
	Element dummy;
	size_t solution_index;
	const Element search = cryptanalysislib::random_data<List, Element>(data, solution_index, SIZE, 1, dummy);

	auto a = cryptanalysislib::branchless_lower_bound(data.begin(), data.end(), search,
		[](const Element &e1, const Element &e2) -> bool {
		  return e1.hash() < e2.hash();
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(branchless_lower_bound_cmp, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	auto a = cryptanalysislib::branchless_lower_bound(data.begin(), data.end(), search,
		 [](const T &e1, const T &e2) -> T {
		   return (e1&MASK) < (e2&MASK);
		 }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(branchless_lower_bound_cmp, karylist_multiple) {
	List data{list_size};
	Element dummy;
	size_t solution_index;
	const Element search = cryptanalysislib::random_data<List, Element>(data, solution_index, SIZE, nr_sols, dummy);

	auto a = cryptanalysislib::branchless_lower_bound(data.begin(), data.end(), search,
		[](const Element &e1, const Element &e2) {
		  return e1.hash() < e2.hash();
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(branchless_lower_bound_cmp, multiple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, nr_sols, MASK);

	auto a = cryptanalysislib::branchless_lower_bound(data.begin(), data.end(), search,
		[](const T &e1, const T &e2) -> T {
		  return (e1&MASK) < (e2&MASK);
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(branchless_lower_bound, karylist_simple) {
	List data{list_size};
	Element dummy;
	size_t solution_index;
	const Element search = cryptanalysislib::random_data<List, Element>(data, solution_index, SIZE, 1, dummy);

	auto a = cryptanalysislib::branchless_lower_bound(data.begin(), data.end(), search,
		[](const Element &e1) {
		  return e1.hash();
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(branchless_lower_bound, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	auto a = cryptanalysislib::branchless_lower_bound(data.begin(), data.end(), search,
		[](const T &e1) {
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(binary_search_dispatch, compare) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	auto a = cryptanalysislib::internal::binary_search_dispatch(data.begin(), data.end(), search,
		[](const T &e1, const T &e2) {
		  return e1 < e2;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(binary_search_dispatch, hash) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, 1, MASK);

	auto a = cryptanalysislib::internal::binary_search_dispatch(data.begin(), data.end(), search,
		[](const T &e1) __attribute__((always_inline)){
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}
TEST(binary_search, found_and_not_found) {
	// sorted, with duplicates and gaps
	const std::vector<T> data{1, 3, 3, 3, 7, 9, 12};
	const auto cmp = [](const T &a, const T &b) { return a < b; };
	const auto hsh = [](const T &a) { return a; };
	for (T v = 0; v < 14; ++v) {
		// expected: first equal element or end
		auto e = data.end();
		for (auto it = data.begin(); it != data.end(); ++it) {
			if (*it == v) { e = it; break; }
		}

		EXPECT_EQ(e, cryptanalysislib::binary_search(data.begin(), data.end(), v));
		EXPECT_EQ(e, cryptanalysislib::binary_search(data.begin(), data.end(), v, cmp));
		EXPECT_EQ(e, cryptanalysislib::binary_search(data.begin(), data.end(), v, hsh));
	}

	const std::vector<T> empty;
	EXPECT_EQ(empty.end(), cryptanalysislib::binary_search(empty.begin(), empty.end(), T(1)));
}

TEST(lower_bound_multiple, random) {
	constexpr size_t N = 5;
	for (uint32_t k = 0; k < 200; ++k) {
		std::vector<uint32_t> v[N];
		const uint32_t *arrays[N];
		size_t lengths[N], out[N];
		uint32_t keys[N];
		for (size_t i = 0; i < N; ++i) {
			const size_t len = rng() % 100u;
			v[i].resize(len);
			for (auto &x : v[i]) { x = rng() % 64u; }
			cryptanalysislib::sort(v[i].begin(), v[i].end());
			arrays[i] = v[i].data();
			lengths[i] = len;
			keys[i] = rng() % 70u;
		}

		cryptanalysislib::lower_bound_multiple<N>(arrays, lengths, keys, out);
		for (size_t i = 0; i < N; ++i) {
			size_t expect = 0;
			while ((expect < lengths[i]) && (v[i][expect] < keys[i])) { expect += 1; }
			EXPECT_EQ(out[i], expect);
		}
	}

	// usable at compile time
	constexpr uint32_t a[4] = {1, 3, 5, 7}, b[2] = {2, 2};
	constexpr auto r = [&]() {
		const uint32_t *arr[2] = {a, b};
		const size_t len[2] = {4, 2};
		const uint32_t keys[2] = {5, 3};
		std::array<size_t, 2> o{};
		cryptanalysislib::lower_bound_multiple<2>(arr, len, keys, o.data());
		return o;
	}();
	static_assert(r[0] == 2 && r[1] == 2);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
