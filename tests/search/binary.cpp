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

	 auto a = upper_bound_standard_binary_search(data.begin(), data.end(), search,
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

	auto a = upper_bound_standard_binary_search(data.begin(), data.end(), search,
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

	auto a = upper_bound_standard_binary_search(data.begin(), data.end(), search,
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

	auto a = lower_bound_standard_binary_search(data.begin(), data.end(), search,
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

	auto a = lower_bound_standard_binary_search(data.begin(), data.end(), search,
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

	const auto b = monobound_binary_search(data.data(), data.size(), search);
	auto a = lower_bound_monobound_binary_search(data.begin(), data.end(), search,
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

	const auto b = monobound_binary_search(data.data(), data.size(), search);
	auto a = lower_bound_monobound_binary_search(data.begin(), data.end(), search,
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

	const auto b = monobound_binary_search(data.data(), data.size(), search);
	auto a = lower_bound_monobound_binary_search(data.begin(), data.end(), search,
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

	const auto b = monobound_binary_search(data.data(), data.size(), search);
	auto a = tripletapped_binary_search(
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
	size_t a = tripletapped_binary_search(data.data(), SIZE, search);
	EXPECT_EQ(solution_index, a);
}

TEST(monobound_quaternary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data<std::vector<T>, T>(data, solution_index, SIZE, 1);

	/// note mask not working
	size_t a = monobound_quaternary_search(data.data(), SIZE, search);
	EXPECT_EQ(solution_index, a);
}

TEST(branchless_lower_bound_cmp, karylist_simple) {
	List data{list_size};
	Element dummy;
	size_t solution_index;
	const Element search = cryptanalysislib::random_data<List, Element>(data, solution_index, SIZE, 1, dummy);

	auto a = branchless_lower_bound(data.begin(), data.end(), search,
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

	auto a = branchless_lower_bound(data.begin(), data.end(), search,
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

	auto a = branchless_lower_bound(data.begin(), data.end(), search,
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

	auto a = branchless_lower_bound(data.begin(), data.end(), search,
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

	auto a = branchless_lower_bound(data.begin(), data.end(), search,
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

	auto a = branchless_lower_bound(data.begin(), data.end(), search,
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

	auto a = cryptanalysislib::search::internal::binary_search_dispatch(data.begin(), data.end(), search,
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

	auto a = cryptanalysislib::search::internal::binary_search_dispatch(data.begin(), data.end(), search,
		[](const T &e1) __attribute__((always_inline)){
		  return e1 & MASK;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}
int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
