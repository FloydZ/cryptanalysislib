//#include "math/abs.h"
//#include "math/ipow.h"
//#include "math/root.h"
//#include "math/exp.h"
//#include "math/round.h"
//#include "math/log.h"

#include "container/array.h"

#include "sort/ska_sort.h"
#include "sort/vergesort.h"

//TODO #include "crypto/sha1.h"

#include "alloc/alloc.h"

#include <benchmark/benchmark.h>

#include <random>
#include <deque>
#include <variant>
#include <vector>
#include <array>
#include <tuple>
#include <utility>
#include <iostream>
#include <algorithm>


#define SKA_SORT_NOINLINE __attribute__((noinline))
#define LIMBS 4
//#define SORT_ON_FIRST_ONLY
#define NUM_SORT_KEYS 1

template<size_t Size>
struct SizedStruct{
	uint8_t array[Size] = {0};
};

template<>
struct SizedStruct<0> {};

typedef std::int64_t benchmark_sort_key;
typedef SizedStruct<1016> benchmark_sort_value;
static constexpr int profile_multiplier = 2;
//static constexpr int max_profile_range = 1 << 24;
static constexpr int max_profile_range = 1 << 22;


template<typename It, typename OutIt, typename ExtractKey>
void counting_sort(It begin, It end, OutIt out_begin, ExtractKey && extract_key){
	detail::counting_sort_impl(begin, end, out_begin, extract_key);
}

template<typename It, typename OutIt>
void counting_sort(It begin, It end, OutIt out_begin){
	using detail::to_unsigned_or_bool;
	detail::counting_sort_impl(begin, end, out_begin, [](auto && a){ return to_unsigned_or_bool(a); });
}

template<typename It, typename OutIt, typename ExtractKey>
bool radix_sort(It begin, It end, OutIt buffer_begin, ExtractKey && extract_key) {
	return detail::RadixSorter<typename std::result_of<ExtractKey(decltype(*begin))>::type>::sort(begin, end, buffer_begin, extract_key);
}

//template<typename It, typename OutIt>
//bool radix_sort(It begin, It end, OutIt buffer_begin) {
//    return detail::RadixSorter<decltype(*begin)>::sort(begin, end, buffer_begin, detail::IdentityFunctor());
//}

template<typename It, typename ExtractKey>
static void inplace_radix_sort(It begin, It end, ExtractKey && extract_key) {
	detail::inplace_radix_sort<1, 1>(begin, end, extract_key);
}

template<typename It>
static void inplace_radix_sort(It begin, It end) {
	inplace_radix_sort(begin, end, detail::IdentityFunctor());
}

template<typename It, typename ExtractKey>
static void american_flag_sort(It begin, It end, ExtractKey && extract_key) {
	detail::inplace_radix_sort<1, std::numeric_limits<std::ptrdiff_t>::max()>(begin, end, extract_key);
}

template<typename It>
static void american_flag_sort(It begin, It end) {
	american_flag_sort(begin, end, detail::IdentityFunctor());
}


static std::vector<std::vector<uint64_t>> SKA_SORT_NOINLINE create_radix_sort_data(std::mt19937_64 & randomness, size_t size) {
	std::vector<std::vector<uint64_t>> result;
	std::uniform_int_distribution<uint64_t> random_size(0, 128);
	std::uniform_int_distribution<uint64_t> random(0, std::numeric_limits<uint64_t>::max());
	result.reserve(size);
	for (size_t i = 0; i < size; ++i) {
		// std::vector<uint64_t> to_add(random_size(randomness));
		std::vector<uint64_t> to_add(LIMBS);
		std::iota(to_add.begin(), to_add.end(), random(randomness));
		result.push_back(std::move(to_add));
	}
	return result;
}

static void benchmark_std_sort(benchmark::State & state) {
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		std::sort(to_sort.begin(), to_sort.end(), [](auto && l, auto && r){ return std::get<0>(l) < std::get<0>(r); });
#else
		std::sort(to_sort.begin(), to_sort.end());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

static void benchmark_stable_std_sort(benchmark::State & state) {
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		std::stable_sort(to_sort.begin(), to_sort.end(), [](auto && l, auto && r){ return std::get<0>(l) < std::get<0>(r); });
#else
		std::stable_sort(to_sort.begin(), to_sort.end());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

static void benchmark_ska_sort(benchmark::State & state){
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		ska_sort(to_sort.begin(), to_sort.end(), [](auto && a) -> decltype(auto){ return std::get<0>(a); });
#else
		ska_sort(to_sort.begin(), to_sort.end());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

static void benchmark_inplace_ska_sort(benchmark::State & state){
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		inplace_radix_sort(to_sort.begin(), to_sort.end(), [](auto && a) -> decltype(auto){ return std::get<0>(a); });
#else
		inplace_radix_sort(to_sort.begin(), to_sort.end());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

//static void benchmark_counting_sort(benchmark::State & state){
//    std::mt19937_64 randomness(77342348);
//    create_radix_sort_data(randomness, state.range(0));
//    while (state.KeepRunning()) {
//        auto to_sort = create_radix_sort_data(randomness, state.range(0));
//        benchmark::DoNotOptimize(to_sort.data());
//#ifdef SORT_ON_FIRST_ONLY
//        counting_sort(to_sort.begin(), to_sort.end(), [](auto && a) -> decltype(auto){ return std::get<0>(a); });
//#else
//        counting_sort(to_sort.begin(), to_sort.end());
//        assert(std::is_sorted(to_sort.begin(), to_sort.end()));
//#endif
//        benchmark::ClobberMemory();
//    }
//}

static void benchmark_american_flag_sort(benchmark::State & state){
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		american_flag_sort(to_sort.begin(), to_sort.end(), [](auto && a) -> decltype(auto){ return std::get<0>(a); });
#else
		american_flag_sort(to_sort.begin(), to_sort.end());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

static void benchmark_verge_sort(benchmark::State & state){
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		vergesort::vergesort(to_sort.begin(), to_sort.end(), [](auto && a) -> decltype(auto){ return std::get<0>(a); });
#else
		vergesort::vergesort(to_sort.begin(), to_sort.end());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

static void benchmark_insertion_sort(benchmark::State & state){
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		vergesort::insertionsort(to_sort.begin(), to_sort.end(), [](auto && a) -> decltype(auto){ return std::get<0>(a); });
#else
		vergesort::detail::insertion_sort(to_sort.begin(), to_sort.end(), std::less<decltype(to_sort[0])>());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

static void benchmark_quick_sort(benchmark::State & state){
	std::mt19937_64 randomness(77342348);
	create_radix_sort_data(randomness, state.range(0));
	while (state.KeepRunning()) {
		auto to_sort = create_radix_sort_data(randomness, state.range(0));
		const size_t size = to_sort.size();
		benchmark::DoNotOptimize(to_sort.data());
#ifdef SORT_ON_FIRST_ONLY
		vergesort::detail::quicksort(to_sort.begin(), to_sort.end(), [](auto && a) -> decltype(auto){ return std::get<0>(a); });
#else
		vergesort::detail::quicksort(to_sort.begin(), to_sort.end(), size, std::less<decltype(to_sort[0])>());
		assert(std::is_sorted(to_sort.begin(), to_sort.end()));
#endif
		benchmark::ClobberMemory();
	}
}

BENCHMARK(benchmark_std_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);
BENCHMARK(benchmark_ska_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);
BENCHMARK(benchmark_inplace_ska_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);
//BENCHMARK(benchmark_counting_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);
BENCHMARK(benchmark_american_flag_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);
BENCHMARK(benchmark_verge_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);
BENCHMARK(benchmark_insertion_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);
BENCHMARK(benchmark_quick_sort)->RangeMultiplier(profile_multiplier)->Range(profile_multiplier, max_profile_range);

constexpr uint32_t doubleit(uint32_t a) {
	return a*a;
}


//bool test_sha1() {
//	char *text[] =
//			{ "",
//			  "a",
//			  "abc",
//			  "message digest",
//			  "abcdefghijklmnopqrstuvwxyz",
//			  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
//			  "12345678901234567890123456789012345678901234567890123456789012345678901234567890"
//			};
//
//	char *SHA1_dgst[] =
//			{ "da39a3ee5e6b4b0d3255bfef95601890afd80709",
//			  "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8",
//			  "a9993e364706816aba3e25717850c26c9cd0d89d",
//			  "c12252ceda8be8994d5fa0290a47231c1d16aae3",
//			  "32d10c7b8cf96570ca04ce37f2a19d84240d3a89",
//			  "761c457bf73b14d27e9e9265c46f4b4dda11f940",
//			  "50abf5706a150990a08b2c5ea40fa0e585554732"
//			};
//
//	bool ret = true;
//	uint8_t dgst[cryptanalysislib::internal::SHA1_DIGEST_LEN+1],
//			tv[cryptanalysislib::internal::SHA1_DIGEST_LEN+1];
//
//	for (uint32_t i = 0; i < sizeof(text)/sizeof(char*); i++) {
//		// TODO not constexpr cryptanalysislib::sha1(dgst, (uint8_t *)text[i], strlen(text[i]));
//		hex2bin(tv, SHA1_dgst[i]);
//
//		if (memcmp(dgst, tv, cryptanalysislib::internal::SHA1_DIGEST_LEN) != 0) {
//			printf ("\nFailed for string : \"%s\"", text[i]);
//			ret = false;
//		}
//	}
//
//	return ret;
//}

int main(){
	// allocator
	auto stackAllocator = StackAllocator<256>();
	auto stackAllocator_ptr = stackAllocator.allocate(277);
	std::cout << stackAllocator_ptr;


	// crypto test
//	uint8_t crypto_in_test[64] = {0};
//	uint8_t crypto_out_test[64] = {0};
//	const size_t crypto_len = sizeof(crypto_in_test);
//	cryptanalysislib::sha1(crypto_out_test, crypto_in_test, crypto_len);
//	test_sha1();

	// container crypt
//	cryptanalysislib::array<uint32_t, 10> v{}, w{};
//	transform(v, doubleit);
//
//	std::cout << v[0] << "\n";
//
//	for (auto a :v) {
//		std::cout << a << "\n";
//	}
//
//	std::cout << std::get<0>(v);


//  math test
//	std::cout << abs(-0.2) << "\n";
//	std::cout << ipow(0.2, 2) << "\n";
//	std::cout << sqrt(9) << "\n";
//	std::cout << exp(9) << "\n";
//	std::cout << log(e()) << "\n";
//	std::cout << exp(log(10.)) << "\n";
//	return 0;
}

//BENCHMARK_MAIN();
