#include <gtest/gtest.h>
#include <cstdint>
#include <bitset>

#include "element.h"
#include "matrix/fq_matrix.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t G_k = 10;
constexpr uint32_t G_l = 5;
constexpr uint32_t G_n = 20;


TEST(Internals, Size) {
	using BinaryValue     = BinaryContainer<G_k + G_l>;
	using BinaryLabel     = BinaryContainer<G_n - G_k>;
	using BinaryMatrix    = FqMatrix<uint64_t, G_k + G_l, G_n-G_k, 2>;
	using BinaryElement   = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;

	BinaryElement e;
	BinaryValue v;
	BinaryLabel l;

	BinaryContainer<G_k + G_l> c1;
	BinaryContainer<G_n - G_k> c2;

	std::cout << "k+l: " << G_k + G_l << "\n";
	std::cout << "n+k: " << G_n - G_k << "\n";

	std::cout << "Size Element: " << sizeof(e) << "\n";
	std::cout << "Size Value:   " << sizeof(e.get_value()) << "\n";
	std::cout << "Size Value:   " << sizeof(v) << "\n";

	std::cout << "Size Label:   " << sizeof(e.get_label()) << "\n";
	std::cout << "Size Label:   " << sizeof(l) << "\n";

	std::cout << "Size CValue:  " << sizeof(c1) << "\n";
	std::cout << "Size CLabel:  " << sizeof(c2) << "\n";

	std::cout << "Size CValueD: " << sizeof(c1.data()) << "\n";
	std::cout << "Size CLabelD: " << sizeof(c2.data()) << "\n";

	std::cout << "PointerOffset:   " << e.label_ptr() - e.value_ptr() << "\n";
}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
