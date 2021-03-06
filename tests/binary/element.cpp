#include <gtest/gtest.h>
#include <cstdint>
#include <bitset>

// Hack for testing private functions (C++ god)
#define private public

#include "element.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t G_k = 10;
constexpr uint32_t G_l = 5;
TEST(Internals, Size) {
	using BinaryValue     = Value_T<BinaryContainer<G_k + G_l>>;
	using BinaryLabel     = Label_T<BinaryContainer<n - G_k>>;
	using BinaryMatrix    = mzd_t *;
	using BinaryElement   = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;

	BinaryElement e;
	BinaryValue v;
	BinaryLabel l;

	BinaryContainer<G_k + G_l> c1;
	BinaryContainer<n - G_k> c2;

	std::cout << "k+l: " << G_k + G_l << "\n";
	std::cout << "n+k: " << n - G_k << "\n";

	std::cout << "Size Element: " << sizeof(e) << "\n";
	std::cout << "Size Value:   " << sizeof(e.get_value()) << "\n";
	std::cout << "Size Value:   " << sizeof(v) << "\n";

	std::cout << "Size Label:   " << sizeof(e.get_label()) << "\n";
	std::cout << "Size Label:   " << sizeof(l) << "\n";

	std::cout << "Size CValue:  " << sizeof(c1) << "\n";
	std::cout << "Size CLabel:  " << sizeof(c2) << "\n";

	std::cout << "Size CValueD: " << sizeof(c1.data()) << "\n";
	std::cout << "Size CLabelD: " << sizeof(c2.data()) << "\n";

	std::cout << "PointerOffset:   " << e.get_label_container_ptr() - e.get_value_container_ptr() << "\n";

}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif