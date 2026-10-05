#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "atomic/atomic_primitives.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

struct likely_padded {
    std::uint8_t c;
    std::uint16_t st;
    std::uint32_t i;
};

TEST(primitives, cas) {
	uint32_t val = 0;
	CAS(&val, &val, 1);
	EXPECT_EQ(val, 1);
}



TEST(__maybe_has_padding, simple) {
    constexpr bool t1 = __atomic_impl::__maybe_has_padding<uint32_t>();
    EXPECT_EQ(t1, false);
    constexpr bool t2 = __atomic_impl::__maybe_has_padding<uint8_t>();
    EXPECT_EQ(t2, false);
    constexpr bool t3 = __atomic_impl::__maybe_has_padding<double>();
    EXPECT_EQ(t3, false);
    constexpr bool t4 = __atomic_impl::__maybe_has_padding<likely_padded>();
    EXPECT_EQ(t4, true);
}


TEST(__clear_padding, simple) {
    likely_padded a;
    likely_padded *p = &a;
    
    auto *t = __atomic_impl::__clear_padding(p);
    EXPECT_EQ(*t, p);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
