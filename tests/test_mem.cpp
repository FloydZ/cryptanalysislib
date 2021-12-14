#include <gtest/gtest.h>
#include <cstdint>

#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace std;

TEST(Mem, Rainer_Beschwert_Sich) {

    std::vector<Element *> v;
    for (uint64_t i = 0; i < (1ull << 20); ++i) {
        Element *e = new Element;
        e->get_value().data()[0] = i;
        v.push_back(e);
    }

}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
