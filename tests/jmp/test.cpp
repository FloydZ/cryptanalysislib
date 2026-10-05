#include <gtest/gtest.h>

#include "jmp/jmp.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using namespace cryptanalysislib::jmp;

constexpr auto expect = [](bool cond) {
     if (not cond) { void failed(); failed(); }
};

TEST(jmp, array) {
    {
      internal::array<int, 1u> array{42};
      expect(1u == array.size());
      expect(42 == array[0u]);
    }
    
    {
      internal::array<uint32_t, 2u> array{4u, 2u};
      expect(2u == array.size());
      expect(4u == array[0u]);
      expect(2u == array[1u]);
    }
}

TEST(jmp, static_branch) {
    {
      static_assert(1u == sizeof(static_branch<bool>));
      static_assert(1u == sizeof(static_branch<uint8_t , 0, 2>));
      static_assert(1u == sizeof(static_branch<uint16_t, 0, 3>));
      static_assert(1u == sizeof(static_branch<uint32_t, 1, 4>));
      static_assert(1u == sizeof(static_branch<uint64_t, 2, 7>));
      static_assert(1u == sizeof(static_branch<size_t, 1, 5>));
    
      static_assert(not [](auto... ts) { return requires { static_branch<bool>{ts...}; }; }());
      static_assert(not [](auto... ts) { return requires { static_branch<bool>{ts...}; }; }(static_branch<bool>{false}));
      static_assert([](auto value) { return requires { static_branch<bool>{value}; }; }(false));
    
      static_assert(not [](auto... ts) { return requires { static_branch<uint32_t, 0, 2>{ts...}; }; }());
      static_assert(not [](auto... ts) { return requires { static_branch<uint32_t, 0, 3>{ts...}; }; }(static_branch<uint32_t, 0, 3>{0u}));
      static_assert([](auto value) { return requires { static_branch<uint32_t, 1, 5>{value}; }; }(0u));
    }
}


TEST(jmp, static_branch2) {
    constexpr auto expect = [](bool cond) {
      if (not cond) { __builtin_abort(); }
    };
    
    static constexpr static_branch<bool> b = false;
    
    auto fn_bool = [&] {
      if (b) {
        return 42;
      } else {
        return 0;
      }
    };

    static constexpr static_branch<int, 0, 2> i = 0;

    auto fn_int = [&] {
      switch (i) {
        default: return 0;
        case 0:  return 42;
        case 1:  return 99;
        case 2:  return 123;
      }
    };

    (void)init();

    {
      expect(0 == fn_bool());

      b = false;
      expect(0 == fn_bool());

      b = true;
      expect(42 == fn_bool());
    }

    {
      expect(42 == fn_int());

      i = 0;
      expect(42 == fn_int());

      i = 1;
      expect(99 == fn_int());

      i = 2;
      expect(123 == fn_int());
    }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
