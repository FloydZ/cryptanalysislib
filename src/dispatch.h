#ifndef CRYPTANALYSISLIB_DISPATCH_H
#define CRYPTANALYSISLIB_DISPATCH_H

#include <cstdlib>
#include <cstdint>
#include <functional>

#include "helper.h"
#include "cpucycles.h"

struct BenchmarkConfig {
    constexpr static size_t number_iterations = 1u<<10;
};
constexpr static BenchmarkConfig benchmarkConfig{};

/// translation
/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGISQBspK4AMngMmAByPgBGmMQgABwAzKQADqgKhE4MHt6%2B/kEZWY4CYRHRLHEJKbaY9qUMQgRMxAR5Pn6BdQ05za0E5VGx8UmpCi1tHQXdEwNDldVjAJS2qF7EyOwc5snhyN5YANQmyW7IE/iCp9gmGgCCu/uHmCdnFwTotHgxN3ePZj2DAOXmOpzcADdMA4SH8Hk9gS83m4qF5gY0xHCAUCQWD3iwmAQEFj/rQBMAjmSjEdkGkvMgAJ4HTAKCAQ1B4dDLE4Adis8I0AE40VlgBF0JTyZLqcQWV5aARTvzHkKmAoWEd2QZHPQIOYzAA6GIMgivLgAViVRpNr0kgqVCgQtAAjkcACTJMykcyWszm4joVRKkiun0%2BgOqb1%2B8NMINmMz/O6Co4p1Np1MgE7x04AESY%2BogsoU8oIyxAIH14cD%2BuWSsTQtlBA2DCORZLdfhPJz9fuE2I9IIRwsjGQCAJxAA1h4GDQKSY%2Bf802gGBNMKo0sQjnNHMgt3gAF6YAD6g4YI2IR8I8UJOQUbxzRy4XnB4K4Gg7jy7H%2BXq/Xm%2B3eC7sOwJjq0U4CLORxxCB47gTOeDAPOVhfskyr/KaLBpNqmDgj%2Bg7AaOsHTpB5gBMus65tBhFgcRCHeg86ZpgQDJpIwrCvAAYvR9yMamzGscwbBHAaIn3MQwAKFiR5HoSBDED8XimtJUAMByDDfBEyy1g8AG7lkh4nkcwCMPE8SAUeVEIBAHFZgEpFUNxvFOU5YkSbZpEiQarQSdyqlrlsaSDkhi6pvpx6DruuZHO%2BqEhSm/CbhAYWGXg97RUqRypeCNIQQhBpnlU8SXqaxA3gIkmoZl1jWNywUMbxu4ALRRbS9JMvQrK1rF9WMZc5bhOyE6YNQpBHN5CieV1ypOZFlgtXSjLMp1H5pvO3YCjxqaNs2NIfmtPYAPRHf8R0HUcAACBBpK0rBHBxJ1HRdV03RqrmSQ8p1PddpUauRCEPWd53fbd6yKh9j1Ay9RxUADF3Axq42w%2Bd23EAw6GYJh2HgvxbFCVxcWMTjgmvJ5b2OTN5X4SOoGTrRc5mGRuWIckOaWURTNSTJBByQpSlHipakaZgWn/Lpe4GYOxkRLK8nIEe%2BAKNdBCjtZtmg%2BTzma1rW6fOWUIwsQ4L3ck2C2Q5BPa5baZve5DOeeNvmoP5mCBby02hQe4VHCwkUs0cjVPrQtCjSw4RHsUaUxe78UkEcSWeylkcZVlZzQwaYUQFNVVzZYtULj16Z4eLXuLR1aVS6ZssWdTFtW5buEcybtd11r1AmOaFh4O3OajeNk0rbxeBUHHpcssi3vIHn0fOaHDDh5kaVd91m2az7aWjxV0%2BrV%2BBf7Rtaag2lMMd7P8%2BSea3bL2mKMtqfxR7TvKr3F44QEJ6hmmhMXAQEXL%2BCO/g4mBTwJn/N%2BZhDKNiTlfVMCU46gIAZlKBFhEHZXzJVLuOcLDAILtfTAQU5p%2BwUM6NoVAAiSAgHgAAVEAgeKY94r1bHgnajYH7rWfq/BBn8CBmB/pTI48DwGAOwQwgREC8FIPrIKSB1U/ZMGobQxhTZUaKNYT2LwWRqQ2SiqIggEBKHLAgDoqaPZ2Sci3PUaEuitEM0bMItMfUQD6yIIbM4xtTZUEknyLhXBRpcITChNCQpf4cMEUcIKfs3xRwJkXZKg4I5RQrjLcyCslYq0bKNDxvjjGdhzBwVYtBODml4H4bgvBUCcDcNVSwW51ibFeLsHgpACCaDyasCcIBzQaH0JwSQxSWmkHKRwXgCgQBdOaRwLQqw4CwCQGgTCdB4jkEoHMtICyEjACkF6GgCp4gjIgDEfpMRwitAZJwRpRzmDEAZAAeRiNoSxZzeBzLYIIa56lTkTN4FgGIXhgBuDELQEZpTSBYAJEYcQnyQV4FlA4PAUIgVaGCKoaEiltiNNfvUfp3wYilSuR4LA/SeYsEeaQKExAYiZEwDmDGhhgAaVAJ81YVADASQAGp4EwAAd2uQJEl/BBAiDEOwKQMhBCKBUOoSFugfEGCMCgKpNhsUjMgKsVAgVbycEatchQZSyXySwMqzOPRLE5BcAwdwnhOh6FCOEYYhUEg%2BOKNkAQ0w/COsyM6hgCxzx6DsCagQ/QpiWoKD4v1sKA2TEGLaxYoxQ2Rtdb6yN3r7USFWAoWpWxU3dI4EU0gJTEWDKOKoRIARGpkKMsgXcUgDRmDjrgQgscGnLF4OMyZbSOldIKRwXpeb%2BmDOGaMppLStLZrMH0yF/ah2MtWGSrIzhJBAA
///
/// @tparam config
/// @tparam F
/// @tparam Args
/// @param f
/// @param args
/// @return
template<const BenchmarkConfig &config=benchmarkConfig,
         typename F,
         typename ...Args>
__attribute__((noinline))
static size_t genereric_bench(F &&f,
                       Args &&...args) noexcept {
    size_t c = 0;
    for (size_t i = 0; i < config.number_iterations; i++) {
        c -= cpucycles();
        std::invoke(f, args...);
        c += cpucycles();
    }

    return c;
}

///
/// @tparam F
/// @tparam Args
/// @tparam config
/// @param out
/// @param f
/// @param args
/// @return
template<typename F,
         typename ...Args,
         const BenchmarkConfig &config=benchmarkConfig>
__attribute__((noinline))
static size_t genereric_dispatch(F &out,
                                 std::vector<F> &f,
                                 Args &&...args) noexcept {
    size_t mc = -1ull, min_pos = 0;
    for (size_t i = 0; i < f.size(); i++) {
        const size_t cycles = genereric_bench
                                <config>
                                (f[i], args...);
        if (cycles < mc) {
            min_pos = i;
            mc = cycles;
        }
    }

    out = f[min_pos];
    return min_pos;
}


#endif
