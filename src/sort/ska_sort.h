//          Copyright Malte Skarupke 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <tuple>
#include <utility>
#include <functional>

#include "helper.h"

namespace detail
{
template<typename count_type, typename It, typename OutIt, typename ExtractKey>
void counting_sort_impl(It begin,
		It end,
		OutIt out_begin, 
		ExtractKey && extract_key) noexcept {
	ZoneScoped;
    count_type counts[256] = {};
    for (It it = begin; it != end; ++it) {
        ++counts[extract_key(*it)];
    }

    count_type total = 0;
    for (count_type & count : counts) {
        count_type old_count = count;
        count = total;
        total += old_count;
    }

    for (; begin != end; ++begin) {
        std::uint8_t key = extract_key(*begin);
        out_begin[counts[key]++] = std::move(*begin);
    }
}

template<typename It, typename OutIt, typename ExtractKey>
void counting_sort_impl(It begin,
						It end,
						OutIt out_begin, 
						ExtractKey && extract_key) noexcept {
	ZoneScoped;
    counting_sort_impl<std::uint64_t>(begin, end, out_begin, extract_key);
}

[[nodiscard]] inline bool to_unsigned_or_bool(bool b) noexcept {
	ZoneScoped;
    return b;
}

[[nodiscard]] inline unsigned char to_unsigned_or_bool(unsigned char c) noexcept {
    return c;
}

[[nodiscard]] inline unsigned char to_unsigned_or_bool(signed char c) {
    return static_cast<unsigned char>(c) + 128;
}

[[nodiscard]] inline unsigned char to_unsigned_or_bool(char c) noexcept {
    return static_cast<unsigned char>(c);
}

[[nodiscard]] inline std::uint16_t to_unsigned_or_bool(char16_t c) noexcept {
    return static_cast<std::uint16_t>(c);
}

[[nodiscard]] inline std::uint32_t to_unsigned_or_bool(char32_t c) noexcept {
    return static_cast<std::uint32_t>(c);
}

[[nodiscard]] inline std::uint32_t to_unsigned_or_bool(wchar_t c) noexcept {
    return static_cast<std::uint32_t>(c);
}

[[nodiscard]] inline unsigned short to_unsigned_or_bool(short i) noexcept {
    return static_cast<unsigned short>(i) + 
		   static_cast<unsigned short>(1 << (sizeof(short) * 8 - 1));
}

[[nodiscard]] inline unsigned short to_unsigned_or_bool(unsigned short i) noexcept {
    return i;
}

[[nodiscard]] inline unsigned int to_unsigned_or_bool(int i) noexcept {
    return static_cast<unsigned int>(i) +
		   static_cast<unsigned int>(1 << (sizeof(int) * 8 - 1));
}

[[nodiscard]] inline unsigned int to_unsigned_or_bool(unsigned int i) noexcept  {
    return i;
}

[[nodiscard]] inline unsigned long to_unsigned_or_bool(long l) noexcept {
    return static_cast<unsigned long>(l) +
		   static_cast<unsigned long>(1l << (sizeof(long) * 8 - 1));
}

[[nodiscard]] inline unsigned long to_unsigned_or_bool(unsigned long l) noexcept {
    return l;
}

[[nodiscard]] inline unsigned long long to_unsigned_or_bool(long long l) noexcept {
    return static_cast<unsigned long long>(l) + static_cast<unsigned long long>(1ll << (sizeof(long long) * 8 - 1));
}

[[nodiscard]] inline unsigned long long to_unsigned_or_bool(unsigned long long l) noexcept {
    return l;
}

[[nodiscard]] inline std::uint32_t to_unsigned_or_bool(float f) noexcept {
	ZoneScoped;
    union
    {
        float f;
        std::uint32_t u;
    } as_union = { f };
    std::uint32_t sign_bit = -std::int32_t(as_union.u >> 31);
    return as_union.u ^ (sign_bit | 0x80000000);
}

[[nodiscard]] inline std::uint64_t to_unsigned_or_bool(double f) noexcept {
	ZoneScoped;
    union
    {
        double d;
        std::uint64_t u;
    } as_union = { f };
    std::uint64_t sign_bit = -std::int64_t(as_union.u >> 63);
    return as_union.u ^ (sign_bit | 0x8000000000000000);
}

template<typename T>
[[nodiscard]] inline size_t to_unsigned_or_bool(T * ptr) noexcept {
    return reinterpret_cast<size_t>(ptr);
}

template<size_t>
struct SizedRadixSorter;

template<>
struct SizedRadixSorter<1>
{
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]]  static bool sort(It begin,
							   It end,
							   OutIt buffer_begin, 
							   ExtractKey && extract_key) {
        counting_sort_impl(begin, end, buffer_begin, [&](auto && o) {
            return to_unsigned_or_bool(extract_key(o));
        });

        return true;
    }

    static constexpr size_t pass_count = 2;
};
template<>
struct SizedRadixSorter<2>
{
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end, 
							   OutIt buffer_begin, 
							   ExtractKey && extract_key) noexcept {
        std::ptrdiff_t num_elements = end - begin;
        if (num_elements <= (1ll << 32u)) {
            return sort_inline<uint32_t>(begin, end, buffer_begin, buffer_begin + num_elements, extract_key);
		} else {
            return sort_inline<uint64_t>(begin, end, buffer_begin, buffer_begin + num_elements, extract_key);
		}
    }

    template<typename count_type, typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort_inline(It begin,
									  It end,
									  OutIt out_begin, 
									  OutIt out_end, 
									  ExtractKey && extract_key) noexcept {
		ZoneScoped;
        count_type counts0[256] = {};
        count_type counts1[256] = {};

        for (It it = begin; it != end; ++it) {
            uint16_t key = to_unsigned_or_bool(extract_key(*it));
            ++counts0[key & 0xff];
            ++counts1[(key >> 8) & 0xff];
        }

		// TODO not optimized:
		//https://godbo.lt/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:28,endLineNumber:16,positionColumn:28,positionLineNumber:16,selectionStartColumn:28,selectionStartLineNumber:16,startColumn:28,startLineNumber:16),source:'//+Type+your+code+here,+or+load+an+example.%0A%23include+%3Cstdint.h%3E%0A%0Aint+square(uint32_t+*counts0,+uint32_t+*counts1)+%7B%0A++++uint32_t+total0+%3D+0%3B%0A++++uint32_t+total1+%3D+0%3B%0A++++for+(uint32_t+i+%3D+0%3B+i+%3C+256%3B+%2B%2Bi)+%7B%0A++++++++uint32_t+old_count0+%3D+counts0%5Bi%5D%3B%0A++++++++uint32_t+old_count1+%3D+counts1%5Bi%5D%3B%0A++++++++counts0%5Bi%5D+%3D+total0%3B%0A++++++++counts1%5Bi%5D+%3D+total1%3B%0A++++++++total0+%2B%3D+old_count0%3B%0A++++++++total1+%2B%3D+old_count1%3B%0A++++%7D%0A%0A++++return+total0+%2B+total1%3B%0A%7D'),l:'5',n:'1',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:g141,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-mavx+-mavx2+-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+gcc+14.1+(Editor+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
        count_type total0 = 0;
        count_type total1 = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            count_type old_count0 = counts0[i];
            count_type old_count1 = counts1[i];
            counts0[i] = total0;
            counts1[i] = total1;
            total0 += old_count0;
            total1 += old_count1;
        }

        for (It it = begin; it != end; ++it) {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it));
            out_begin[counts0[key]++] = std::move(*it);
        }

        for (OutIt it = out_begin; it != out_end; ++it) {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 8;
            begin[counts1[key]++] = std::move(*it);
        }

        return false;
    }

    static constexpr size_t pass_count = 3;
};

template<>
struct SizedRadixSorter<4>
{

    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]]  static bool sort(It begin,
					It end,
					OutIt buffer_begin, 
					ExtractKey && extract_key) noexcept {
        std::ptrdiff_t num_elements = end - begin;
        if (num_elements <= (1ll << 32u)) {
            return sort_inline<uint32_t>(begin, end, buffer_begin, buffer_begin + num_elements, extract_key);
		} else {
            return sort_inline<uint64_t>(begin, end, buffer_begin, buffer_begin + num_elements, extract_key);
		}
    }

    template<typename count_type, typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort_inline(It begin,
			It end,
			OutIt out_begin,
			OutIt out_end,
			ExtractKey && extract_key) noexcept {
		ZoneScoped;
        count_type counts0[256] = {};
        count_type counts1[256] = {};
        count_type counts2[256] = {};
        count_type counts3[256] = {};

        for (It it = begin; it != end; ++it)
        {
            uint32_t key = to_unsigned_or_bool(extract_key(*it));
            ++counts0[key & 0xff];
            ++counts1[(key >> 8) & 0xff];
            ++counts2[(key >> 16) & 0xff];
            ++counts3[(key >> 24) & 0xff];
        }
        count_type total0 = 0;
        count_type total1 = 0;
        count_type total2 = 0;
        count_type total3 = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            count_type old_count0 = counts0[i];
            count_type old_count1 = counts1[i];
            count_type old_count2 = counts2[i];
            count_type old_count3 = counts3[i];
            counts0[i] = total0;
            counts1[i] = total1;
            counts2[i] = total2;
            counts3[i] = total3;
            total0 += old_count0;
            total1 += old_count1;
            total2 += old_count2;
            total3 += old_count3;
        }

        for (It it = begin; it != end; ++it){
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it));
            out_begin[counts0[key]++] = std::move(*it);
        }
        for (OutIt it = out_begin; it != out_end; ++it){
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 8;
            begin[counts1[key]++] = std::move(*it);
        }
        for (It it = begin; it != end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 16;
            out_begin[counts2[key]++] = std::move(*it);
        }
        for (OutIt it = out_begin; it != out_end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 24;
            begin[counts3[key]++] = std::move(*it);
        }
        return false;
    }

    static constexpr size_t pass_count = 5;
};
template<>
struct SizedRadixSorter<8>
{
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
			It end, 
			OutIt buffer_begin, 
			ExtractKey && extract_key) noexcept {
        std::ptrdiff_t num_elements = end - begin;
        if (num_elements <= (1ll << 32))
            return sort_inline<uint32_t>(begin, end, buffer_begin, buffer_begin + num_elements, extract_key);
        else
            return sort_inline<uint64_t>(begin, end, buffer_begin, buffer_begin + num_elements, extract_key);
    }

    template<typename count_type, typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort_inline(It begin,
									  It end,
									  OutIt out_begin, 
									  OutIt out_end,
									  ExtractKey && extract_key) noexcept {
		ZoneScoped;
		count_type counts0[256] = {};
        count_type counts1[256] = {};
        count_type counts2[256] = {};
        count_type counts3[256] = {};
        count_type counts4[256] = {};
        count_type counts5[256] = {};
        count_type counts6[256] = {};
        count_type counts7[256] = {};

        for (It it = begin; it != end; ++it) {
            uint64_t key = to_unsigned_or_bool(extract_key(*it));
            ++counts0[key & 0xff];
            ++counts1[(key >> 8) & 0xff];
            ++counts2[(key >> 16) & 0xff];
            ++counts3[(key >> 24) & 0xff];
            ++counts4[(key >> 32) & 0xff];
            ++counts5[(key >> 40) & 0xff];
            ++counts6[(key >> 48) & 0xff];
            ++counts7[(key >> 56) & 0xff];
        }
        count_type total0 = 0;
        count_type total1 = 0;
        count_type total2 = 0;
        count_type total3 = 0;
        count_type total4 = 0;
        count_type total5 = 0;
        count_type total6 = 0;
        count_type total7 = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            const count_type old_count0 = counts0[i];
            const count_type old_count1 = counts1[i];
            const count_type old_count2 = counts2[i];
            const count_type old_count3 = counts3[i];
            const count_type old_count4 = counts4[i];
            const count_type old_count5 = counts5[i];
            const count_type old_count6 = counts6[i];
            const count_type old_count7 = counts7[i];
            counts0[i] = total0;
            counts1[i] = total1;
            counts2[i] = total2;
            counts3[i] = total3;
            counts4[i] = total4;
            counts5[i] = total5;
            counts6[i] = total6;
            counts7[i] = total7;
            total0 += old_count0;
            total1 += old_count1;
            total2 += old_count2;
            total3 += old_count3;
            total4 += old_count4;
            total5 += old_count5;
            total6 += old_count6;
            total7 += old_count7;
        }
        for (It it = begin; it != end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it));
            out_begin[counts0[key]++] = std::move(*it);
        }
        for (OutIt it = out_begin; it != out_end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 8;
            begin[counts1[key]++] = std::move(*it);
        }
        for (It it = begin; it != end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 16;
            out_begin[counts2[key]++] = std::move(*it);
        }
        for (OutIt it = out_begin; it != out_end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 24;
            begin[counts3[key]++] = std::move(*it);
        }
        for (It it = begin; it != end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 32;
            out_begin[counts4[key]++] = std::move(*it);
        }
        for (OutIt it = out_begin; it != out_end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 40;
            begin[counts5[key]++] = std::move(*it);
        }
        for (It it = begin; it != end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 48;
            out_begin[counts6[key]++] = std::move(*it);
        }
        for (OutIt it = out_begin; it != out_end; ++it)
        {
            std::uint8_t key = to_unsigned_or_bool(extract_key(*it)) >> 56;
            begin[counts7[key]++] = std::move(*it);
        }
        return false;
    }

    static constexpr size_t pass_count = 9;
};

template<typename>
struct RadixSorter;
template<>
struct RadixSorter<bool> {
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end,
							   OutIt buffer_begin, 
							   ExtractKey && extract_key) noexcept {
		ZoneScoped;
        size_t false_count = 0;
        for (It it = begin; it != end; ++it) {
            if (!extract_key(*it))
                ++false_count;
        }

        size_t true_position = false_count;
        false_count = 0;
        for (; begin != end; ++begin) {
            if (extract_key(*begin)) {
                buffer_begin[true_position++] = std::move(*begin);
			} else {
                buffer_begin[false_count++] = std::move(*begin);
			}
        }
        return true;
    }

    static constexpr size_t pass_count = 2;
};

template<>
struct RadixSorter<signed char> : SizedRadixSorter<sizeof(signed char)>
{};

template<>
struct RadixSorter<unsigned char> : SizedRadixSorter<sizeof(unsigned char)>
{};

template<>
struct RadixSorter<signed short> : SizedRadixSorter<sizeof(signed short)>
{};

template<>
struct RadixSorter<unsigned short> : SizedRadixSorter<sizeof(unsigned short)>
{};

template<>
struct RadixSorter<signed int> : SizedRadixSorter<sizeof(signed int)>
{};

template<>
struct RadixSorter<unsigned int> : SizedRadixSorter<sizeof(unsigned int)>
{};

template<>
struct RadixSorter<signed long> : SizedRadixSorter<sizeof(signed long)>
{};

template<>
struct RadixSorter<unsigned long> : SizedRadixSorter<sizeof(unsigned long)>
{};

template<>
struct RadixSorter<signed long long> : SizedRadixSorter<sizeof(signed long long)>
{};

template<>
struct RadixSorter<unsigned long long> : SizedRadixSorter<sizeof(unsigned long long)>
{};

template<>
struct RadixSorter<float> : SizedRadixSorter<sizeof(float)>
{};

template<>
struct RadixSorter<double> : SizedRadixSorter<sizeof(double)>
{};

template<>
struct RadixSorter<char> : SizedRadixSorter<sizeof(char)>
{};

template<>
struct RadixSorter<wchar_t> : SizedRadixSorter<sizeof(wchar_t)>
{};

template<>
struct RadixSorter<char16_t> : SizedRadixSorter<sizeof(char16_t)>
{};

template<>
struct RadixSorter<char32_t> : SizedRadixSorter<sizeof(char32_t)>
{};

template<typename K, typename V>
struct RadixSorter<std::pair<K, V>> {
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end,
							   OutIt buffer_begin, 
							   ExtractKey && extract_key) noexcept {
		ZoneScoped;
        bool first_result = RadixSorter<V>::sort(begin, end, buffer_begin, [&](auto && o) {
            return extract_key(o).second;
        });

        auto extract_first = [&](auto && o) {
            return extract_key(o).first;
        };

        if (first_result) {
            return !RadixSorter<K>::sort(buffer_begin, buffer_begin + (end - begin), begin, extract_first);
        } else {
            return RadixSorter<K>::sort(begin, end, buffer_begin, extract_first);
        }
    }

    static constexpr size_t pass_count = RadixSorter<K>::pass_count + RadixSorter<V>::pass_count;
};

template<typename K, typename V>
struct RadixSorter<const std::pair<K, V> &> {
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end,
							   OutIt buffer_begin, 
							   ExtractKey && extract_key) noexcept {
        bool first_result = RadixSorter<V>::sort(begin, end, buffer_begin, [&](auto && o) -> const V& {
            return extract_key(o).second;
        });
        auto extract_first = [&](auto && o) -> const K& {
            return extract_key(o).first;
        };

        if (first_result){
            return !RadixSorter<K>::sort(buffer_begin, buffer_begin + (end - begin), begin, extract_first);
        } else {
            return RadixSorter<K>::sort(begin, end, buffer_begin, extract_first);
        }
    }

    static constexpr size_t pass_count = RadixSorter<K>::pass_count + RadixSorter<V>::pass_count;
};

template<size_t I, size_t S, typename Tuple>
struct TupleRadixSorter {
    using NextSorter = TupleRadixSorter<I + 1, S, Tuple>;
    using ThisSorter = RadixSorter<typename std::tuple_element<I, Tuple>::type>;

    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end,
							   OutIt out_begin, 
							   OutIt out_end, 
							   ExtractKey && extract_key) noexcept {
        bool which = NextSorter::sort(begin, end, out_begin, out_end, extract_key);
        auto extract_i = [&](auto && o) {
            return std::get<I>(extract_key(o));
        };

        if (which) {
            return !ThisSorter::sort(out_begin, out_end, begin, extract_i);
		} else {
            return ThisSorter::sort(begin, end, out_begin, extract_i);
		}
    }

    static constexpr size_t pass_count = ThisSorter::pass_count + NextSorter::pass_count;
};

template<size_t I, size_t S, typename Tuple>
struct TupleRadixSorter<I, S, const Tuple &> {
    using NextSorter = TupleRadixSorter<I + 1, S, const Tuple &>;
    using ThisSorter = RadixSorter<typename std::tuple_element<I, Tuple>::type>;

    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end,
							   OutIt out_begin, 
							   OutIt out_end,
							   ExtractKey && extract_key) noexcept {
        bool which = NextSorter::sort(begin, end, out_begin, out_end, extract_key);
        auto extract_i = [&](auto && o) -> decltype(auto) {
            return std::get<I>(extract_key(o));
        };

        if (which) {
            return !ThisSorter::sort(out_begin, out_end, begin, extract_i);
		} else {
            return ThisSorter::sort(begin, end, out_begin, extract_i);
		}
    }

    static constexpr size_t pass_count = ThisSorter::pass_count + NextSorter::pass_count;
};

template<size_t I, typename Tuple>
struct TupleRadixSorter<I, I, Tuple> {
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] constexpr static bool sort(It, It, OutIt, OutIt, ExtractKey &&) noexcept {
        return false;
    }

    static constexpr size_t pass_count = 0;
};

template<size_t I, typename Tuple>
struct TupleRadixSorter<I, I, const Tuple &> {
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] constexpr static bool sort(It, It, OutIt, OutIt, ExtractKey &&) noexcept {
        return false;
    }

    static constexpr size_t pass_count = 0;
};

template<typename... Args>
struct RadixSorter<std::tuple<Args...>> {
    using SorterImpl = TupleRadixSorter<0, sizeof...(Args), std::tuple<Args...>>;

    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end, 
							   OutIt buffer_begin,
							   ExtractKey && extract_key) noexcept {
        return SorterImpl::sort(begin, end, buffer_begin, buffer_begin + (end - begin), extract_key);
    }

    static constexpr size_t pass_count = SorterImpl::pass_count;
};

template<typename... Args>
struct RadixSorter<const std::tuple<Args...> &> {
    using SorterImpl = TupleRadixSorter<0, sizeof...(Args), const std::tuple<Args...> &>;

    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end,
							   OutIt buffer_begin, 
							   ExtractKey && extract_key) noexcept {
        return SorterImpl::sort(begin, end, buffer_begin, buffer_begin + (end - begin), extract_key);
    }

    static constexpr size_t pass_count = SorterImpl::pass_count;
};

template<typename T, size_t S>
struct RadixSorter<std::array<T, S>> {
    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							  It end,
							  OutIt buffer_begin, 
							  ExtractKey && extract_key) noexcept {
        auto buffer_end = buffer_begin + (end - begin);
        bool which = false;
        for (size_t i = S; i > 0; --i) {
            auto extract_i = [&, i = i - 1](auto && o) {
                return extract_key(o)[i];
            };

            if (which) {
                which = !RadixSorter<T>::sort(buffer_begin, buffer_end, begin, extract_i);
			} else {
                which = RadixSorter<T>::sort(begin, end, buffer_begin, extract_i);
			}
        }
        return which;
    }

    static constexpr size_t pass_count = RadixSorter<T>::pass_count * S;
};

template<typename T>
struct RadixSorter<const T> : RadixSorter<T>
{};

template<typename T>
struct RadixSorter<T &> : RadixSorter<const T &>
{};

template<typename T>
struct RadixSorter<T &&> : RadixSorter<T>
{};

template<typename T>
struct RadixSorter<const T &> : RadixSorter<T>
{};

template<typename T>
struct RadixSorter<const T &&> : RadixSorter<T>
{};

// these structs serve two purposes
// 1. they serve as illustration for how to implement the to_radix_sort_key function
// 2. they help produce better error messages. with these overloads you get the
//    error message "no matching function for call to to_radix_sort(your_type)"
//    without these examples, you'd get the error message "to_radix_sort_key was
//    not declared in this scope" which is a much less useful error message
struct ExampleStructA { int i; };
struct ExampleStructB { float f; };
inline int to_radix_sort_key(ExampleStructA a) { return a.i; }
inline float to_radix_sort_key(ExampleStructB b) { return b.f; }
template<typename T, typename Enable = void>
struct FallbackRadixSorter : RadixSorter<decltype(to_radix_sort_key(std::declval<T>()))> {
    using base = RadixSorter<decltype(to_radix_sort_key(std::declval<T>()))>;

    template<typename It, typename OutIt, typename ExtractKey>
	[[nodiscard]] static bool sort(It begin,
							   It end,
							   OutIt buffer_begin, 
							   ExtractKey && extract_key) noexcept {
        return base::sort(begin, end, buffer_begin, [&](auto && a) -> decltype(auto) {
            return to_radix_sort_key(extract_key(a));
        });
    }
};

template<typename...>
struct nested_void {
	using type = void;
};

template<typename... Args>
using void_t = typename nested_void<Args...>::type;

template<typename T>
struct has_subscript_operator_impl {
	template<typename U, typename = decltype(std::declval<U>()[0])>
	static std::true_type test(int);
	template<typename>
	static std::false_type test(...);

	using type = decltype(test<T>(0));
};

template<typename T>
using has_subscript_operator = typename has_subscript_operator_impl<T>::type;


template<typename T>
struct FallbackRadixSorter<T, void_t<decltype(to_unsigned_or_bool(std::declval<T>()))>>
    : RadixSorter<decltype(to_unsigned_or_bool(std::declval<T>()))>
{};

template<typename T>
struct RadixSorter : FallbackRadixSorter<T>
{};

template<typename T>
size_t radix_sort_pass_count = RadixSorter<T>::pass_count;

template<typename It, typename Func>
inline void unroll_loop_four_times(It begin,
		size_t iteration_count, 
		Func && to_call) noexcept {
    size_t loop_count = iteration_count / 4;
    const size_t remainder_count = iteration_count - loop_count * 4;
    for (; loop_count > 0; --loop_count) {
		ZoneScoped;
        to_call(begin);
        ++begin;
        to_call(begin);
        ++begin;
        to_call(begin);
        ++begin;
        to_call(begin);
        ++begin;
    }

    switch(remainder_count) {
    case 3:
        to_call(begin);
        ++begin;
		__attribute__ ((fallthrough));
    case 2:
        to_call(begin);
        ++begin;
		__attribute__ ((fallthrough));
    case 1:
        to_call(begin);
    }
}

template<typename It, typename F>
[[nodiscard]] inline It custom_std_partition(It begin, It end, F && func) noexcept {
    for (;; ++begin) {
        if (begin == end)
            return end;
        if (!func(*begin))
            break;
    }

    It it = begin;
    for(++it; it != end; ++it) {
        if (!func(*it))
            continue;

        std::iter_swap(begin, it);
        ++begin;
    }
    return begin;
}

struct PartitionInfo {
    PartitionInfo()
        : count(0)
    {}

    union {
        size_t count;
        size_t offset;
    };
    size_t next_offset;
};

template<size_t>
struct UnsignedForSize;
template<>
struct UnsignedForSize<1>
{
    typedef uint8_t type;
};

template<>
struct UnsignedForSize<2>
{
    typedef uint16_t type;
};

template<>
struct UnsignedForSize<4>
{
    typedef uint32_t type;
};

template<>
struct UnsignedForSize<8>
{
    typedef uint64_t type;
};

template<typename T>
struct SubKey;

template<size_t Size>
struct SizedSubKey {
    template<typename T>
    static auto sub_key(T && value, void *) {
        return to_unsigned_or_bool(value);
    }

    typedef SubKey<void> next;

    using sub_key_type = typename UnsignedForSize<Size>::type;
};

template<typename T>
struct SubKey<const T> : SubKey<T>
{};

template<typename T>
struct SubKey<T &> : SubKey<T>
{};

template<typename T>
struct SubKey<T &&> : SubKey<T>
{};

template<typename T>
struct SubKey<const T &> : SubKey<T>
{};

template<typename T>
struct SubKey<const T &&> : SubKey<T>
{};

template<typename T, typename Enable = void>
struct FallbackSubKey
    : SubKey<decltype(to_radix_sort_key(std::declval<T>()))> {
    using base = SubKey<decltype(to_radix_sort_key(std::declval<T>()))>;

    template<typename U>
    static decltype(auto) sub_key(U && value, void * data) noexcept {
        return base::sub_key(to_radix_sort_key(value), data);
    }
};

template<typename T>
struct FallbackSubKey<T, void_t<decltype(to_unsigned_or_bool(std::declval<T>()))>>
    : SubKey<decltype(to_unsigned_or_bool(std::declval<T>()))>
{};

template<typename T>
struct SubKey : FallbackSubKey<T>
{};

template<>
struct SubKey<bool> {
    template<typename T>
	[[nodiscard]] static bool sub_key(T && value, void *) noexcept {
        return value;
    }

    typedef SubKey<void> next;

    using sub_key_type = bool;
};

template<>
struct SubKey<void>;

template<>
struct SubKey<unsigned char> : SizedSubKey<sizeof(unsigned char)>
{};

template<>
struct SubKey<unsigned short> : SizedSubKey<sizeof(unsigned short)>
{};

template<>
struct SubKey<unsigned int> : SizedSubKey<sizeof(unsigned int)>
{};

template<>
struct SubKey<unsigned long> : SizedSubKey<sizeof(unsigned long)>
{};

template<>
struct SubKey<unsigned long long> : SizedSubKey<sizeof(unsigned long long)>
{};

template<typename T>
struct SubKey<T *> : SizedSubKey<sizeof(T *)>
{};

template<typename F, typename S, typename Current>
struct PairSecondSubKey : Current {
	[[nodiscard]] static decltype(auto) sub_key(const std::pair<F, S> & value, void * sort_data) noexcept {
        return Current::sub_key(value.second, sort_data);
    }

    using next = typename std::conditional<std::is_same<SubKey<void>, typename Current::next>::value, SubKey<void>, PairSecondSubKey<F, S, typename Current::next>>::type;
};

template<typename F, typename S, typename Current>
struct PairFirstSubKey : Current {
	[[nodiscard]] static decltype(auto) sub_key(const std::pair<F, S> & value, void * sort_data) noexcept {
        return Current::sub_key(value.first, sort_data);
    }

    using next = typename std::conditional<std::is_same<SubKey<void>, typename Current::next>::value, PairSecondSubKey<F, S, SubKey<S>>, PairFirstSubKey<F, S, typename Current::next>>::type;
};

template<typename F, typename S>
struct SubKey<std::pair<F, S>> : PairFirstSubKey<F, S, SubKey<F>>
{};

template<size_t Index, typename First, typename... More>
struct TypeAt : TypeAt<Index - 1, More..., void>
{};

template<typename First, typename... More>
struct TypeAt<0, First, More...> {
    typedef First type;
};

template<size_t Index, typename Current, typename First, typename... More>
struct TupleSubKey;

template<size_t Index, typename Next, typename First, typename... More>
struct NextTupleSubKey {
    using type = TupleSubKey<Index, Next, First, More...>;
};

template<size_t Index, typename First, typename Second, typename... More>
struct NextTupleSubKey<Index, SubKey<void>, First, Second, More...> {
    using type = TupleSubKey<Index + 1, SubKey<Second>, Second, More...>;
};

template<size_t Index, typename First>
struct NextTupleSubKey<Index, SubKey<void>, First> {
    using type = SubKey<void>;
};

template<size_t Index, typename Current, typename First, typename... More>
struct TupleSubKey : Current {
    template<typename Tuple>
    static decltype(auto) sub_key(const Tuple & value, void * sort_data) noexcept {
        return Current::sub_key(std::get<Index>(value), sort_data);
    }

    using next = typename NextTupleSubKey<Index, typename Current::next, First, More...>::type;
};

template<size_t Index, typename Current, typename First>
struct TupleSubKey<Index, Current, First> : Current {
    template<typename Tuple>
	
    static decltype(auto) sub_key(const Tuple & value, void * sort_data) noexcept {
        return Current::sub_key(std::get<Index>(value), sort_data);
    }

    using next = typename NextTupleSubKey<Index, typename Current::next, First>::type;
};

template<typename First, typename... More>
struct SubKey<std::tuple<First, More...>> : TupleSubKey<0, SubKey<First>, First, More...>
{};

struct BaseListSortData
{
    size_t current_index;
    size_t recursion_limit;
    void * next_sort_data;
};
template<typename It, typename ExtractKey>
struct ListSortData : BaseListSortData
{
    void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *);
};

template<typename CurrentSubKey, typename T>
struct ListElementSubKey : SubKey<typename std::decay<decltype(std::declval<T>()[0])>::type>
{
    using base = SubKey<typename std::decay<decltype(std::declval<T>()[0])>::type>;

    using next = ListElementSubKey;

    template<typename U>
    static decltype(auto) sub_key(U && value, void * sort_data)
    {
        BaseListSortData * list_sort_data = static_cast<BaseListSortData *>(sort_data);
        const T & list = CurrentSubKey::sub_key(value, list_sort_data->next_sort_data);
        return base::sub_key(list[list_sort_data->current_index], list_sort_data->next_sort_data);
    }
};

template<typename T>
struct ListSubKey
{
    using next = SubKey<void>;

    using sub_key_type = T;

    static const T & sub_key(const T & value, void *)
    {
        return value;
    }
};

template<typename T>
struct FallbackSubKey<T, typename std::enable_if<has_subscript_operator<T>::value>::type> : ListSubKey<T>
{
};

template<typename It, typename ExtractKey>
inline void StdSortFallback(It begin, It end, ExtractKey & extract_key)
{
    std::sort(begin, end, [&](auto && l, auto && r){ return extract_key(l) < extract_key(r); });
}

template<std::ptrdiff_t StdSortThreshold, typename It, typename ExtractKey>
inline bool StdSortIfLessThanThreshold(It begin, It end, std::ptrdiff_t num_elements, ExtractKey & extract_key)
{
    if (num_elements <= 1)
        return true;
    if (num_elements >= StdSortThreshold)
        return false;
    StdSortFallback(begin, end, extract_key);
    return true;
}

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType = typename CurrentSubKey::sub_key_type>
struct InplaceSorter;

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, size_t NumBytes, size_t Offset = 0>
struct UnsignedInplaceSorter
{
    static constexpr size_t ShiftAmount = (((NumBytes - 1) - Offset) * 8);
    template<typename T>
    inline static uint8_t current_byte(T && elem, void * sort_data)
    {
        return CurrentSubKey::sub_key(elem, sort_data) >> ShiftAmount;
    }
    template<typename It, typename ExtractKey>
    static void sort(It begin, It end, std::ptrdiff_t num_elements, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * sort_data)
    {
        if (num_elements < AmericanFlagSortThreshold)
            american_flag_sort(begin, end, extract_key, next_sort, sort_data);
        else
            ska_byte_sort(begin, end, extract_key, next_sort, sort_data);
    }

    template<typename It, typename ExtractKey>
    static void american_flag_sort(It begin,
			It end,
			ExtractKey & extract_key, 
			void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), 
			void * sort_data) noexcept {
		ZoneScoped;
        PartitionInfo partitions[256];
        for (It it = begin; it != end; ++it) {
            ++partitions[current_byte(extract_key(*it), sort_data)].count;
        }

        size_t total = 0;
        uint8_t remaining_partitions[256];
        int num_partitions = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            size_t count = partitions[i].count;
            if (!count) {
                continue;
			}
            partitions[i].offset = total;
            total += count;
            partitions[i].next_offset = total;
            remaining_partitions[num_partitions] = i;
            ++num_partitions;
        }
        
		if (num_partitions > 1) {
            uint8_t * current_block_ptr = remaining_partitions;
            PartitionInfo * current_block = partitions + *current_block_ptr;
            uint8_t * last_block = remaining_partitions + num_partitions - 1;
            It it = begin;
            It block_end = begin + current_block->next_offset;
            It last_element = end - 1;
            for (;;) {
                PartitionInfo * block = partitions + current_byte(extract_key(*it), sort_data);
                if (block == current_block) {
                    ++it;
                    if (it == last_element) {
                        break;
					} else if (it == block_end) {
                        for (;;) {
                            ++current_block_ptr;
                            if (current_block_ptr == last_block) {
                                goto recurse;
							}

                            current_block = partitions + *current_block_ptr;
                            if (current_block->offset != current_block->next_offset) {
                                break;
							}
                        }

                        it = begin + current_block->offset;
                        block_end = begin + current_block->next_offset;
                    }
                } else {
                    size_t offset = block->offset++;
                    std::iter_swap(it, begin + offset);
                }
            }
        }

        recurse:
        if (Offset + 1 != NumBytes || next_sort){
            size_t start_offset = 0;
            It partition_begin = begin;
            for (uint8_t * it = remaining_partitions, * end = remaining_partitions + num_partitions; it != end; ++it) {
                size_t end_offset = partitions[*it].next_offset;
                It partition_end = begin + end_offset;
                std::ptrdiff_t num_elements = end_offset - start_offset;
                if (!StdSortIfLessThanThreshold<StdSortThreshold>(partition_begin, partition_end, num_elements, extract_key))
                {
                    UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, NumBytes, Offset + 1>::sort(partition_begin, partition_end, num_elements, extract_key, next_sort, sort_data);
                }
                start_offset = end_offset;
                partition_begin = partition_end;
            }
        }
    }

    template<typename It, typename ExtractKey>
    static void ska_byte_sort(It begin, It end, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * sort_data)
    {
		ZoneScoped;
        PartitionInfo partitions[256];
        for (It it = begin; it != end; ++it) {
            ++partitions[current_byte(extract_key(*it), sort_data)].count;
        }

        uint8_t remaining_partitions[256];
        size_t total = 0;
        int num_partitions = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            size_t count = partitions[i].count;
            if (count) {
                partitions[i].offset = total;
                total += count;
                remaining_partitions[num_partitions] = i;
                ++num_partitions;
            }
            partitions[i].next_offset = total;
        }

        for (uint8_t * last_remaining = remaining_partitions + num_partitions, * end_partition = remaining_partitions + 1; last_remaining > end_partition;)
        {
            last_remaining = custom_std_partition(remaining_partitions, last_remaining, [&](uint8_t partition)
            {
                size_t & begin_offset = partitions[partition].offset;
                size_t & end_offset = partitions[partition].next_offset;
                if (begin_offset == end_offset)
                    return false;

                unroll_loop_four_times(begin + begin_offset, end_offset - begin_offset, [partitions = partitions, begin, &extract_key, sort_data](It it)
                {
                    uint8_t this_partition = current_byte(extract_key(*it), sort_data);
                    size_t offset = partitions[this_partition].offset++;
                    std::iter_swap(it, begin + offset);
                });
                return begin_offset != end_offset;
            });
        }
        if (Offset + 1 != NumBytes || next_sort)
        {
            for (uint8_t * it = remaining_partitions + num_partitions; it != remaining_partitions; --it)
            {
                uint8_t partition = it[-1];
                size_t start_offset = (partition == 0 ? 0 : partitions[partition - 1].next_offset);
                size_t end_offset = partitions[partition].next_offset;
                It partition_begin = begin + start_offset;
                It partition_end = begin + end_offset;
                std::ptrdiff_t num_elements = end_offset - start_offset;
                if (!StdSortIfLessThanThreshold<StdSortThreshold>(partition_begin, partition_end, num_elements, extract_key))
                {
                    UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, NumBytes, Offset + 1>::sort(partition_begin, partition_end, num_elements, extract_key, next_sort, sort_data);
                }
            }
        }
    }
};

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, size_t NumBytes>
struct UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, NumBytes, NumBytes>
{
    template<typename It, typename ExtractKey>
    inline static void sort(It begin, It end, std::ptrdiff_t num_elements, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * next_sort_data)
    {
        next_sort(begin, end, num_elements, extract_key, next_sort_data);
    }
};

template<typename It, typename ExtractKey, typename ElementKey>
size_t CommonPrefix(It begin, It end, size_t start_index, ExtractKey && extract_key, ElementKey && element_key)
{
	ZoneScoped;
    const auto & largest_match_list = extract_key(*begin);
    size_t largest_match = largest_match_list.size();
    if (largest_match == start_index)
        return start_index;
    for (++begin; begin != end; ++begin)
    {
        const auto & current_list = extract_key(*begin);
        size_t current_size = current_list.size();
        if (current_size < largest_match)
        {
            largest_match = current_size;
            if (largest_match == start_index)
                return start_index;
        }
        if (element_key(largest_match_list[start_index]) != element_key(current_list[start_index]))
            return start_index;
        for (size_t i = start_index + 1; i < largest_match; ++i)
        {
            if (element_key(largest_match_list[i]) != element_key(current_list[i]))
            {
                largest_match = i;
                break;
            }
        }
    }
    return largest_match;
}

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename ListType>
struct ListInplaceSorter
{
    using ElementSubKey = ListElementSubKey<CurrentSubKey, ListType>;
    template<typename It, typename ExtractKey>
    static void sort(It begin, It end, ExtractKey & extract_key, ListSortData<It, ExtractKey> * sort_data)
    {
		ZoneScoped;
        size_t current_index = sort_data->current_index;
        void * next_sort_data = sort_data->next_sort_data;
        auto current_key = [&](auto && elem) -> decltype(auto)
        {
            return CurrentSubKey::sub_key(extract_key(elem), next_sort_data);
        };
        auto element_key = [&](auto && elem) -> decltype(auto)
        {
            return ElementSubKey::base::sub_key(elem, sort_data);
        };
        sort_data->current_index = current_index = CommonPrefix(begin, end, current_index, current_key, element_key);
        It end_of_shorter_ones = std::partition(begin, end, [&](auto && elem)
        {
            return current_key(elem).size() <= current_index;
        });
        std::ptrdiff_t num_shorter_ones = end_of_shorter_ones - begin;
        if (sort_data->next_sort && !StdSortIfLessThanThreshold<StdSortThreshold>(begin, end_of_shorter_ones, num_shorter_ones, extract_key))
        {
            sort_data->next_sort(begin, end_of_shorter_ones, num_shorter_ones, extract_key, next_sort_data);
        }
        std::ptrdiff_t num_elements = end - end_of_shorter_ones;
        if (!StdSortIfLessThanThreshold<StdSortThreshold>(end_of_shorter_ones, end, num_elements, extract_key))
        {
            void (*sort_next_element)(It, It, std::ptrdiff_t, ExtractKey &, void *) = static_cast<void (*)(It, It, std::ptrdiff_t, ExtractKey &, void *)>(&sort_from_recursion);
            InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, ElementSubKey>::sort(end_of_shorter_ones, end, num_elements, extract_key, sort_next_element, sort_data);
        }
    }

    template<typename It, typename ExtractKey>
    static void sort_from_recursion(It begin, It end, std::ptrdiff_t, ExtractKey & extract_key, void * next_sort_data)
    {
		ZoneScoped;
        ListSortData<It, ExtractKey> offset = *static_cast<ListSortData<It, ExtractKey> *>(next_sort_data);
        ++offset.current_index;
        --offset.recursion_limit;
        if (offset.recursion_limit == 0)
        {
            StdSortFallback(begin, end, extract_key);
        }
        else
        {
            sort(begin, end, extract_key, &offset);
        }
    }


    template<typename It, typename ExtractKey>
    static void sort(It begin, It end, std::ptrdiff_t, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * next_sort_data)
    {
		ZoneScoped;
        ListSortData<It, ExtractKey> offset;
        offset.current_index = 0;
        offset.recursion_limit = 16;
        offset.next_sort = next_sort;
        offset.next_sort_data = next_sort_data;
        sort(begin, end, extract_key, &offset);
    }
};

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, bool>
{
    template<typename It, typename ExtractKey>
    static void sort(It begin,
		             It end,
		             std::ptrdiff_t,
		             ExtractKey & extract_key,
		             void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * sort_data) noexcept {
		ZoneScoped;
        It middle = std::partition(begin, end, [&](auto && a){ return !CurrentSubKey::sub_key(extract_key(a), sort_data); });
        if (next_sort)
        {
            next_sort(begin, middle, middle - begin, extract_key, sort_data);
            next_sort(middle, end, end - middle, extract_key, sort_data);
        }
    }
};

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint8_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 1>
{
};
template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint16_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 2>
{
};
template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint32_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 4>
{
};
template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint64_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 8>
{
};
template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType, typename Enable = void>
struct FallbackInplaceSorter;

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType>
struct InplaceSorter : FallbackInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, SubKeyType>
{
};

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType>
struct FallbackInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, SubKeyType, typename std::enable_if<has_subscript_operator<SubKeyType>::value>::type>
	: ListInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, SubKeyType>
{
};

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
struct SortStarter;
template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold>
struct SortStarter<StdSortThreshold, AmericanFlagSortThreshold, SubKey<void>>
{
    template<typename It, typename ExtractKey>
    static void sort(It, It, std::ptrdiff_t, ExtractKey &, void *) noexcept {}
};

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
struct SortStarter
{
    template<typename It, typename ExtractKey>
    static void sort(It begin,
		             It end,
		             std::ptrdiff_t num_elements,
		             ExtractKey & extract_key,
		             void *next_sort_data = nullptr) noexcept {
        if (StdSortIfLessThanThreshold<StdSortThreshold>(begin, end, num_elements, extract_key))
            return;

        void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *) = static_cast<void (*)(It, It, std::ptrdiff_t, ExtractKey &, void *)>(&SortStarter<StdSortThreshold, AmericanFlagSortThreshold, typename CurrentSubKey::next>::sort);
        if (next_sort == static_cast<void (*)(It, It, std::ptrdiff_t, ExtractKey &, void *)>(&SortStarter<StdSortThreshold, AmericanFlagSortThreshold, SubKey<void>>::sort))
            next_sort = nullptr;
        InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey>::sort(begin, end, num_elements, extract_key, next_sort, next_sort_data);
    }
};

template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename It, typename ExtractKey>
void inplace_radix_sort(It begin, It end, ExtractKey & extract_key)
{
    using SubKey = SubKey<decltype(extract_key(*begin))>;
    SortStarter<StdSortThreshold, AmericanFlagSortThreshold, SubKey>::sort(begin, end, end - begin, extract_key);
}

struct IdentityFunctor
{
    template<typename T>
    decltype(auto) operator()(T && i) const
    {
        return std::forward<T>(i);
    }
};
}

template<typename It, typename ExtractKey>
static void ska_sort(It begin, It end, ExtractKey && extract_key)
{
    detail::inplace_radix_sort<128, 1024>(begin, end, extract_key);
}

template<typename It>
static void ska_sort(It begin, It end)
{
    ska_sort(begin, end, detail::IdentityFunctor());
}

template<typename It, typename OutIt, typename ExtractKey>
[[nodiscard]] bool ska_sort_copy(It begin, It end, OutIt buffer_begin, ExtractKey && key)
{
    std::ptrdiff_t num_elements = end - begin;
#if defined(__clang__) && !defined (__APPLE__)
    if (num_elements < 128 || detail::radix_sort_pass_count<typename std::result_of<ExtractKey(decltype(*begin))>> >= 8)
#else 
    if (num_elements < 128 || detail::radix_sort_pass_count<typename std::invoke_result<ExtractKey(decltype(*begin))>> >= 8)
#endif
    {
        ska_sort(begin, end, key);
        return false;
    }
    else
	{

#if defined(__clang__) && !defined (__APPLE__)
        return detail::RadixSorter<typename std::result_of<ExtractKey(decltype(*begin))>::type>::sort(begin, end, buffer_begin, key);
#else 
        return detail::RadixSorter<typename std::invoke_result<ExtractKey(decltype(*begin))>::type>::sort(begin, end, buffer_begin, key);
#endif
	}
}
template<typename It, typename OutIt>
[[nodiscard]] bool ska_sort_copy(It begin, It end, OutIt buffer_begin)
{
    return ska_sort_copy(begin, end, buffer_begin, detail::IdentityFunctor());
}
