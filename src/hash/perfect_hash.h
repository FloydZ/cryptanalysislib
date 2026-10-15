// The MIT License (MIT)
//
// Copyright (c) 2024 Kris Jusiak <kris@jusiak.net>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// [Minimal] static perfect hash functions:
/// https://github.com/qlibs/mph (v5.0.5), in namespace cryptanalysislib::mph.
/// The `$` in the identifiers is replaced by `_` (`lookup$pext` -> `lookup_pext`),
/// the self tests are in `tests/hash/perfect_hash.cpp`.
/// Differences to upstream: `find` compares the keys as `key_type` (upstream
/// truncates them to the 32 bit mask type, which misses keys >= 2^32 and
/// strings longer than 4 chars), `find_simd` does not read `values` at
/// `find_first_set` of an empty match, and `find_simd` is only enabled if
/// `<experimental/simd>` is implemented (not in libc++).
///
/// Usage:
/// \code
/// 	enum class color { red, green, blue };
/// 	constexpr auto colors = std::array{
/// 		std::pair{"red"sv, color::red},
/// 		std::pair{"green"sv, color::green},
/// 		std::pair{"blue"sv, color::blue},
/// 	};
/// 	// only valid keys, returns the mapped value
/// 	static_assert(color::green == cryptanalysislib::mph::lookup<colors>("green"));
/// 	// any key, returns an optional of the mapped value
/// 	static_assert(not cryptanalysislib::mph::find<colors>("unknown"));
/// 	static_assert(*cryptanalysislib::mph::find<colors>("red") == color::red);
/// \endcode
/// `lookup` searches a magic number `m` and a 32/64 bit lookup table `lut`,
/// s.t. `(lut >> ((key * m) >> shift)) & mask` is the value of `key`. If none
/// exists, the lookup table is indexed by `pext(key, mask)`, where `mask` is
/// a minimal set of bits which distinguishes all keys.
/// `find` stores the key next to the value and compares it. Up to 64 keys
/// via `pext`, above via buckets of SIMD registers (`<experimental/simd>`,
/// libstdc++ only).
/// All keys must fit into 128 bits (also strings). `find` with more than 64
/// keys does not support the key 0.
/// NOTE: the tables are computed at compile time, for many keys increase
/// 	`-fconstexpr-ops-limit=N` (gcc) / `-fconstexpr-steps=N` (clang).

#pragma once
#pragma GCC system_header

#if __has_include(<experimental/simd>)
#include <experimental/simd>
#endif

// NOTE: libc++ has the header, but without the implementation
#if defined(__cpp_lib_experimental_parallel_simd)
#define CRYPTANALYSISLIB_MPH_SIMD 1
#endif

namespace cryptanalysislib::mph {
using i8   = __INT8_TYPE__;
using u8   = __UINT8_TYPE__;
using i16  = __INT16_TYPE__;
using u16  = __UINT16_TYPE__;
using i32  = __INT32_TYPE__;
using u32  = __UINT32_TYPE__;
using i64  = __INT64_TYPE__;
using u64  = __UINT64_TYPE__;
#if defined(__SIZEOF_INT128__)
using i128 = __int128;
using u128 = unsigned __int128;
#endif

namespace utility {
template<class T1, class T2>
struct compressed_pair {
  using first_type = T1;
  using second_type = T2;
  [[no_unique_address]] T1 first;
  [[no_unique_address]] T2 second;
};
template<class... Ts> compressed_pair(Ts...) -> compressed_pair<Ts...>;

template<class T, u32 Size>
struct array {
  using value_type = T;
  [[nodiscard]] constexpr const T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr T* data() noexcept { return data_; }
  [[nodiscard]] constexpr auto begin() const noexcept { return &data_[0]; }
  [[nodiscard]] constexpr auto begin() noexcept { return &data_[0]; }
  [[nodiscard]] constexpr auto end() const noexcept { return &data_[0] + Size; }
  [[nodiscard]] constexpr auto end() noexcept { return &data_[0] + Size; }
  [[nodiscard]] constexpr auto size() const noexcept { return Size; }
  [[nodiscard]] constexpr const auto& operator[](u32 i) const noexcept { return data_[i]; }
  [[nodiscard]] constexpr auto& operator[](u32 i) noexcept { return data_[i]; }
  constexpr void fill(const T& value) noexcept { for (auto i = 0u; i < Size; ++i) { data_[i] = value; } }
  T data_[Size];
};
template<class T, class... Ts> array(T, Ts...) -> array<T, 1u + sizeof...(Ts)>;
} // namespace utility

namespace type_traits {
namespace detail {
template<bool> struct conditional;
template<> struct conditional<false> { template <class, class T> using fn = T; };
template<> struct conditional<true>  { template <class T, class> using fn = T; };
template<class T> auto value_type(const T&) -> typename T::value_type;
} // namespace detail

template<bool B, class T, class F>
using conditional_t = typename detail::conditional<B>::template fn<T, F>;

template<const auto& v>
using value_type_t = decltype(detail::value_type(v));

template<auto v> struct constant {
  static constexpr auto value = v;
  constexpr operator auto() const noexcept { return value; }
};
template<auto v> inline constexpr constant<v> constant_v{};
template<class T> struct constant_t { static constexpr T value{}; };
} // namespace type_traits

namespace random {
// https://www.pcg-random.org
struct pcg {
  [[nodiscard]] constexpr auto operator()() noexcept -> u32 {
    u64 old_state = state;
    state = old_state * 6364136223846793005u + increment;
    u32 xor_shifted = ((old_state >> 18u) ^ old_state) >> 27u;
    u32 rot = u32(old_state >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31u));
  }
  u64 increment{1442695040888963407u};
  u64 state{5573589319906701683u + increment};
};
} // namespace random

template<u8 probability> requires (probability >= 0u and probability <= 100u)
inline constexpr auto conditional =
  []<class T>([[maybe_unused]] const bool cond, [[maybe_unused]] const T lhs, [[maybe_unused]] const T rhs) {
    if constexpr (probability == 0u) {
      return rhs;
    } else if constexpr (probability == 100u) {
      return lhs;
    } else if constexpr (probability == 50u) {
        #if __has_builtin(__builtin_unpredictable)
        if (__builtin_is_constant_evaluated()) {
          return cond ? lhs : rhs;
        } else if (__builtin_unpredictable(cond)) {
          return lhs;
        } else {
          return rhs;
        }
        #else
        constexpr auto underlying = [](const auto value) {
          if constexpr (__is_enum(decltype(value))) {
            return static_cast<__underlying_type(decltype(value))>(value);
          } else {
            return value;
          }
        };
        if constexpr (requires { T(cond * underlying(lhs.value), cond); }) {
          return T(cond * underlying(lhs.value), cond);
        } else if constexpr (requires { T(cond * underlying(lhs)); }) {
          return T(cond * underlying(lhs));
        } else {
          return __builtin_expect_with_probability(cond, 1, .5f) ? lhs : rhs;
        }
        #endif
    } else {
      return __builtin_expect_with_probability(cond, 1,
        [] { return float(probability) / float(100u); }()) ? lhs : rhs;
    }
  };

template<class T>
[[nodiscard]] [[gnu::always_inline]] constexpr auto to(const auto& data) noexcept -> T {
  if (__builtin_is_constant_evaluated()) {
    if constexpr (requires { data.data(); data.size(); }) {
      T t{};
      for (auto i = 0u; i < data.size(); ++i) {
        t = (t << __CHAR_BIT__) | data.data()[data.size() - i - 1u];
      }
      return t;
    } else if constexpr (requires(u32 n) { data[n]; }) {
      u32 size{};
      auto ptr = data;
      while (*ptr++) size++;
      T t{};
      for (auto i = 0u; i < size; ++i) {
        t = (t << __CHAR_BIT__) | data[size - i - 1u];
      }
      return t;
    } else {
      return static_cast<T>(data);
    }
  } else if constexpr (requires { static_cast<T>(data); } ) {
    return static_cast<T>(data);
  } else if constexpr (requires { data.data(); data.size(); }) {
    if constexpr (requires { []<template<class, auto> class T_, class _, auto size>(T_<_, size>){}(data); }) {
      return [&]<template<class, auto> class T_, class _, auto size>(T_<_, size>) {
        if constexpr (not size) {
          return T{};
        } else if constexpr (size == sizeof(T)) {
          return *__builtin_bit_cast(const T*, data.data());
        } else if constexpr (size <= sizeof(T)) {
          T t{};
          __builtin_memcpy(&t, data.data(), size);
          return t;
        }
      }(data);
    } else {
      #if defined (MPH_PAGE_SIZE)
      // https://github.com/bminor/glibc/blob/master/sysdeps/generic/memcopy.h#L162
      if ((u64(data.data()) & (MPH_PAGE_SIZE - 1ul)) > (MPH_PAGE_SIZE - sizeof(T))) [[unlikely]] {
        return [&data] [[gnu::cold]] {
          T t{};
          __builtin_memcpy(&t, data.data(), data.size());
          return t;
        }();
      }
      #endif
      T t{};
      __builtin_memcpy(&t, data.data(), data.size() < sizeof(t) ? data.size() : sizeof(t));
      const auto index = T(data.size() * __CHAR_BIT__);
      #ifdef __BMI2__
      if constexpr (sizeof(t) <= sizeof(u32)) {
        return __builtin_ia32_bzhi_si(t, index);
      } else if constexpr (sizeof(t) <= sizeof(u64)) {
        return __builtin_ia32_bzhi_di(t, index);
      } else {
      #endif
        constexpr T size = sizeof(T) * __CHAR_BIT__;
        return index >= size ? t : t & ((T(1) << index) - T(1));
      #ifdef __BMI2__
      }
      #endif
    }
  } else { // unsafe
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Warray-bounds"
    return *reinterpret_cast<const T*>(data);
    #pragma GCC diagnostic pop
  }
  __builtin_unreachable();
}

namespace detail {
template<class T, T size = sizeof(T) * __CHAR_BIT__>
[[nodiscard]] [[gnu::always_inline]] constexpr auto pext(const T a, const auto mask) noexcept -> T {
  #ifdef __BMI2__
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=pext
  if (__builtin_is_constant_evaluated()) {
    T result{};
    T m = mask;
    auto k = 0u;
    for (T i{}; i < size; ++i) {
      if (m & 1) result |= ((a >> i) & 1) << k++;
      m >>= 1;
    }
    return result;
  }
  else if constexpr (requires { u32{mask}; }) { return __builtin_ia32_pext_si(a, mask); }
  else if constexpr (requires { u64{mask}; }) { return __builtin_ia32_pext_di(a, mask); }
  else
  #endif
  if constexpr (requires { mask.value; }) {
    // https://stackoverflow.com/questions/14547087/extracting-bits-with-a-single-multiplication
    // https://github.com/intel/compile-time-init-build/blob/main/include/lookup/pseudo_pext_lookup.hpp
    constexpr auto clz = [](auto x) {
      decltype(x) n{sizeof(x) * __CHAR_BIT__ - 1};
      if (not x) return decltype(x)(sizeof(x) * __CHAR_BIT__);
      while (x >>= 1u) n--;
      return n;
    };
    constexpr auto mbits = __builtin_popcountl(mask);
    constexpr auto nbits = size - mbits - clz(mask.value);
    constexpr auto coefficient = [&] {
      auto set = false;
      auto dst = nbits;
      T result{};
      for (auto i = 0u; i < size; ++i) {
        const auto curr = ((T(1) << i) & mask) != T();
        if (curr and not set) result = result | (T(1) << (dst - i));
        dst += curr;
        set = curr;
      }
      return result;
    }();
    if constexpr(constexpr auto final_mask = ((T(1) << mbits) - T(1));
      ((((mask) * coefficient) >> nbits) & final_mask) == final_mask) {
      return (((a & mask) * coefficient) >> nbits) & final_mask;
    }
  }
  T result{};
  T m = mask;
  auto k = 0u;
  for (T i{}; i < size; ++i) {
    if (m & 1) result |= ((a >> i) & 1) << k++;
    m >>= 1;
  }
  return result;
}

template<class T, u32 N = 1u>
[[nodiscard]] constexpr auto mask(const auto& entries) noexcept -> T {
  utility::array<T, decltype(entries){}.size()> vs;
  T max{};
  for (auto i = 0u; i < vs.size(); ++i) {
    if (vs[i] = entries[i].first; vs[i] > max) max = vs[i];
  }
  constexpr auto H = (N * vs.size()) << 1u;
  constexpr auto N_ = N - 1u;
  const auto lz = sizeof(max) <= sizeof(u32) ? __builtin_clz(max) : __builtin_clzl(max);
  const auto nbits = sizeof(T) * __CHAR_BIT__ - lz - 1u;
  utility::array<T, H> hashed;
  T mask = ((T(1) << nbits) - 1u);
  for (int i = nbits; i >= 0; --i) {
    mask &= ~(T(1) << i);
    hashed = {};
    for (auto j = 0u; j < vs.size(); ++j) {
      const T masked = (vs[j] & mask) + 1u;
      auto slot = masked % H;
      auto n = N_;
      auto found = false;
      while (hashed[slot]) {
        if (hashed[slot] == masked and not n--) {
          found = true;
          break;
        }
        slot = (slot + 1u) % H;
      }
      if (found) {
        mask |= (T(1) << i);
        break;
      }
      hashed[slot] = masked;
    }
  }
  return mask;
}

template<auto entries>
[[nodiscard]] constexpr auto narrow_mask() noexcept {
    static constexpr auto wide_mask = mask<u64>(entries);
    // NOTE: not `UINT32_MAX`, `<cstdint>` is not included
    if constexpr (wide_mask <= u32(~u32{})) {
        return static_cast<u32>(wide_mask);
    } else {
        return wide_mask;
    }
}

template<const auto& entries>
struct traits {
  static constexpr auto mappings = [] {
    if constexpr (requires { entries[0].size(); }) { // array{"a"sv, "b"sv, ...}
      constexpr auto max_len = [] {
        u32 len{};
        for (const auto& entry : entries) { if (entry.size() > len) { len = entry.size(); } }
        return len;
      }();

      using key_type = decltype([] {
            if constexpr (max_len <= sizeof(u8)) { return u8{}; }
        else if constexpr (max_len <= sizeof(u16)) { return u16{}; }
        else if constexpr (max_len <= sizeof(u32)) { return u32{}; }
        else if constexpr (max_len <= sizeof(u64)) { return u64{}; }
        #if defined(__SIZEOF_INT128__)
        else if constexpr (max_len <= sizeof(u128)) { return u128{}; }
        #endif
      }());

      using mapped_type = decltype([] {
             if constexpr (entries.size() < u8(u8{}-1u)) { return u8{}; }
        else if constexpr (entries.size() < u16(u16{}-1u)) { return u16{}; }
        else if constexpr (entries.size() < u32(u32{}-1u)) { return u32{}; }
        else if constexpr (entries.size() < u64(u64{}-1u)) { return u64{}; }
        #if defined(__SIZEOF_INT128__)
        else if constexpr (entries.size() < u128(u128{}-1u)) { return u128{}; }
        #endif
      }());

      utility::array<utility::compressed_pair<key_type, mapped_type>, entries.size()> entries_;
      for (mapped_type i{}; i < entries.size(); ++i) {
        entries_[i] = {to<key_type>(entries[i]), mapped_type(i)};
      }
      return entries_;
    } else if constexpr (requires (u32 n) { entries[n].first.size(); entries[n].second; }) { // array{pair{"a"sv, 1}, ...}
      constexpr auto max_len = [] {
        u32 len{};
        for (const auto& entry : entries) { if (entry.first.size() > len) { len = entry.first.size(); } }
        return len;
      }();

      using key_type = decltype([] {
            if constexpr (max_len <= sizeof(u8)) { return u8{}; }
        else if constexpr (max_len <= sizeof(u16)) { return u16{}; }
        else if constexpr (max_len <= sizeof(u32)) { return u32{}; }
        else if constexpr (max_len <= sizeof(u64)) { return u64{}; }
        #if defined(__SIZEOF_INT128__)
        else if constexpr (max_len <= sizeof(u128)) { return u128{}; }
        #endif
      }());
      using mapped_type = typename type_traits::value_type_t<entries>::second_type;

      utility::array<utility::compressed_pair<key_type, mapped_type>, entries.size()> entries_;
      for (auto i = 0u; i < entries.size(); ++i) {
        entries_[i] = {to<key_type>(entries[i].first), entries[i].second};
      }
      return entries_;
    } else if constexpr (not requires (u32 n) { entries[n].first; entries[n].second; }) { // array{1, 2, 3, ...}
      using key_type = type_traits::value_type_t<entries>;
      using mapped_type = decltype([] {
             if constexpr (entries.size() < u8(u8{}-1u)) { return u8{}; }
        else if constexpr (entries.size() < u16(u16{}-1u)) { return u16{}; }
        else if constexpr (entries.size() < u32(u32{}-1u)) { return u32{}; }
        else if constexpr (entries.size() < u64(u64{}-1u)) { return u64{}; }
        #if defined(__SIZEOF_INT128__)
        else if constexpr (entries.size() < u128(u128{}-1u)) { return u128{}; }
        #endif
      }());

      utility::array<utility::compressed_pair<key_type, mapped_type>, entries.size()> entries_;
      for (auto i = 0u; i < entries.size(); ++i) {
        entries_[i] = {entries[i], mapped_type(i)};
      }
      return entries_;
    } else if constexpr (__is_enum(typename type_traits::value_type_t<entries>::first_type)) {
      using key_type = __underlying_type(typename type_traits::value_type_t<entries>::first_type);
      using mapped_type = typename type_traits::value_type_t<entries>::second_type;

      utility::array<utility::compressed_pair<key_type, mapped_type>, entries.size()> entries_;
      for (auto i = 0u; i < entries.size(); ++i) {
        entries_[i] = {static_cast<key_type>(entries[i].first), entries[i].second};
      }
      return entries_;
    } else { // array{pair{1, 2}, ...}
      return entries;
    }
  }();
  using key_type = typename type_traits::value_type_t<mappings>::first_type;
  using mapped_type = typename type_traits::value_type_t<mappings>::second_type;
  using key_mapped_type = utility::compressed_pair<key_type, mapped_type>;
};
} // namespace detail

template<class T>
struct optional {
  using value_type = T;
  constexpr optional() noexcept = default;
  constexpr optional(const value_type value, const bool flag = true) noexcept
    : value(value), flag(flag)
  { }
  [[nodiscard]] constexpr operator bool() const noexcept { return flag; }
  [[nodiscard]] constexpr auto operator*() const noexcept { return value; }
  T value{};
  bool flag{};
};

template<const auto& entries>
struct lookup_magic_lut {
  using key_type = typename detail::traits<entries>::key_type;
  using mapped_type = typename detail::traits<entries>::mapped_type;
  using value_type = type_traits::conditional_t<sizeof(key_type) <= sizeof(u32), u32, u64>;
  using result_type = mapped_type;

  constexpr explicit lookup_magic_lut(u32 max_attempts = 100'000) noexcept {
    const auto nbits = [] {
      value_type max{};
      for (const auto& [_, v]: detail::traits<entries>::mappings) {
        if (value_type(v) > max) max = value_type(v);
      }
      return max ? (sizeof(value_type) * __CHAR_BIT__ - __builtin_clz(max)) : value_type{};
    }();

    if (nbits * entries.size() > sizeof(value_type) * __CHAR_BIT__) {
      return;
    }

    mask = (1u << nbits) - 1u;
    shift = sizeof(value_type) * __CHAR_BIT__ - nbits;

    random::pcg random{};
    while (not lut and max_attempts--) {
      [&] {
        magic = random();
        for (const auto& [k, v] : detail::traits<entries>::mappings) {
          const auto shl = value_type(k * magic) >> shift;
          if (shl >= sizeof(value_type) * __CHAR_BIT__) {
            lut = {};
            return;
          }
          lut |= value_type(v) << shl;
        }
        for (const auto& [k, v] : detail::traits<entries>::mappings) {
          if (((lut >> (value_type(k * magic) >> shift)) & mask) != value_type(v)) {
            lut = {};
            return;
          }
        }
      }();
    }
  }

  [[nodiscard]] constexpr auto operator()(const auto& key) const noexcept -> result_type {
    return (lut >> ((to<key_type>(key) * magic) >> shift)) & mask;
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return mask and shift and magic and lut;
  }

  value_type mask{};
  value_type shift{};
  value_type magic{};
  value_type lut{};
};

template<const auto& entries>
struct lookup_pext {
  using key_type = typename detail::traits<entries>::key_type;
  using mapped_type = typename detail::traits<entries>::mapped_type;
  static constexpr auto mask = type_traits::constant_v<detail::narrow_mask<detail::traits<entries>::mappings>()>;
  using mask_type = type_traits::conditional_t<sizeof(mask.value) <= sizeof(u32), u32, u64>;
  using result_type = mapped_type;

  constexpr lookup_pext() noexcept {
    for (const auto& [k, v] : detail::traits<entries>::mappings) {
      lut[detail::pext(k, mask)] = v;
    }
  }

  [[nodiscard]] constexpr auto operator()(const auto& key) const noexcept -> result_type {
    return lut[detail::pext(to<mask_type>(key), mask)];
  }

  static constexpr auto size = [] {
    key_type max{};
    for (const auto& [k, _] : detail::traits<entries>::mappings) {
      if (const auto p = detail::pext(k, mask); p > max) {
        max = p;
      }
    }
    return max + 1u;
  }();
  utility::array<mapped_type, size> lut{};
};

template<const auto& entries>
struct find_pext {
  using key_type = typename detail::traits<entries>::key_type;
  using mapped_type = typename detail::traits<entries>::mapped_type;
  using key_mapped_type = utility::compressed_pair<key_type, mapped_type>;
  static constexpr auto mask = type_traits::constant_v<detail::narrow_mask<detail::traits<entries>::mappings>()>;
  using mask_type = type_traits::conditional_t<sizeof(mask.value) <= sizeof(u32), u32, u64>;
  using result_type = optional<mapped_type>;

  constexpr find_pext() noexcept {
    for (const auto& [key, value] : detail::traits<entries>::mappings) {
      lut[detail::pext(key, mask)] = {key, value};
    }
  }

  /// NOTE: the key is compared as `key_type`: `mask_type` is only 32 bits
  ///   if all distinguishing bits are in the low 32 bits of the keys
  template<u8 probability = 50u> requires (probability >= 0u and probability <= 100u)
  [[nodiscard]] constexpr auto operator()(const auto& key, const auto&... ts) const noexcept -> result_type {
    auto&& lhs = to<key_type>(key);
    auto&& [rhs, value] = lut[detail::pext(mask_type(lhs), mask)];
    return conditional<probability>(lhs == rhs, result_type(value), result_type(ts...));
  }

  utility::array<key_mapped_type, mask_type(1) << __builtin_popcountl(mask)> lut{};
};

#ifdef CRYPTANALYSISLIB_MPH_SIMD
template<const auto& entries, u32 BucketSize>
struct find_simd {
  using key_type = typename detail::traits<entries>::key_type;
  using mapped_type = typename detail::traits<entries>::mapped_type;
  using key_mapped_type = utility::compressed_pair<key_type, mapped_type>;
  static constexpr auto mask = type_traits::constant_v<detail::narrow_mask<detail::traits<entries>::mappings>()>;
  using mask_type = type_traits::conditional_t<sizeof(mask.value) <= sizeof(u32), u32, u64>;
  using result_type = optional<mapped_type>;

  constexpr find_simd() noexcept {
    for (const auto& [key, value] : detail::traits<entries>::mappings) {
      auto slot = BucketSize * detail::pext(key, mask);
      while (keys[slot]) slot++;
      keys[slot] = key;
      values[slot] = value;
    }
  }

  /// NOTE: the key is compared as `key_type` (see `find_pext`). Empty slots
  ///   are marked by the key 0, hence 0 is not supported as a key.
  ///   `find_first_set` is undefined without a match, so `values` is only
  ///   read at a matching lane (or the first one).
  template<u8 probability = 50u> requires (probability >= 0u and probability <= 100u)
  [[nodiscard]] constexpr auto operator()(const auto& key, const auto&... ts) const noexcept -> result_type {
    auto&& value = to<key_type>(key);
    auto&& index = BucketSize * detail::pext(mask_type(value), mask);
    const std::experimental::fixed_size_simd<key_type, BucketSize> lhs{value};
    const std::experimental::fixed_size_simd<key_type, BucketSize> rhs{&keys[index], std::experimental::element_aligned};
    const auto match = lhs == rhs;
    const bool found = std::experimental::any_of(match) and (value != key_type{});
    const auto lane = found ? std::experimental::find_first_set(match) : 0;
    return conditional<probability>(
      found,
      result_type(values[index + lane]),
      result_type(ts...)
    );
  }

  static constexpr auto size = BucketSize * (mask_type(1) << __builtin_popcountl(mask));
  utility::array<key_type, size> keys{};
  utility::array<mapped_type, size> values{};
};
#endif

/**
 * Static [minimal] perfect hash lookup function
 * @tparam entries constexpr array of keys or key/value pairs
 */
template<const auto& entries> inline constexpr auto lookup =
  []([[maybe_unused]] const auto& key) {
    using key_type = typename detail::traits<entries>::key_type;
    using mapped_type = typename detail::traits<entries>::mapped_type;
    using result_type = mapped_type;
    if constexpr (entries.size() == 0u) {
      return result_type{};
    } else if constexpr (entries.size() == 1u) {
      return result_type(detail::traits<entries>::mappings[0].second);
    } else if constexpr (constexpr lookup_magic_lut<entries> lookup{}; lookup) {
      return lookup(key);
    } else {
      return type_traits::constant_t<lookup_pext<entries>>::value(key);
    }
  };

/**
 * Static perfect hash find function
 * @tparam entries constexpr array of keys or key/value pairs
 */
template<const auto& entries> inline constexpr auto find =
  []<u8 probability = 50u>
    requires (probability >= 0u and probability <= 100u)
    #ifndef CRYPTANALYSISLIB_MPH_SIMD
    and (entries.size() <= 64u)
    #endif
    ([[maybe_unused]] const auto& key, [[maybe_unused]] const auto&... ts) {
    using key_type = typename detail::traits<entries>::key_type;
    using mapped_type = typename detail::traits<entries>::mapped_type;
    using result_type = optional<mapped_type>;
    if constexpr (entries.size() == 0u) {
      return result_type(ts...);
    } else if constexpr (entries.size() == 1u) {
      auto&& lhs = to<key_type>(key);
      auto&& [rhs, value] = detail::traits<entries>::mappings[0];
      return conditional<probability>(lhs == rhs, result_type(value), result_type(ts...));
    } else if constexpr (entries.size() <= 64u) {
      return type_traits::constant_t<find_pext<entries>>::value.template
        operator()<probability>(key, ts...);
    }
    #ifdef CRYPTANALYSISLIB_MPH_SIMD
    else {
      constexpr auto bucket_size = std::experimental::simd_size_v<key_type, std::experimental::simd_abi::native<key_type>>;
      return type_traits::constant_t<find_simd<entries, bucket_size>>::value.template
        operator()<probability>(key, ts...);
    }
    #endif
  };
} // namespace cryptanalysislib::mph
