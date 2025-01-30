#pragma once


// Swap the two central blocks of 16 bits.
template <typename T>
static inline T butterfly_16(T x) {
    const T ml = 0x0000ffff00000000UL;
    const T s = 16;
    const T mr = ml >> s;
    const T t = ((x & ml) >> s ) | ((x & mr) << s );
    x = (x & ~(ml | mr)) | t;
    return  x;
}

// Swap in each block of 32 bits the two central blocks of 8 bits.
template <typename T>
static inline T butterfly_8(T x) {
    constexpr T ml = 0x00ff000000ff0000UL;
    constexpr T s = 8;
    const T mr = ml >> s;
    const T t = ((x & ml) >> s ) | ((x & mr) << s );
    x = (x & ~(ml | mr)) | t;
    return  x;
}

// Swap in each block of 16 bits the two central blocks of 4 bits.
template <typename T>
static inline T butterfly_4(T x) {
    constexpr T ml = 0x0f000f000f000f00UL;
    constexpr T s = 4;
    const T mr = ml >> s;
    const T t = ((x & ml) >> s ) | ((x & mr) << s );
    x = (x & ~(ml | mr)) | t;
    return  x;
}

// Swap in each block of 8 bits the two central blocks of 2 bits.
template <typename T>
static inline T butterfly_2(T x) {
    constexpr T ml = 0x3030303030303030UL;
    constexpr T s = 2;
    const T mr = ml >> s;
    const T t = ((x & ml) >> s ) | ((x & mr) << s );
    x = (x & ~(ml | mr)) | t;
    return  x;
}

/// Swap in each block of 4 bits the two central bits.
/// \param x[in]:
template <typename T>
static inline T butterfly_1(T x) {
    constexpr T ml = 0x4444444444444444UL;
    constexpr T s = 1;
    const T mr = ml >> s;
    const T t = ((x & ml) >> s ) | ((x & mr) << s );
    x = (x & ~(ml | mr)) | t;
    return  x;
}


// Return 64-bit word with two leftmost quarters swapped.
template <typename T>
constexpr static inline T left_swap_16(T x) noexcept {
    constexpr T mr = 0x0000ffff00000000UL;
    constexpr T s = 16;
    constexpr T ml = mr << s;
    const T t = ((x & ml) >> s) | ((x & mr) << s);
    x = (x & ~(ml | mr)) | t;
    return  x;
}

// Return word with two leftmost quarters of each 32-bit block swapped.
template <typename T>
constexpr static inline T left_swap_8(T x) noexcept {
    constexpr T mr = 0x00ff000000ff0000UL;
    constexpr T s = 8;
    constexpr T ml = mr << s;
    const T t = ((x & ml) >> s) | ((x & mr) << s);
    x = (x & ~(ml | mr)) | t;
    return  x;
}

// Return word with two leftmost quarters of each 16-bit block swapped.
template <typename T>
constexpr static inline T left_swap_4(T x) noexcept {
    constexpr T mr = 0x0f000f000f000f00UL;
    constexpr T s = 4;
    constexpr T ml = mr << s;
    const T t = ((x & ml) >> s) | ((x & mr) << s);
    x = (x & ~(ml | mr)) | t;
    return  x;
}

// Return word with two leftmost quarters of each 8-bit block swapped.
template <typename T>
constexpr static inline T left_swap_2(T x) noexcept {
    constexpr T mr = 0x3030303030303030UL;
    constexpr T s = 2;
    constexpr T ml = mr << s;
    const T t = ((x & ml) >> s) | ((x & mr) << s);
    x = (x & ~(ml | mr)) | t;
    return  x;
}

// Return word with two leftmost bits of each 4-bit block swapped.
template <typename T>
constexpr static inline T left_swap_1(T x) noexcept {
    constexpr T mr = 0x4444444444444444UL;
    constexpr T s = 1;
    constexpr T ml = mr << s;
    const T t = ((x & ml) >> s) | ((x & mr) << s);
    x = (x & ~(ml | mr)) | t;
    return  x;
}
