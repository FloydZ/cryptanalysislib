# TODOs: src-simd

58 entries. Format: `- [ ] file:line [MARKER] text`.

## src/simd/README.md

- [ ] `src/simd/README.md:40` **TODO** `=====`

## src/simd/avx2.h

- [ ] `src/simd/avx2.h:483` **TODO** optimize — `[[nodiscard]] constexpr static inline S div(const S in1,`
- [ ] `src/simd/avx2.h:2444` **TODO** impl: https://github.com/ridiculousfish/libdivide/blob/master/libdivide.h#L144
- [ ] `src/simd/avx2.h:3050` **TODO** https://stackoverflow.com/questions/16822757/sse-integer-division
- [ ] `src/simd/avx2.h:3259` **TODO** https://stackoverflow.com/questions/58823140/count-leading-zero-bits-for-each-element-in-avx2-vector-emulate-mm256-lzcnt-ep
- [ ] `src/simd/avx2.h:3268` **TODO** `(void)in;`
- [ ] `src/simd/avx2.h:3309` **TODO** `[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {`
- [ ] `src/simd/avx2.h:4770` **TODO** probably move somewhere useful `transpose.h`?
- [ ] `src/simd/avx2.h:4797` **TODO** probably move somewhere useful `transpose.h`?

## src/simd/avx512.h

- [ ] `src/simd/avx512.h:1675` **TODO** opt — checks if all bytes are equal
- [ ] `src/simd/avx512.h:1691` **TODO** optimize — `[[nodiscard]] constexpr static inline S permute(const S in,`
- [ ] `src/simd/avx512.h:2138` **TODO** `[[nodiscard]] constexpr static inline S div(const S in1,`
- [ ] `src/simd/avx512.h:2667` **TODO** extend to avx2
- [ ] `src/simd/avx512.h:2677` **TODO** extend to avx2
- [ ] `src/simd/avx512.h:2820` **TODO** `[[nodiscard]] constexpr static inline S div(const S in1,`
- [ ] `src/simd/avx512.h:3151` **TODO** test and implement for uint32x16 and so on and implement hadd_epu16,...
- [ ] `src/simd/avx512.h:3162` **TODO** test, and implement for uint32x16 and so on
- [ ] `src/simd/avx512.h:3220` **TODO** make generic

## src/simd/avx512_intrinsics.h

- [ ] `src/simd/avx512_intrinsics.h:5` **TODO** docs — src: https://github.com/InstLatx64/InstLatX64_Demo/

## src/simd/generic.h

- [ ] `src/simd/generic.h:619` **TODO** all these function should return a binarycontainer
- [ ] `src/simd/generic.h:642` **TODO** move this into a seperate call in this class, so
- [ ] `src/simd/generic.h:677` **TODO** doc — \param in1
- [ ] `src/simd/generic.h:689` **TODO** doc — \param in1
- [ ] `src/simd/generic.h:702` **TODO** doc — \param ptr
- [ ] `src/simd/generic.h:721` **TODO** doc — \param ptr
- [ ] `src/simd/generic.h:738` **TODO** doc — \param ptr

## src/simd/neon.h

- [ ] `src/simd/neon.h:474` **TODO** optimize — `[[nodiscard]] constexpr static inline S div(const S in1,`
- [ ] `src/simd/neon.h:1637` **TODO** not correct, carry and sruff
- [ ] `src/simd/neon.h:1998` **TODO** \param in
- [ ] `src/simd/neon.h:2020` **TODO** not optimized
- [ ] `src/simd/neon.h:2034` **TODO** not optimized
- [ ] `src/simd/neon.h:2705` **TODO** \param in
- [ ] `src/simd/neon.h:2736` **TODO** \param a
- [ ] `src/simd/neon.h:2750` **TODO** \param a
- [ ] `src/simd/neon.h:3045` **TODO** not correct, carry and sruff
- [ ] `src/simd/neon.h:3431` **TODO** arm instruction
- [ ] `src/simd/neon.h:3465` **TODO** \param in
- [ ] `src/simd/neon.h:3499` **TODO** \param in
- [ ] `src/simd/neon.h:3933` **TODO** `return ret;`
- [ ] `src/simd/neon.h:4191` **TODO** \param ptr
- [ ] `src/simd/neon.h:4210` **TODO** \param ptr
- [ ] `src/simd/neon.h:4229` **TODO** \tparam in2
- [ ] `src/simd/neon.h:4261` **TODO** \param in

## src/simd/simd.h

- [ ] `src/simd/simd.h:12` **TODO** #include "swar.h"
- [ ] `src/simd/simd.h:18` **TODO** whats that? — define bit_shuffle_const(b0, b1, b2, b3, b4, b5, b6, b7) \
- [ ] `src/simd/simd.h:5486` **TODO** signed type — `namespace cryptanalysislib {`
- [ ] `src/simd/simd.h:5592` **TODO** store
- [ ] `src/simd/simd.h:5613` **TODO** { S::eq_(s, s) } -> std::convertible_to<S>;
- [ ] `src/simd/simd.h:5614` **TODO** { S::eq(s, s) } -> std::convertible_to<typename S::limb_type>;

## src/simd/swar.h

- [ ] `src/simd/swar.h:7` **TODO** copy tests from  https://github.com/qlibs/swar/blob/main/swar and add them
- [ ] `src/simd/swar.h:40` **TODO** `template<class T,`
- [ ] `src/simd/swar.h:72` **TODO** `template<typename T,`
- [ ] `src/simd/swar.h:98` **TODO** doc — `constexpr explicit swar(const auto& gen) noexcept requires requires(size_t i) { gen(i); } {`
- [ ] `src/simd/swar.h:104` **TODO** doc — `[[nodiscard]] constexpr explicit operator abi_type() const noexcept {`
- [ ] `src/simd/swar.h:108` **TODO** doc — `[[nodiscard]] constexpr auto operator[](const size_t index) const noexcept -> T {`
- [ ] `src/simd/swar.h:113` **TODO** doc — `[[nodiscard]] static constexpr auto size() noexcept -> size_t {`
- [ ] `src/simd/swar.h:117` **TODO** doc — `[[nodiscard]] friend constexpr`
- [ ] `src/simd/swar.h:128` **TODO** use cryptanalysislib functions
