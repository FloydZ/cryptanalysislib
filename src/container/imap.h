#ifndef CRYPTANALYSISLIB_CONTAINER_IMAP_H
#define CRYPTANALYSISLIB_CONTAINER_IMAP_H

#include <cstdint>

#include "helper.h"
#include "simd/simd.h"
#include "alloc/alloc.h"
#include "popcount/popcount.h"

using namespace cryptanalysislib::popcount;
   
// TODO all those structs iDownloadsnto the imap container
// TODO iterators
// TODO remove? use __uint128?
typedef struct imap_u128 {
    uint64_t v[2];
} imap_u128_t;


struct imap_node_t {
    /* 64 bytes */
    union {
        uint32_t vec32[16];
        uint64_t vec64[8];
        imap_u128 vec128[4];
    };
};

struct imap_iter_t {
    uint32_t stack[16];
    uint32_t stackp;
};

struct imap_pair_t {
    uint64_t x;
    // imap_slot_t *slot;
	uint32_t *slot;
};

/// TODO replace with allocator
static inline
void *imap__aligned_alloc__(uint64_t alignment,
                            const uint64_t size) {
    void *p = malloc(size + sizeof(void *) + alignment - 1);
    if (!p) {
        return p;
	}
    void **ap = (void **)(((uint64_t)p + sizeof(void *) + alignment - 1) & ~(alignment - 1));
    ap[-1] = p;
    return ap;
}

/// TODO replace with allocator
static inline void imap__aligned_free__(void *p) {
    if (nullptr != p) {
        free(((void **)p)[-1]);
	}
}

#define IMAP_ALIGNED_ALLOC(a, s)    (imap__aligned_alloc__(a, s))
#define IMAP_ALIGNED_FREE(p)        (imap__aligned_free__(p))


struct ImapConfig : public AlignmentConfig {
};


struct imap_tree_t {
private:
	imap_node_t *tree;

public:
	/// TODO move to math.h
	constexpr static inline
    uint32_t imap__bsr__(const uint64_t x) noexcept {
        return 63u - __builtin_clzll(x | 1u);
    }


	/// TODO move to math.
    constexpr static inline
    uint64_t imap__ceilpow2__(const uint64_t x) noexcept {
        return 1ull << (imap__bsr__(x - 1) + 1);
    }


#if defined (USE_AVX2)
    static inline
    uint64_t imap__extract_lo4_simd__(const uint32_t vec32[16]) noexcept {
#if defined (USE_AVX512F)
        __m512i vecmm = _mm512_load_epi32(vec32);
        vecmm = _mm512_and_epi32(vecmm, _mm512_set1_epi32(0xf));
        vecmm = _mm512_sllv_epi64(vecmm, _mm512_setr_epi64(0, 4, 8, 12, 16, 20, 24, 28));
        return _mm512_reduce_add_epi64(vecmm);
#else
        __m256i veclo = _mm256_load_si256((__m256i *) vec32);
        __m256i vechi = _mm256_load_si256((__m256i *)(vec32 + 8));
        __m256i mskmm = _mm256_set1_epi32(0xf);
        veclo = _mm256_and_si256(veclo, mskmm);
        vechi = _mm256_and_si256(vechi, mskmm);
        veclo = _mm256_sllv_epi64(veclo, _mm256_setr_epi64x(0, 4, 8, 12));
        vechi = _mm256_sllv_epi64(vechi, _mm256_setr_epi64x(16, 20, 24, 28));
        veclo = _mm256_add_epi64(veclo, vechi);
        veclo = _mm256_add_epi64(veclo, _mm256_permute4x64_epi64(veclo, _MM_SHUFFLE(0, 1, 2, 3)));
        return _mm256_extract_epi64(veclo, 0) + _mm256_extract_epi64(veclo, 1);
#endif
    }

	    static inline
    void imap__deposit_lo4_simd__(const uint32_t vec32[16],
							 	  const uint64_t value) noexcept {
#if defined (USE_AVX512F)
        __m512i vecmm = _mm512_load_epi32(vec32);
        __m512i valmm = _mm512_set1_epi64(value);
        vecmm = _mm512_and_epi32(vecmm, _mm512_set1_epi32(~0xf));
        valmm = _mm512_srlv_epi64(valmm, _mm512_setr_epi64(0, 4, 8, 12, 16, 20, 24, 28));
        valmm = _mm512_and_epi32(valmm, _mm512_set1_epi32(0xf));
        vecmm = _mm512_or_epi32(vecmm, valmm);
        _mm512_store_epi32(vec32, vecmm);
    #else
        __m256i veclo = _mm256_load_si256((__m256i *)vec32);
        __m256i vechi = _mm256_load_si256((__m256i *)(vec32 + 8));
        __m256i vallo, valhi = _mm256_set1_epi64x(value);
        __m256i invmm = _mm256_set1_epi32(~0xf);
        veclo = _mm256_and_si256(veclo, invmm);
        vechi = _mm256_and_si256(vechi, invmm);
        vallo = _mm256_srlv_epi64(valhi, _mm256_setr_epi64x(0, 4, 8, 12));
        valhi = _mm256_srlv_epi64(valhi, _mm256_setr_epi64x(16, 20, 24, 28));
        __m256i mskmm = _mm256_set1_epi32(0xf);
        vallo = _mm256_and_si256(vallo, mskmm);
        valhi = _mm256_and_si256(valhi, mskmm);
        veclo = _mm256_or_si256(veclo, vallo);
        vechi = _mm256_or_si256(vechi, valhi);
        _mm256_store_si256((__m256i *)vec32, veclo);
        _mm256_store_si256((__m256i *)(vec32 + 8), vechi);
    #endif
    }

  static inline uint32_t imap__popcnt_hi28_simd__(uint32_t vec32[16],
									  uint32_t *p) noexcept {
#if defined (USE_AVX512F)
        __m512i vecmm = _mm512_load_epi32(vec32);
        vecmm = _mm512_and_epi32(vecmm, _mm512_set1_epi32(~0xf));
        __mmask16 mask = _mm512_cmp_epi32_mask(vecmm, _mm512_setzero_epi32(), _MM_CMPINT_NE);
        *p = vec32[imap__bsr__(mask)];
    #if defined(_MSC_VER)
        return __popcnt(mask);
    #elif defined(__GNUC__)
        return __builtin_popcount(mask);
    #endif
#else
        __m256i veclo = _mm256_load_si256((__m256i *)vec32);
        __m256i vechi = _mm256_load_si256((__m256i *)(vec32 + 8));
        __m256i invmm = _mm256_set1_epi32(~0xf);
        veclo = _mm256_and_si256(veclo, invmm);
        vechi = _mm256_and_si256(vechi, invmm);
        __m256i zermm = _mm256_setzero_si256();
        __m256i cmplo = _mm256_cmpeq_epi32(veclo, zermm);
        __m256i cmphi = _mm256_cmpeq_epi32(vechi, zermm);
        uint32_t msklo = _mm256_movemask_epi8(cmplo);
        uint32_t mskhi = _mm256_movemask_epi8(cmphi);
        uint64_t msk64 = (uint64_t)msklo | ((uint64_t)mskhi << 32);
        msk64 = ~msk64;
        msk64 &= 0x1111111111111111ull;
        *p = vec32[imap__bsr__(msk64) >> 2];
		return popcount::popcount<uint64_t>(msk64);
#endif
    }

    #define imap__extract_lo4__         imap__extract_lo4_simd__
    #define imap__deposit_lo4__         imap__deposit_lo4_simd__
    #define imap__popcnt_hi28__         imap__popcnt_hi28_simd__

#else
    constexpr static inline
    uint64_t imap__extract_lo4_port__(uint32_t vec32[16]) noexcept {
        union {
            uint32_t *vec32;
            uint64_t *vec64;
        } u;
        u.vec32 = vec32;
        return
            ((u.vec64[0] & 0xf0000000full)) |
            ((u.vec64[1] & 0xf0000000full) << 4) |
            ((u.vec64[2] & 0xf0000000full) << 8) |
            ((u.vec64[3] & 0xf0000000full) << 12) |
            ((u.vec64[4] & 0xf0000000full) << 16) |
            ((u.vec64[5] & 0xf0000000full) << 20) |
            ((u.vec64[6] & 0xf0000000full) << 24) |
            ((u.vec64[7] & 0xf0000000full) << 28);
    }

    constexpr static inline
    void imap__deposit_lo4_port__(uint32_t vec32[16], uint64_t value) noexcept {
        union
        {
            uint32_t *vec32;
            uint64_t *vec64;
        } u;
        u.vec32 = vec32;
        u.vec64[0] = (u.vec64[0] & ~0xf0000000full) | ((value) & 0xf0000000full);
        u.vec64[1] = (u.vec64[1] & ~0xf0000000full) | ((value >> 4) & 0xf0000000full);
        u.vec64[2] = (u.vec64[2] & ~0xf0000000full) | ((value >> 8) & 0xf0000000full);
        u.vec64[3] = (u.vec64[3] & ~0xf0000000full) | ((value >> 12) & 0xf0000000full);
        u.vec64[4] = (u.vec64[4] & ~0xf0000000full) | ((value >> 16) & 0xf0000000full);
        u.vec64[5] = (u.vec64[5] & ~0xf0000000full) | ((value >> 20) & 0xf0000000full);
        u.vec64[6] = (u.vec64[6] & ~0xf0000000full) | ((value >> 24) & 0xf0000000full);
        u.vec64[7] = (u.vec64[7] & ~0xf0000000full) | ((value >> 28) & 0xf0000000full);
    }

    constexpr static inline
    uint32_t imap__popcnt_hi28_port__(uint32_t vec32[16], uint32_t *p) noexcept {
        uint32_t pcnt = 0, sval, dirn;
        *p = 0;
        for (dirn = 0; 16 > dirn; dirn++) {
            sval = vec32[dirn];
            if (sval & ~0xf) {
                *p = sval;
                pcnt++;
            }
        }
        return pcnt;
    }

    #define imap__extract_lo4__         imap__extract_lo4_port__
    #define imap__deposit_lo4__         imap__deposit_lo4_port__
    #define imap__popcnt_hi28__         imap__popcnt_hi28_port__
#endif // USE_AVX2

    constexpr static uint32_t imap__tree_root__ = 0;
    constexpr static uint32_t imap__tree_resv__ = 1;
    constexpr static uint32_t imap__tree_mark__ = 2;
    constexpr static uint32_t imap__tree_size__ = 3;
    constexpr static uint32_t imap__tree_nfre__ = 4;
    constexpr static uint32_t imap__tree_vfre__ = 5;

    constexpr static uint32_t imap__prefix_pos__ = 0xf;
    constexpr static uint32_t imap__slot_pmask__ = 0x0000000f;
    constexpr static uint32_t imap__slot_node__  = 0x00000010;
    constexpr static uint32_t imap__slot_scalar__= 0x00000020;
    constexpr static uint32_t imap__slot_value__ = 0xffffffe0;
    constexpr static uint32_t imap__slot_shift__ = 6;
    constexpr static uint32_t imap__slot_sbits__ = (32 - imap__slot_shift__);

	#define imap__node_zero__           (imap_node_t{ { { 0 } } })
    #define imap__pair_zero__           (imap_pair_t{ 0,0 })
    #define imap__pair__(x, slot)       (imap_pair_t{ (x), (slot) })
    #define imap__slot_boxed__(sval)    (!((sval) & imap__slot_scalar__) && ((sval) >> imap__slot_shift__))

    constexpr inline uint32_t imap__alloc_node__(imap_node_t *tree) noexcept {
        uint32_t mark = tree->vec32[imap__tree_nfre__];
        if (mark) { 
            tree->vec32[imap__tree_nfre__] = *(uint32_t *)((uint8_t *)tree + mark);
		} else {
            mark = tree->vec32[imap__tree_mark__];
            ASSERT(mark + sizeof(imap_node_t) <= tree->vec32[imap__tree_size__]);
            tree->vec32[imap__tree_mark__] = mark + sizeof(imap_node_t);
        }
        return mark;
    }

    constexpr inline void imap__free_node__(imap_node_t *tree,
								  uint32_t mark) noexcept {
        *(uint32_t *)((uint8_t *)tree + mark) = tree->vec32[imap__tree_nfre__];
        tree->vec32[imap__tree_nfre__] = mark;
    }


    constexpr inline
    imap_node_t *imap__node__(imap_node_t *tree, 
			const uint32_t val) const noexcept {
        return (imap_node_t *)((uint8_t *)tree + val);
    }

    constexpr static inline
    uint64_t imap__node_prefix__(imap_node_t *node)
    {
        return imap__extract_lo4__(node->vec32);
    }

    static inline
    void imap__node_setprefix__(imap_node_t *node, uint64_t prefix)
    {
        imap__deposit_lo4__(node->vec32, prefix);
    }

    static inline
    uint32_t imap__node_pos__(imap_node_t *node)
    {
        return node->vec32[0] & 0xf;
    }

    static inline
    uint32_t imap__node_popcnt__(imap_node_t *node, uint32_t *p)
    {
        return imap__popcnt_hi28__(node->vec32, p);
    }

    constexpr static inline uint64_t imap__xpfx__(uint64_t x, uint32_t pos) {
        return x & (~0xfull << (pos << 2));
    }

    constexpr static inline uint32_t imap__xpos__(const uint64_t x) {
        return imap__bsr__(x) >> 2;
    }

    static inline
    uint32_t imap__xdir__(uint64_t x, uint32_t pos){
        return (x >> (pos << 2)) & 0xf;
    }

    constexpr inline uint32_t imap__alloc_val__(imap_node_t *tree) noexcept {
        uint32_t mark = imap__alloc_node__(tree);
        imap_node_t *node = imap__node__(tree, mark);
        mark <<= 3;
        tree->vec32[imap__tree_vfre__] = mark;
		// TODO avx stuff
        node->vec64[0] = mark + (1 << imap__slot_shift__);
        node->vec64[1] = mark + (2 << imap__slot_shift__);
        node->vec64[2] = mark + (3 << imap__slot_shift__);
        node->vec64[3] = mark + (4 << imap__slot_shift__);
        node->vec64[4] = mark + (5 << imap__slot_shift__);
        node->vec64[5] = mark + (6 << imap__slot_shift__);
        node->vec64[6] = mark + (7 << imap__slot_shift__);
        node->vec64[7] = 0;
        return mark;
    }

    constexpr inline uint32_t imap__alloc_val128__(imap_node_t *tree) noexcept {
        uint32_t mark = imap__alloc_node__(tree);
        imap_node_t *node = imap__node__(tree, mark);
        mark <<= 3;
        tree->vec32[imap__tree_vfre__] = mark;
        node->vec128[0].v[0] = mark + (2 << imap__slot_shift__);
        node->vec128[1].v[0] = mark + (4 << imap__slot_shift__);
        node->vec128[2].v[0] = mark + (6 << imap__slot_shift__);
        node->vec128[3].v[0] = 0;
        return mark;
    }

    inline imap_node_t *imap__ensure__(imap_node_t *tree,
	                                   const uint32_t n,
									   const uint32_t ysize) noexcept {
        imap_node_t *newtree;
        uint32_t hasnfre, hasvfre, newmark, oldsize, newsize;
        uint64_t newsize64;
        if (0 == n) {
            return tree;
		}

        if (nullptr == tree) {
            hasnfre = 0;
            hasvfre = 1;
            newmark = sizeof(imap_node_t);
            oldsize = 0;
        } else {
            hasnfre = !!tree->vec32[imap__tree_nfre__];
            hasvfre = !!tree->vec32[imap__tree_vfre__];
            newmark = tree->vec32[imap__tree_mark__];
            oldsize = tree->vec32[imap__tree_size__];
        }

        newmark += (n * 2 - hasnfre) * sizeof(imap_node_t) + (n - hasvfre) * ysize;
        if (newmark <= oldsize)
            return tree;
        newsize64 = imap__ceilpow2__(newmark);
        if (0x20000000 < newsize64)
            return 0;
        newsize = (uint32_t)newsize64;
        newtree = (imap_node_t *)IMAP_ALIGNED_ALLOC(sizeof(imap_node_t), newsize);
        if (!newtree) {
            return newtree;
		}

        if (nullptr == tree) {
            newtree->vec32[imap__tree_root__] = 0;
            newtree->vec32[imap__tree_resv__] = 0;
            newtree->vec32[imap__tree_mark__] = sizeof(imap_node_t);
            newtree->vec32[imap__tree_size__] = newsize;
            newtree->vec32[imap__tree_nfre__] = 0;
            if (sizeof(uint64_t) == ysize) {
                newtree->vec32[imap__tree_vfre__] = 3 << imap__slot_shift__;
                newtree->vec64[3] = 4 << imap__slot_shift__;
                newtree->vec64[4] = 5 << imap__slot_shift__;
                newtree->vec64[5] = 6 << imap__slot_shift__;
                newtree->vec64[6] = 7 << imap__slot_shift__;
                newtree->vec64[7] = 0;
            } else if (sizeof(__uint128_t) == ysize) {
                newtree->vec32[imap__tree_vfre__] = 4 << imap__slot_shift__;
                newtree->vec128[1].v[1] = 0;
                newtree->vec128[2].v[0] = 6 << imap__slot_shift__;
                newtree->vec128[2].v[1] = 0;
                newtree->vec128[3].v[0] = 0;
                newtree->vec128[3].v[1] = 0;
            } else {
                newtree->vec32[imap__tree_vfre__] = 0;
                newtree->vec64[3] = 0;
                newtree->vec64[4] = 0;
                newtree->vec64[5] = 0;
                newtree->vec64[6] = 0;
                newtree->vec64[7] = 0;
            }
        } else {
            memcpy(newtree, tree, tree->vec32[imap__tree_mark__]);
			// TODO yo what
            IMAP_ALIGNED_FREE(tree);
            newtree->vec32[imap__tree_size__] = newsize;
        }
        return newtree;
    }

   	inline imap_node_t *ensure0(const uint32_t n) noexcept {
		tree = imap__ensure__(tree, n, 0);
        return tree;
    }

   	inline imap_node_t *ensure64(const uint32_t n) noexcept {
        tree = imap__ensure__(tree, n, sizeof(uint64_t));
		return tree;
    }

   	inline imap_node_t *ensure128(const uint32_t n) noexcept {
        tree = imap__ensure__(tree, n, sizeof(__uint128_t));
		return tree;
    }


	// ensures that there is enough space for n elements
	constexpr inline void ensure(const uint32_t n) noexcept {
		tree = imap__ensure__(tree, n, sizeof(uint64_t));
	}

	using imap_slot_t = uint32_t;

	/// \param n max number of integers to store in the container
	constexpr imap_tree_t (const size_t n = 0) noexcept : tree(nullptr) {
		ensure(n+1);
	}

	~imap_tree_t() noexcept {
		IMAP_ALIGNED_FREE(tree);
	}

	///
	/// \param x
	/// \return
    [[nodiscard]] constexpr imap_slot_t *lookup(const uint64_t x) const noexcept {
        imap_node_t *node = tree;
        imap_slot_t *slot;
        uint32_t sval, posn = 16, dirn = 0;
        for (;;) {
            slot = &node->vec32[dirn];
            sval = *slot;
            if (!(sval & imap__slot_node__)) {
                if ((sval & imap__slot_value__) && 
					(imap__node_prefix__(node) == (x & ~0xfull))) {
                    ASSERT(0 == posn);
                    return slot;
                }

                return nullptr;
            }

            node = imap__node__(tree, sval & imap__slot_value__);
            posn = imap__node_pos__(node);
            dirn = imap__xdir__(x, posn);
        }
    }

	/// \param x
	/// \return
    [[nodiscard]] constexpr imap_slot_t *assign(const uint64_t x) noexcept {
        imap_slot_t *slotstack[16 + 1];
        uint32_t posnstack[16 + 1];
        uint32_t stackp, stacki;
        imap_node_t *newnode, *node = tree;
        imap_slot_t *slot;
        uint32_t newmark, sval, diff, posn = 16, dirn = 0;
        uint64_t prfx;
        stackp = 0;
        for (;;) {
            slot = &node->vec32[dirn];
            sval = *slot;
            slotstack[stackp] = slot, posnstack[stackp++] = posn;
            if (!(sval & imap__slot_node__)) {
                prfx = imap__node_prefix__(node);
                if (0 == posn && prfx == (x & ~0xfull)) {
					return slot;
				}

                diff = imap__xpos__(prfx ^ x);
                ASSERT(diff < 16);
                for (stacki = stackp; diff > posn;) {
                    posn = posnstack[--stacki];
				}

                if (stacki != stackp) {
                    slot = slotstack[stacki];
                    sval = *slot;
                    ASSERT(sval & imap__slot_node__);
                    newmark = imap__alloc_node__(tree);
                    *slot = (*slot & imap__slot_pmask__) | imap__slot_node__ | newmark;
                    newnode = imap__node__(tree, newmark);
                    *newnode = imap__node_zero__;
                    newmark = imap__alloc_node__(tree);
                    newnode->vec32[imap__xdir__(prfx, diff)] = sval;
                    newnode->vec32[imap__xdir__(x, diff)] = imap__slot_node__ | newmark;
                    imap__node_setprefix__(newnode, imap__xpfx__(prfx, diff) | diff);
                } else {
                    newmark = imap__alloc_node__(tree);
                    *slot = (*slot & imap__slot_pmask__) | imap__slot_node__ | newmark;
                }

                newnode = imap__node__(tree, newmark);
                *newnode = imap__node_zero__;
                imap__node_setprefix__(newnode, x & ~0xfull);
                return &newnode->vec32[x & 0xfull];
            }

            node = imap__node__(tree, sval & imap__slot_value__);
            posn = imap__node_pos__(node);
            dirn = imap__xdir__(x, posn);
        }
    }

	/// \param slot
	/// \return
    [[nodiscard]] constexpr inline bool hasval(const imap_slot_t *slot) noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        return sval & imap__slot_value__;
    }

	/// \param slot
	/// \return
    [[nodiscard]] constexpr inline uint64_t getval(const imap_slot_t *slot) const noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        if (!imap__slot_boxed__(sval)) {
			return sval >> imap__slot_shift__;
		} else {
			return tree->vec64[sval >> imap__slot_shift__];
		}
    }

	///
	/// \param tree
	/// \param slot
	/// \return
    [[nodiscard]] constexpr uint32_t getval0(const imap_slot_t *slot) const noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        return sval >> imap__slot_shift__;
    }

	/// \param slot
	/// \return
    [[nodiscard]] constexpr uint64_t getval64(const imap_slot_t *slot) const noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        return tree->vec64[sval >> imap__slot_shift__];
    }

	/// \param slot
	/// \return
    [[nodiscard]] constexpr imap_u128 getval128(const imap_slot_t *slot) const noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        return tree->vec128[sval >> (imap__slot_shift__ + 1)];
    }

	/// \param slot
	/// \param y
	/// \return
    constexpr void setval(imap_slot_t *slot,
	                      const uint64_t y) noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        if (y < (1 << (imap__slot_sbits__))) {
            if (imap__slot_boxed__(sval)) {
                tree->vec64[sval >> imap__slot_shift__] = tree->vec32[imap__tree_vfre__];
                tree->vec32[imap__tree_vfre__] = sval & imap__slot_value__;
            }

            *slot = (*slot & imap__slot_pmask__) | imap__slot_scalar__ | (uint32_t)(y << imap__slot_shift__);
        } else {
            if (!imap__slot_boxed__(sval)) {
                sval = tree->vec32[imap__tree_vfre__];
                if (!sval) {
                    sval = imap__alloc_val__(tree);
				}
                ASSERT(sval >> imap__slot_shift__);
                tree->vec32[imap__tree_vfre__] = (uint32_t)tree->vec64[sval >> imap__slot_shift__];
            }

            ASSERT(!(sval & imap__slot_node__));
            ASSERT(imap__slot_boxed__(sval));
            *slot = (*slot & imap__slot_pmask__) | sval;
            tree->vec64[sval >> imap__slot_shift__] = y;
        }
    }

	///
	/// \param slot
	/// \param y
    constexpr void setval0(imap_slot_t *slot,
	                      const uint32_t y) noexcept{
        ASSERT(!(*slot & imap__slot_node__));
        *slot = (*slot & imap__slot_pmask__) | imap__slot_scalar__ | (uint32_t)(y << imap__slot_shift__);
    }

    constexpr void setval64(imap_slot_t *slot,
	                        const uint64_t y) noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        if (!(sval >> imap__slot_shift__)) {
            sval = tree->vec32[imap__tree_vfre__];
            if (!sval) {
				sval = imap__alloc_val__(tree);
			}
            ASSERT(sval >> imap__slot_shift__);
            tree->vec32[imap__tree_vfre__] = (uint32_t)tree->vec64[sval >> imap__slot_shift__];
        }
        ASSERT(!(sval & imap__slot_node__));
        ASSERT(imap__slot_boxed__(sval));
        *slot = (*slot & imap__slot_pmask__) | sval;
        tree->vec64[sval >> imap__slot_shift__] = y;
    }

    constexpr void setval128(imap_slot_t *slot,
					    	 const imap_u128 y) noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        if (!(sval >> imap__slot_shift__)) {
            sval = tree->vec32[imap__tree_vfre__];
            if (!sval)
                sval = imap__alloc_val128__(tree);
            ASSERT(sval >> imap__slot_shift__);
            tree->vec32[imap__tree_vfre__] = (uint32_t)tree->vec128[sval >> (imap__slot_shift__ + 1)].v[0];
        }
        ASSERT(!(sval & imap__slot_node__));
        ASSERT(imap__slot_boxed__(sval));
        *slot = (*slot & imap__slot_pmask__) | sval;
        tree->vec128[sval >> (imap__slot_shift__ + 1)] = y;
    }

    constexpr inline void delval(imap_slot_t *slot) noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        if (imap__slot_boxed__(sval)) {
            tree->vec64[sval >> imap__slot_shift__] = tree->vec32[imap__tree_vfre__];
            tree->vec32[imap__tree_vfre__] = sval & imap__slot_value__;
        }
        *slot &= imap__slot_pmask__;
    }

    [[nodiscard]] constexpr inline uint64_t *addrof64(imap_slot_t *slot) noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        return &tree->vec64[sval >> imap__slot_shift__];
    }

    [[nodiscard]] constexpr inline imap_u128 *addrof128(imap_slot_t *slot) noexcept {
        ASSERT(!(*slot & imap__slot_node__));
        uint32_t sval = *slot;
        return &tree->vec128[sval >> (imap__slot_shift__ + 1)];
    }

    constexpr void remove(uint64_t x) noexcept {
        imap_slot_t *slotstack[16 + 1];
        uint32_t stackp;
        imap_node_t *node = tree;
        imap_slot_t *slot;
        uint32_t sval, pval, posn = 16, dirn = 0;
        stackp = 0;
        for (;;) {
            slot = &node->vec32[dirn];
            sval = *slot;
            if (!(sval & imap__slot_node__)) {
                if ((sval & imap__slot_value__) &&
				    (imap__node_prefix__(node) == (x & ~0xfull))) {
                    ASSERT(0 == posn);
                    delval(slot);
                }
                while (stackp)
                {
                    slot = slotstack[--stackp];
                    sval = *slot;
                    node = imap__node__(tree, sval & imap__slot_value__);
                    posn = imap__node_pos__(node);
                    if (!!posn != imap__node_popcnt__(node, &pval))
                        break;
                    imap__free_node__(tree, sval & imap__slot_value__);
                    *slot = (sval & imap__slot_pmask__) | (pval & ~imap__slot_pmask__);
                }
                return;
            }
            node = imap__node__(tree, sval & imap__slot_value__);
            posn = imap__node_pos__(node);
            dirn = imap__xdir__(x, posn);
            slotstack[stackp++] = slot;
        }
    }

    constexpr imap_pair_t locate(imap_iter_t *iter,
	                             uint64_t x) noexcept {
        imap_node_t *node = tree;
        imap_slot_t *slot;
        uint32_t sval, posn = 16, dirn = 0;
        uint64_t prfx, xpfx;
        iter->stackp = 0;
        for (;;) {
            slot = &node->vec32[dirn];
            sval = *slot;
            if (!(sval & imap__slot_node__)) {
                prfx = imap__node_prefix__(node);
                if ((sval & imap__slot_value__) && prfx == (x & ~0xfull)) {
                    ASSERT(0 == posn);
                    return imap__pair__(prfx | dirn, slot);
                }
                if (iter->stackp)
                    for (;;) {
                        prfx = imap__xpfx__(prfx, posn);
                        xpfx = imap__xpfx__(x, posn);
                        if (prfx == xpfx) {
                            break;
						}

                        if (prfx > xpfx) {
                            if (!--iter->stackp) {
                                // start at beginning of tree; same as supplying restart=1
                                iter->stack[iter->stackp++] &= imap__slot_value__;
                                break;
                            }
                            iter->stack[iter->stackp - 1]--;
                        } else { // if (prfx < xpfx)
                            if (!--iter->stackp)
                                break;
                        }
                        sval = iter->stack[iter->stackp - 1];
                        node = imap__node__(tree, sval & imap__slot_value__);
                        posn = imap__node_pos__(node);
                    }
                return iterate(iter, 0);
            }
            node = imap__node__(tree, sval & imap__slot_value__);
            posn = imap__node_pos__(node);
            dirn = imap__xdir__(x, posn);
            iter->stack[iter->stackp++] = (sval & imap__slot_value__) | (dirn + 1);
        }
    }

	///
	/// \param tree
	/// \param iter
	/// \param restart
	/// \return
    imap_pair_t iterate(imap_iter_t *iter,
						int restart) noexcept {
        imap_node_t *node;
        imap_slot_t *slot;
        uint32_t sval, dirn;
        if (restart) {
            iter->stackp = 0;
            sval = dirn = 0;
            goto enter;
        }
        // loop while stack is not empty
        while (iter->stackp) {
            // get slot value and increment direction
            sval = iter->stack[iter->stackp - 1]++;
            dirn = sval & 31;
            if (15 < dirn) {
                // if directions 0-15 have been examined, pop node from stack
                iter->stackp--;
                continue;
            }
        enter:
            node = imap__node__(tree, sval & imap__slot_value__);
            slot = &node->vec32[dirn];
            sval = *slot;
            if (sval & imap__slot_node__)
                // push node into stack
                iter->stack[iter->stackp++] = sval & imap__slot_value__;
            else if (sval & imap__slot_value__)
                return imap__pair__(imap__node_prefix__(node) | dirn, slot);
        }

        return imap__pair_zero__;
    }
};

#undef imap__extract_lo4__
#undef imap__deposit_lo4__
#undef imap__popcnt_hi28_
#endif
