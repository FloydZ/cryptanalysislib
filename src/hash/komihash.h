// source: https://github.com/avaneev/komihash 5.1.2

#define KOMIHASH_LIKELY( x ) __builtin_expect( x, 1 )
#define KOMIHASH_UNLIKELY( x ) __builtin_expect( x, 0 )
#define KOMIHASH_PREFETCH( a ) __builtin_prefetch( a, 0, 2 )

/**
 * @def KOMIHASH_PREFETCH_2
 * @brief Compiler-dependent address prefetch macro, ordered position 2.
 * @param a Prefetch address.
 */

#if defined( __clang__ )
	#define KOMIHASH_PREFETCH_1( a ) KOMIHASH_PREFETCH( a )
	#define KOMIHASH_PREFETCH_2( a )
#else // defined( __clang__ )
	#define KOMIHASH_PREFETCH_1( a )
	#define KOMIHASH_PREFETCH_2( a ) KOMIHASH_PREFETCH( a )
#endif // defined( __clang__ )

/**
 * @def KOMIHASH_NOEX
 * @brief Macro that defines the "noexcept" function specifier for C++
 * environment.
 */

#if defined( __cplusplus ) && __cplusplus >= 201103L

	#include <cstdint>
	#include <cstring>

	#define KOMIHASH_U64_C( x ) UINT64_C( x )
	#define KOMIHASH_NOEX noexcept

#else // __cplusplus

	#include <stdint.h>
	#include <string.h>

	#define KOMIHASH_U64_C( x ) ( x )
	#define KOMIHASH_NOEX

#endif // __cplusplus

/**
 * @{
 * @brief Unsigned 64-bit constant that defines the initial state of the
 * hash function (first mantissa bits of PI).
 */
#define KOMIHASH_IVAL1 KOMIHASH_U64_C( 0x243F6A8885A308D3 )
#define KOMIHASH_IVAL2 KOMIHASH_U64_C( 0x13198A2E03707344 )
#define KOMIHASH_IVAL3 KOMIHASH_U64_C( 0xA4093822299F31D0 )
#define KOMIHASH_IVAL4 KOMIHASH_U64_C( 0x082EFA98EC4E6C89 )
#define KOMIHASH_IVAL5 KOMIHASH_U64_C( 0x452821E638D01377 )
#define KOMIHASH_IVAL6 KOMIHASH_U64_C( 0xBE5466CF34E90C6C )
#define KOMIHASH_IVAL7 KOMIHASH_U64_C( 0xC0AC29B7C97C50DD )
#define KOMIHASH_IVAL8 KOMIHASH_U64_C( 0x3F84D5B5B5470917 )

/** @} */

/**
 * @def KOMIHASH_VAL01
 * @brief Unsigned 64-bit constant with `01` bit-pair replication.
 */

#define KOMIHASH_VAL01 KOMIHASH_U64_C( 0x5555555555555555 )

/**
 * @def KOMIHASH_VAL10
 * @brief Unsigned 64-bit constant with `10` bit-pair replication.
 */

#define KOMIHASH_VAL10 KOMIHASH_U64_C( 0xAAAAAAAAAAAAAAAA )



#define KOMIHASH_EC32( v ) __builtin_bswap32( v )
#define KOMIHASH_EC64( v ) __builtin_bswap64( v )

/**
 * @brief Load unsigned 32-bit value with endianness-correction.
 *
 * An auxiliary function that returns an unsigned 32-bit value created out of
 * a sequence of bytes in memory. This function is used to convert endianness
 * of in-memory 32-bit unsigned values, and to avoid unaligned memory
 * accesses.
 *
 * @param p Pointer to 4 bytes in memory. Alignment is unimportant.
 * @return Endianness-corrected 32-bit value from memory.
 */

static inline 
uint32_t kh_lu32ec(const uint8_t* const p ) noexcept {
	uint32_t v = *(const uint32_t *)p;
	return( KOMIHASH_EC32( v ));
}

/**
 * @brief Load unsigned 64-bit value with endianness-correction.
 *
 * An auxiliary function that returns an unsigned 64-bit value created out of
 * a sequence of bytes in memory. This function is used to convert endianness
 * of in-memory 64-bit unsigned values, and to avoid unaligned memory
 * accesses.
 *
 * @param p Pointer to 8 bytes in memory. Alignment is unimportant.
 * @return Endianness-corrected 64-bit value from memory.
 */
static inline 
uint64_t kh_lu64ec(const uint8_t* const p) noexcept {
	uint64_t v = *(uint64_t *)p;
	//memcpy(&v, p, 8 );
	return( KOMIHASH_EC64( v ));
}

/**
 * @brief Load unsigned 64-bit value with padding (Msg-3 reads).
 *
 * Function builds an unsigned 64-bit value out of remaining bytes in a
 * message, and pads it with the "final byte". This function can only be
 * called if less than 8 bytes are left to read. The message should be "long",
 * permitting `Msg[ -3 ]` reads.
 *
 * @param Msg Message pointer, alignment is unimportant.
 * @param MsgLen Message's remaining length, in bytes; can be 0.
 * @return Final byte-padded value from the message.
 */
static inline
uint64_t kh_lpu64ec_l3(const uint8_t* const Msg,
	                   const size_t MsgLen ) noexcept {
	const int ml8 = (int) ( MsgLen * 8 );

	if( MsgLen < 4 )
	{
		const uint8_t* const Msg3 = Msg + MsgLen - 3;
		const uint64_t m = (uint64_t) Msg3[ 0 ] | (uint64_t) Msg3[ 1 ] << 8 |
			(uint64_t) Msg3[ 2 ] << 16;

		return( (uint64_t) 1 << ml8 | m >> ( 24 - ml8 ));
	}

	const uint64_t mh = kh_lu32ec( Msg + MsgLen - 4 );
	const uint64_t ml = kh_lu32ec( Msg );

	return( (uint64_t) 1 << ml8 | ml | ( mh >> ( 64 - ml8 )) << 32 );
}

/**
 * @brief Load unsigned 64-bit value with padding (non-zero message length).
 *
 * Function builds an unsigned 64-bit value out of remaining bytes in a
 * message, and pads it with the "final byte". This function can only be
 * called if less than 8 bytes are left to read. Can be used on "short"
 * messages, but `MsgLen` should be greater than 0.
 *
 * @param Msg Message pointer, alignment is unimportant.
 * @param MsgLen Message's remaining length, in bytes; cannot be 0.
 * @return Final byte-padded value from the message.
 */
static inline uint64_t kh_lpu64ec_nz(const uint8_t* const Msg,
	                                 const size_t MsgLen ) noexcept {
	const int ml8 = (int) ( MsgLen * 8 );

	if( MsgLen < 4 )
	{
		uint64_t m = Msg[ 0 ];

		if( MsgLen > 1 )
		{
			m |= (uint64_t) Msg[ 1 ] << 8;

			if( MsgLen > 2 )
			{
				m |= (uint64_t) Msg[ 2 ] << 16;
			}
		}

		return( (uint64_t) 1 << ml8 | m );
	}

	const uint64_t mh = kh_lu32ec( Msg + MsgLen - 4 );
	const uint64_t ml = kh_lu32ec( Msg );

	return( (uint64_t) 1 << ml8 | ml | ( mh >> ( 64 - ml8 )) << 32 );
}

/**
 * @brief Load unsigned 64-bit value with padding (Msg-4 reads).
 *
 * Function builds an unsigned 64-bit value out of remaining bytes in a
 * message, and pads it with the "final byte". This function can only be
 * called if less than 8 bytes are left to read. The message should be "long",
 * permitting `Msg[ -4 ]` reads.
 *
 * @param Msg Message pointer, alignment is unimportant.
 * @param MsgLen Message's remaining length, in bytes; can be 0.
 * @return Final byte-padded value from the message.
 */
static inline uint64_t kh_lpu64ec_l4(const uint8_t* const Msg,
	                                 const size_t MsgLen ) noexcept 
{
	const int ml8 = (int) ( MsgLen * 8 );

	if( MsgLen < 5 )
	{
		const uint64_t m = kh_lu32ec( Msg + MsgLen - 4 );

		return( (uint64_t) 1 << ml8 | m >> ( 32 - ml8 ));
	}

	const uint64_t m = kh_lu64ec( Msg + MsgLen - 8 );

	return( (uint64_t) 1 << ml8 | m >> ( 64 - ml8 ));
}

/**
 * @fn void kh_m128( uint64_t u, uint64_t v, uint64_t* rl, uint64_t* rha )
 * @brief 64-bit by 64-bit unsigned multiplication with result accumulation.
 *
 * @param u Multiplier 1.
 * @param v Multiplier 2.
 * @param[out] rl The lower half of the 128-bit result.
 * @param[in,out] rha The accumulator to receive the higher half of the
 * 128-bit result.
 */

/**
 * @def KOMIHASH_EMULU( u, v )
 * @brief Macro for 32-bit by 32-bit unsigned multiplication with 64-bit
 * result.
 * @param u Multiplier 1.
 * @param v Multiplier 2.
 */

#if defined( _MSC_VER ) && ( defined( _M_ARM64 ) || defined( _M_ARM64EC ) || \
	( defined( _M_X64 ) && defined( __INTEL_COMPILER )))

	#include <intrin.h>

	KOMIHASH_INLINE_F void kh_m128( const uint64_t u, const uint64_t v,
		uint64_t* const rl, uint64_t* const rha ) KOMIHASH_NOEX
	{
		*rl = u * v;
		*rha += __umulh( u, v );
	}

#elif defined( _MSC_VER ) && ( defined( _M_X64 ) || defined( _M_IA64 ))

	#include <intrin.h>
	#pragma intrinsic(_umul128)

	KOMIHASH_INLINE_F void kh_m128( const uint64_t u, const uint64_t v,
		uint64_t* const rl, uint64_t* const rha ) KOMIHASH_NOEX
	{
		uint64_t rh;
		*rl = _umul128( u, v, &rh );
		*rha += rh;
	}

#elif defined( __SIZEOF_INT128__ )
    static inline
	void kh_m128( const uint64_t u, const uint64_t v,
		uint64_t* const rl, uint64_t* const rha ) noexcept
	{
		const __uint128_t r = (__uint128_t) u * v;

		*rha += (uint64_t) ( r >> 64 );
		*rl = (uint64_t) r;
	}

#elif ( defined( __IBMC__ ) || defined( __IBMCPP__ )) && defined( __LP64__ )

	KOMIHASH_INLINE_F void kh_m128( const uint64_t u, const uint64_t v,
		uint64_t* const rl, uint64_t* const rha ) KOMIHASH_NOEX
	{
		*rl = u * v;
		*rha += __mulhdu( u, v );
	}

#else // defined( __IBMC__ )

	// _umul128() code for 32-bit systems, adapted from Hacker's Delight,
	// Henry S. Warren, Jr.

	#if defined( _MSC_VER ) && !defined( __INTEL_COMPILER ) && \
		!defined( _M_ARM )

		#include <intrin.h>
		#pragma intrinsic(__emulu)

		#define KOMIHASH_EMULU( u, v ) __emulu( u, v )

	#else // defined( _MSC_VER ) && !defined( __INTEL_COMPILER )

		#define KOMIHASH_EMULU( u, v ) ( (uint64_t) ( u ) * ( v ))

	#endif // defined( _MSC_VER ) && !defined( __INTEL_COMPILER )

	KOMIHASH_INLINE void kh_m128( const uint64_t u, const uint64_t v,
		uint64_t* const rl, uint64_t* const rha ) KOMIHASH_NOEX
	{
		*rl = u * v;

		const uint32_t u0 = (uint32_t) u;
		const uint32_t v0 = (uint32_t) v;
		const uint64_t w0 = KOMIHASH_EMULU( u0, v0 );
		const uint32_t u1 = (uint32_t) ( u >> 32 );
		const uint32_t v1 = (uint32_t) ( v >> 32 );
		const uint64_t t = KOMIHASH_EMULU( u1, v0 ) + (uint32_t) ( w0 >> 32 );
		const uint64_t w1 = KOMIHASH_EMULU( u0, v1 ) + (uint32_t) t;

		*rha += KOMIHASH_EMULU( u1, v1 ) + (uint32_t) ( w1 >> 32 ) +
			(uint32_t) ( t >> 32 );
	}

#endif // defined( __IBMC__ )

/**
 * @def KOMIHASH_HASHROUND()
 * @brief Macro for a common hashing round without input.
 *
 * The three instructions in this macro (multiply, add, and XOR) represent the
 * simplest constantless PRNG, scalable to any even-sized state variables,
 * with the `Seed1` being the PRNG output (2^64 PRNG period). It passes
 * `PractRand` tests with rare non-systematic "unusual" evaluations.
 *
 * To make this PRNG reliable, self-starting, and eliminate a risk of
 * stopping, the following variant can be used, which adds a "register
 * checker-board", a source of raw entropy. The PRNG is available as the
 * komirand() function. Not required for hashing (but works for it) since the
 * input entropy is usually available in abundance during hashing.
 *
 * `Seed5 += 0xAAAAAAAAAAAAAAAA;`
 *
 * (the `0xAAAA...` constant should match register's size; essentially, it is
 * a replication of the `10` bit-pair; it is not an arbitrary constant).
 */

#define KOMIHASH_HASHROUND() \
	kh_m128( Seed1, Seed5, &Seed1, &Seed5 ); \
	Seed1 ^= Seed5;

/**
 * @def KOMIHASH_HASH16( m )
 * @brief Macro for a common hashing round with 16-byte input.
 * @param m Message pointer, alignment is unimportant.
 */

#define KOMIHASH_HASH16( m ) \
	kh_m128( Seed1 ^ kh_lu64ec( m ), \
		Seed5 ^ kh_lu64ec( m + 8 ), &Seed1, &Seed5 ); \
	Seed1 ^= Seed5;

/**
 * @def KOMIHASH_HASHFIN()
 * @brief Macro for common hashing finalization round.
 *
 * The final hashing input is expected in the `r1h` and `r2h` temporary
 * variables. The macro inserts the function return instruction.
 */

#define KOMIHASH_HASHFIN() \
	kh_m128( r1h, r2h, &Seed1, &Seed5 ); \
	Seed1 ^= Seed5; \
	KOMIHASH_HASHROUND(); \
	return( Seed1 );

/**
 * @def KOMIHASH_HASHLOOP64()
 * @brief Macro for a common 64-byte full-performance hashing loop.
 *
 * Expects `Msg` and `MsgLen` values (greater than 63), requires initialized
 * `Seed1-8` values.
 *
 * The "shifting" arrangement of `Seed1-4` XORs (below) does not increase
 * individual `SeedN` PRNG period beyond 2^64, but reduces a chance of any
 * occassional synchronization between PRNG lanes happening. Practically,
 * `Seed1-4` together become a single "fused" 256-bit PRNG value, having 2^66
 * summary PRNG period.
 */

#define KOMIHASH_HASHLOOP64() \
	do \
	{ \
		KOMIHASH_PREFETCH_1( Msg ); \
	\
		kh_m128( Seed1 ^ kh_lu64ec( Msg ), \
			Seed5 ^ kh_lu64ec( Msg + 32 ), &Seed1, &Seed5 ); \
	\
		kh_m128( Seed2 ^ kh_lu64ec( Msg + 8 ), \
			Seed6 ^ kh_lu64ec( Msg + 40 ), &Seed2, &Seed6 ); \
	\
		kh_m128( Seed3 ^ kh_lu64ec( Msg + 16 ), \
			Seed7 ^ kh_lu64ec( Msg + 48 ), &Seed3, &Seed7 ); \
	\
		kh_m128( Seed4 ^ kh_lu64ec( Msg + 24 ), \
			Seed8 ^ kh_lu64ec( Msg + 56 ), &Seed4, &Seed8 ); \
	\
		Msg += 64; \
		MsgLen -= 64; \
	\
		KOMIHASH_PREFETCH_2( Msg ); \
	\
		Seed2 ^= Seed5; \
		Seed3 ^= Seed6; \
		Seed4 ^= Seed7; \
		Seed1 ^= Seed8; \
	\
	} while( KOMIHASH_LIKELY( MsgLen > 63 ));

/**
 * @brief The hashing epilogue function (for internal use).
 *
 * @param Msg Pointer to the remaining part of the message.
 * @param MsgLen Remaining part's length, can be 0.
 * @param Seed1 Latest Seed1 value.
 * @param Seed5 Latest Seed5 value.
 * @return 64-bit hash value.
 */
static inline
uint64_t komihash_epi(const uint8_t* Msg,
                      size_t MsgLen, 
                      uint64_t Seed1, 
                      uint64_t Seed5 ) noexcept {
	uint64_t r1h, r2h;

	if( KOMIHASH_LIKELY( MsgLen > 31 )) {
		KOMIHASH_HASH16( Msg );
		KOMIHASH_HASH16( Msg + 16 );

		Msg += 32;
		MsgLen -= 32;
	}

	if( MsgLen > 15 )
	{
		KOMIHASH_HASH16( Msg );

		Msg += 16;
		MsgLen -= 16;
	}

	if( MsgLen > 7 )
	{
		r2h = Seed5 ^ kh_lpu64ec_l4( Msg + 8, MsgLen - 8 );
		r1h = Seed1 ^ kh_lu64ec( Msg );
	}
	else
	{
		r1h = Seed1 ^ kh_lpu64ec_l4( Msg, MsgLen );
		r2h = Seed5;
	}

	KOMIHASH_HASHFIN();
}

/**
 * @brief KOMIHASH 64-bit hash function.
 *
 * Produces and returns a 64-bit hash value of the specified message, string,
 * or binary data block. Designed for 64-bit hash-table and hash-map uses.
 * Produces identical hashes on both big- and little-endian systems.
 *
 * @param Msg0 The message to produce a hash from. The alignment of this
 * pointer is unimportant. It is valid to pass 0 when `MsgLen` equals 0
 * (assuming that compiler's implementation of the address prefetch permits
 * the use of zero address).
 * @param MsgLen Message's length, in bytes, can be zero.
 * @param UseSeed Optional value, to use instead of the default seed. To use
 * the default seed, set to 0. The UseSeed value can have any bit length and
 * statistical quality, and is used only as an additional entropy source. May
 * need endianness-correction via KOMIHASH_EC64(), if this value is shared
 * between big- and little-endian systems.
 * @return 64-bit hash of the input data. Should be endianness-corrected when
 * this value is shared between big- and little-endian systems.
 */
static inline 
uint64_t komihash(const void* const Msg0, size_t MsgLen,
                  const uint64_t UseSeed ) noexcept
{
	const uint8_t* Msg = (const uint8_t*) Msg0;

	uint64_t Seed1 = KOMIHASH_IVAL1 ^ ( UseSeed & KOMIHASH_VAL01 );
	uint64_t Seed5 = KOMIHASH_IVAL5 ^ ( UseSeed & KOMIHASH_VAL10 );
	uint64_t r1h, r2h;

	KOMIHASH_PREFETCH( Msg );

	KOMIHASH_HASHROUND(); // Required for Perlin Noise.

	if( KOMIHASH_LIKELY( MsgLen < 16 ))
	{
		r1h = Seed1;
		r2h = Seed5;

		if( MsgLen > 7 )
		{
			// The following two XOR instructions are equivalent to mixing a
			// message with a cryptographic one-time-pad (bitwise modulo 2
			// addition). Message's statistics and distribution are thus
			// unimportant.

			r2h ^= kh_lpu64ec_l3( Msg + 8, MsgLen - 8 );
			r1h ^= kh_lu64ec( Msg );
		}
		else
		if( KOMIHASH_LIKELY( MsgLen != 0 ))
		{
			r1h ^= kh_lpu64ec_nz( Msg, MsgLen );
		}

		KOMIHASH_HASHFIN();
	}

	if( KOMIHASH_LIKELY( MsgLen < 32 ))
	{
		KOMIHASH_HASH16( Msg );

		if( MsgLen > 23 )
		{
			r2h = Seed5 ^ kh_lpu64ec_l4( Msg + 24, MsgLen - 24 );
			r1h = Seed1 ^ kh_lu64ec( Msg + 16 );
			KOMIHASH_HASHFIN();
		}
		else
		{
			r1h = Seed1 ^ kh_lpu64ec_l4( Msg + 16, MsgLen - 16 );
			r2h = Seed5;
			KOMIHASH_HASHFIN();
		}
	}

	if( KOMIHASH_LIKELY( MsgLen > 63 ))
	{
		uint64_t Seed2 = KOMIHASH_IVAL2 ^ Seed1;
		uint64_t Seed3 = KOMIHASH_IVAL3 ^ Seed1;
		uint64_t Seed4 = KOMIHASH_IVAL4 ^ Seed1;
		uint64_t Seed6 = KOMIHASH_IVAL6 ^ Seed5;
		uint64_t Seed7 = KOMIHASH_IVAL7 ^ Seed5;
		uint64_t Seed8 = KOMIHASH_IVAL8 ^ Seed5;

		KOMIHASH_HASHLOOP64();

		Seed5 ^= Seed6 ^ Seed7 ^ Seed8;
		Seed1 ^= Seed2 ^ Seed3 ^ Seed4;
	}

	return( komihash_epi( Msg, MsgLen, Seed1, Seed5 ));
}
