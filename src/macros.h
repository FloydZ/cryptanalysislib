// TODO merge with helper.h

#ifndef SMALLSECRETLWE_MACROS_H
#define SMALLSECRETLWE_MACROS_H


/* ecrypt-config.h */

/* *** Normally, it should not be necessary to edit this file. *** */

#ifndef ECRYPT_CONFIG
#define ECRYPT_CONFIG

// Check windows
#if _WIN32 || _WIN64
#if _WIN64
#define ENVIRONMENT64
#else
#define ENVIRONMENT32
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define ENVIRONMENT64
#else
#define ENVIRONMENT32
#endif
#endif


/* ------------------------------------------------------------------------- */

/* Guess the endianness of the target architecture. */

/*
 * The LITTLE endian machines:
 */
#if defined(__ultrix)           /* Older MIPS */
#define ECRYPT_LITTLE_ENDIAN
#elif defined(__alpha)          /* Alpha */
#define ECRYPT_LITTLE_ENDIAN
#elif defined(i386)             /* x86 (gcc) */
#define ECRYPT_LITTLE_ENDIAN
#elif defined(__i386)           /* x86 (gcc) */
#define ECRYPT_LITTLE_ENDIAN
#elif defined(_M_IX86)          /* x86 (MSC, Borland) */
#define ECRYPT_LITTLE_ENDIAN
#elif defined(_MSC_VER)         /* x86 (surely MSC) */
#define ECRYPT_LITTLE_ENDIAN
#elif defined(__INTEL_COMPILER) /* x86 (surely Intel compiler icl.exe) */
#define ECRYPT_LITTLE_ENDIAN

/*
 * The BIG endian machines:
 */
#elif defined(sun)              /* Newer Sparc's */
#define ECRYPT_BIG_ENDIAN
#elif defined(__ppc__)          /* PowerPC */
#define ECRYPT_BIG_ENDIAN

/*
 * Finally machines with UNKNOWN endianness:
 */
#elif defined (_AIX)            /* RS6000 */
#define ECRYPT_UNKNOWN
#elif defined(__hpux)           /* HP-PA */
#define ECRYPT_UNKNOWN
#elif defined(__aux)            /* 68K */
#define ECRYPT_UNKNOWN
#elif defined(__dgux)           /* 88K (but P6 in latest boxes) */
#define ECRYPT_UNKNOWN
#elif defined(__sgi)            /* Newer MIPS */
#define ECRYPT_UNKNOWN
#else	                        /* Any other processor */
#define ECRYPT_UNKNOWN
#endif

/* ------------------------------------------------------------------------- */

/*
 * Find minimal-width types to store 8-bit, 16-bit, 32-bit, and 64-bit
 * integers.
 *
 * Note: to enable 64-bit types on 32-bit compilers, it might be
 * necessary to switch from ISO C90 mode to ISO C99 mode (e.g., gcc
 * -std=c99).
 */

#include <limits.h>

/* --- check char --- */

#if (UCHAR_MAX / 0xFU > 0xFU)
#ifndef I8T
#define I8T char
#define U8C(v) (v##U)

#if (UCHAR_MAX == 0xFFU)
#define ECRYPT_I8T_IS_BYTE
#endif

#endif

#if (UCHAR_MAX / 0xFFU > 0xFFU)
#ifndef I16T
#define I16T char
#define U16C(v) (v##U)
#endif

#if (UCHAR_MAX / 0xFFFFU > 0xFFFFU)
#ifndef I32T
#define I32T char
#define U32C(v) (v##U)
#endif

#if (UCHAR_MAX / 0xFFFFFFFFU > 0xFFFFFFFFU)
#ifndef I64T
#define I64T char
#define U64C(v) (v##U)
#define ECRYPT_NATIVE64
#endif

#endif
#endif
#endif
#endif

/* --- check short --- */

#if (USHRT_MAX / 0xFU > 0xFU)
#ifndef I8T
#define I8T short
#define U8C(v) (v##U)

#if (USHRT_MAX == 0xFFU)
#define ECRYPT_I8T_IS_BYTE
#endif

#endif

#if (USHRT_MAX / 0xFFU > 0xFFU)
#ifndef I16T
#define I16T short
#define U16C(v) (v##U)
#endif

#if (USHRT_MAX / 0xFFFFU > 0xFFFFU)
#ifndef I32T
#define I32T short
#define U32C(v) (v##U)
#endif

#if (USHRT_MAX / 0xFFFFFFFFU > 0xFFFFFFFFU)
#ifndef I64T
#define I64T short
#define U64C(v) (v##U)
#define ECRYPT_NATIVE64
#endif

#endif
#endif
#endif
#endif

/* --- check int --- */

#if (UINT_MAX / 0xFU > 0xFU)
#ifndef I8T
#define I8T int
#define U8C(v) (v##U)

#if (ULONG_MAX == 0xFFU)
#define ECRYPT_I8T_IS_BYTE
#endif

#endif

#if (UINT_MAX / 0xFFU > 0xFFU)
#ifndef I16T
#define I16T int
#define U16C(v) (v##U)
#endif

#if (UINT_MAX / 0xFFFFU > 0xFFFFU)
#ifndef I32T
#define I32T int
#define U32C(v) (v##U)
#endif

#if (UINT_MAX / 0xFFFFFFFFU > 0xFFFFFFFFU)
#ifndef I64T
#define I64T int
#define U64C(v) (v##U)
#define ECRYPT_NATIVE64
#endif

#endif
#endif
#endif
#endif

/* --- check long --- */

#if (ULONG_MAX / 0xFUL > 0xFUL)
#ifndef I8T
#define I8T long
#define U8C(v) (v##UL)

#if (ULONG_MAX == 0xFFUL)
#define ECRYPT_I8T_IS_BYTE
#endif

#endif

#if (ULONG_MAX / 0xFFUL > 0xFFUL)
#ifndef I16T
#define I16T long
#define U16C(v) (v##UL)
#endif

#if (ULONG_MAX / 0xFFFFUL > 0xFFFFUL)
#ifndef I32T
#define I32T long
#define U32C(v) (v##UL)
#endif

#if (ULONG_MAX / 0xFFFFFFFFUL > 0xFFFFFFFFUL)
#ifndef I64T
#define I64T long
#define U64C(v) (v##UL)
#define ECRYPT_NATIVE64
#endif

#endif
#endif
#endif
#endif

/* --- check long long --- */

#ifdef ULLONG_MAX

#if (ULLONG_MAX / 0xFULL > 0xFULL)
#ifndef I8T
#define I8T long long
#define U8C(v) (v##ULL)

#if (ULLONG_MAX == 0xFFULL)
#define ECRYPT_I8T_IS_BYTE
#endif

#endif

#if (ULLONG_MAX / 0xFFULL > 0xFFULL)
#ifndef I16T
#define I16T long long
#define U16C(v) (v##ULL)
#endif

#if (ULLONG_MAX / 0xFFFFULL > 0xFFFFULL)
#ifndef I32T
#define I32T long long
#define U32C(v) (v##ULL)
#endif

#if (ULLONG_MAX / 0xFFFFFFFFULL > 0xFFFFFFFFULL)
#ifndef I64T
#define I64T long long
#define U64C(v) (v##ULL)
#endif

#endif
#endif
#endif
#endif

#endif

/* --- check __int64 --- */

#ifdef _UI64_MAX

#if (_UI64_MAX / 0xFFFFFFFFui64 > 0xFFFFFFFFui64)
#ifndef I64T
#define I64T __int64
#define U64C(v) (v##ui64)
#endif

#endif

#endif

/* ------------------------------------------------------------------------- */

#

#include <stdlib.h>

//#define ROTL32(v, n) _lrotl(v, n)
//#define ROTR32(v, n) _lrotr(v, n)
//#define ROTL64(v, n) _rotl64(v, n)
//#define ROTR64(v, n) _rotr64(v, n)

/*
 * WARNING: the conversions defined below are implemented as macros,
 * and should be used carefully. They should NOT be used with
 * parameters which perform some action. E.g., the following two lines
 * are not equivalent:
 *
 *  1) ++x; y = ROTL32(x, n);
 *  2) y = ROTL32(++x, n);
 */

/*
 * *** Please do not edit this file. ***
 *
 * The default macros can be overridden for specific architectures by
 * editing 'machine.h'.
 */

#ifndef ECRYPT_PORTABLE
#define ECRYPT_PORTABLE

#include <stdio.h>      // for printf
#include <inttypes.h>   // for printf(uint64_t)

#include "config.h"

/* ------------------------------------------------------------------------- */

/*
 * The following types are defined (if available):
 *
 * u8:  unsigned integer type, at least 8 bits
 * u16: unsigned integer type, at least 16 bits
 * u32: unsigned integer type, at least 32 bits
 * u64: unsigned integer type, at least 64 bits
 *
 * s8, s16, s32, s64 -> signed counterparts of u8, u16, u32, u64
 *
 * The selection of minimum-width integer types is taken care of by
 * 'config.h'. Note: to enable 64-bit types on 32-bit
 * compilers, it might be necessary to switch from ISO C90 mode to ISO
 * C99 mode (e.g., gcc -std=c99).
 */

#ifdef I8T
typedef signed I8T s8;
typedef unsigned I8T u8;
#endif

#ifdef I16T
typedef signed I16T s16;
typedef unsigned I16T u16;
#endif

#ifdef I32T
typedef signed I32T s32;
typedef unsigned I32T u32;
#endif

#ifdef I64T
typedef signed I64T s64;
typedef unsigned I64T u64;
#endif

/*
 * The following macros are used to obtain exact-width results.
 */

#define U8V(v) ((u8)(v) & U8C(0xFF))
#define U16V(v) ((u16)(v) & U16C(0xFFFF))
#define U32V(v) ((u32)(v) & U32C(0xFFFFFFFF))
#define U64V(v) ((u64)(v) & U64C(0xFFFFFFFFFFFFFFFF))

/* ------------------------------------------------------------------------- */

/*
 * The following macros return words with their bits rotated over n
 * positions to the left/right.
 */

#define ECRYPT_DEFAULT_ROT

#define ROTL8(v, n) \
  (U8V((v) << (n)) | ((v) >> (8 - (n))))

#define ROTL16(v, n) \
  (U16V((v) << (n)) | ((v) >> (16 - (n))))

#define ROTL32(v, n) \
  (U32V((v) << (n)) | ((v) >> (32 - (n))))

#define ROTL64(v, n) \
  (U64V((v) << (n)) | ((v) >> (64 - (n))))

#define ROTR8(v, n) ROTL8(v, 8 - (n))
#define ROTR16(v, n) ROTL16(v, 16 - (n))
#define ROTR32(v, n) ROTL32(v, 32 - (n))
#define ROTR64(v, n) ROTL64(v, 64 - (n))


/* ------------------------------------------------------------------------- */

/*
 * The following macros return a word with bytes in reverse order.
 */

#define ECRYPT_DEFAULT_SWAP

#define SWAP16(v) \
  ROTL16(v, 8)

#define SWAP32(v) \
  ((ROTL32(v,  8) & U32C(0x00FF00FF)) | \
   (ROTL32(v, 24) & U32C(0xFF00FF00)))

#ifdef ECRYPT_NATIVE64
#define SWAP64(v) \
  ((ROTL64(v,  8) & U64C(0x000000FF000000FF)) | \
   (ROTL64(v, 24) & U64C(0x0000FF000000FF00)) | \
   (ROTL64(v, 40) & U64C(0x00FF000000FF0000)) | \
   (ROTL64(v, 56) & U64C(0xFF000000FF000000)))
#else
#define SWAP64(v) \
  (((u64)SWAP32(U32V(v)) << 32) | (u64)SWAP32(U32V(v >> 32)))
#endif


#define ECRYPT_DEFAULT_WTOW

#ifdef ECRYPT_LITTLE_ENDIAN
#define U16TO16_LITTLE(v) (v)
#define U32TO32_LITTLE(v) (v)
#define U64TO64_LITTLE(v) (v)

#define U16TO16_BIG(v) SWAP16(v)
#define U32TO32_BIG(v) SWAP32(v)
#define U64TO64_BIG(v) SWAP64(v)
#endif

#ifdef ECRYPT_BIG_ENDIAN
#define U16TO16_LITTLE(v) SWAP16(v)
#define U32TO32_LITTLE(v) SWAP32(v)
#define U64TO64_LITTLE(v) SWAP64(v)

#define U16TO16_BIG(v) (v)
#define U32TO32_BIG(v) (v)
#define U64TO64_BIG(v) (v)
#endif


/*
 * The following macros load words from an array of bytes with
 * different types of endianness, and vice versa.
 */

#define ECRYPT_DEFAULT_BTOW

#if (!defined(ECRYPT_UNKNOWN) && defined(ECRYPT_I8T_IS_BYTE))

#define U8TO16_LITTLE(p) U16TO16_LITTLE(((u16*)(p))[0])
#define U8TO32_LITTLE(p) U32TO32_LITTLE(((u32*)(p))[0])
#define U8TO64_LITTLE(p) U64TO64_LITTLE(((u64*)(p))[0])

#define U8TO16_BIG(p) U16TO16_BIG(((u16*)(p))[0])
#define U8TO32_BIG(p) U32TO32_BIG(((u32*)(p))[0])
#define U8TO64_BIG(p) U64TO64_BIG(((u64*)(p))[0])

#define U16TO8_LITTLE(p, v) (((u16*)(p))[0] = U16TO16_LITTLE(v))
#define U32TO8_LITTLE(p, v) (((u32*)(p))[0] = U32TO32_LITTLE(v))
#define U64TO8_LITTLE(p, v) (((u64*)(p))[0] = U64TO64_LITTLE(v))

#define U16TO8_BIG(p, v) (((u16*)(p))[0] = U16TO16_BIG(v))
#define U32TO8_BIG(p, v) (((u32*)(p))[0] = U32TO32_BIG(v))
#define U64TO8_BIG(p, v) (((u64*)(p))[0] = U64TO64_BIG(v))

#else

#define U8TO16_LITTLE(p) \
  (((u16)((p)[0])      ) | \
   ((u16)((p)[1]) <<  8))

#define U8TO32_LITTLE(p) \
  (((u32)((p)[0])      ) | \
   ((u32)((p)[1]) <<  8) | \
   ((u32)((p)[2]) << 16) | \
   ((u32)((p)[3]) << 24))

#ifdef ECRYPT_NATIVE64
#define U8TO64_LITTLE(p) \
  (((u64)((p)[0])      ) | \
   ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | \
   ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | \
   ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | \
   ((u64)((p)[7]) << 56))
#else
#define U8TO64_LITTLE(p) \
  ((u64)U8TO32_LITTLE(p) | ((u64)U8TO32_LITTLE((p) + 4) << 32))
#endif

#define U8TO16_BIG(p) \
  (((u16)((p)[0]) <<  8) | \
   ((u16)((p)[1])      ))

#define U8TO32_BIG(p) \
  (((u32)((p)[0]) << 24) | \
   ((u32)((p)[1]) << 16) | \
   ((u32)((p)[2]) <<  8) | \
   ((u32)((p)[3])      ))

#ifdef ECRYPT_NATIVE64
#define U8TO64_BIG(p) \
  (((u64)((p)[0]) << 56) | \
   ((u64)((p)[1]) << 48) | \
   ((u64)((p)[2]) << 40) | \
   ((u64)((p)[3]) << 32) | \
   ((u64)((p)[4]) << 24) | \
   ((u64)((p)[5]) << 16) | \
   ((u64)((p)[6]) <<  8) | \
   ((u64)((p)[7])      ))
#else
#define U8TO64_BIG(p) \
  (((u64)U8TO32_BIG(p) << 32) | (u64)U8TO32_BIG((p) + 4))
#endif

#define U16TO8_LITTLE(p, v) \
  do { \
    (p)[0] = U8V((v)      ); \
    (p)[1] = U8V((v) >>  8); \
  } while (0)

#define U32TO8_LITTLE(p, v) \
  do { \
    (p)[0] = U8V((v)      ); \
    (p)[1] = U8V((v) >>  8); \
    (p)[2] = U8V((v) >> 16); \
    (p)[3] = U8V((v) >> 24); \
  } while (0)

#ifdef ECRYPT_NATIVE64
#define U64TO8_LITTLE(p, v) \
  do { \
    (p)[0] = U8V((v)      ); \
    (p)[1] = U8V((v) >>  8); \
    (p)[2] = U8V((v) >> 16); \
    (p)[3] = U8V((v) >> 24); \
    (p)[4] = U8V((v) >> 32); \
    (p)[5] = U8V((v) >> 40); \
    (p)[6] = U8V((v) >> 48); \
    (p)[7] = U8V((v) >> 56); \
  } while (0)
#else
#define U64TO8_LITTLE(p, v) \
  do { \
    U32TO8_LITTLE((p),     U32V((v)      )); \
    U32TO8_LITTLE((p) + 4, U32V((v) >> 32)); \
  } while (0)
#endif

#define U16TO8_BIG(p, v) \
  do { \
    (p)[0] = U8V((v)      ); \
    (p)[1] = U8V((v) >>  8); \
  } while (0)

#define U32TO8_BIG(p, v) \
  do { \
    (p)[0] = U8V((v) >> 24); \
    (p)[1] = U8V((v) >> 16); \
    (p)[2] = U8V((v) >>  8); \
    (p)[3] = U8V((v)      ); \
  } while (0)

#ifdef ECRYPT_NATIVE64
#define U64TO8_BIG(p, v) \
  do { \
    (p)[0] = U8V((v) >> 56); \
    (p)[1] = U8V((v) >> 48); \
    (p)[2] = U8V((v) >> 40); \
    (p)[3] = U8V((v) >> 32); \
    (p)[4] = U8V((v) >> 24); \
    (p)[5] = U8V((v) >> 16); \
    (p)[6] = U8V((v) >>  8); \
    (p)[7] = U8V((v)      ); \
  } while (0)
#else
#define U64TO8_BIG(p, v) \
  do { \
    U32TO8_BIG((p),     U32V((v) >> 32)); \
    U32TO8_BIG((p) + 4, U32V((v)      )); \
  } while (0)
#endif

#endif


#endif
#endif
#endif //SMALLSECRETLWE_MACROS_H
