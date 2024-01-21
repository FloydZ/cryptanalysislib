/*
	Copyright (C) 2014-2022 Igor van den Hoven ivdhoven@gmail.com
*/

/*
	Permission is hereby granted, free of charge, to any person obtaining
	a copy of this software and associated documentation files (the
	"Software"), to deal in the Software without restriction, including
	without limitation the rights to use, copy, modify, merge, publish,
	distribute, sublicense, and/or sell copies of the Software, and to
	permit persons to whom the Software is furnished to do so, subject to
	the following conditions:

	The above copyright notice and this permission notice shall be
	included in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
	MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
	CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
	TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
	SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
	quadsort 1.2.1.2
*/
// When sorting an const_array of pointers, like a string const_array, the QUAD_CACHE needs
// to be set for proper performance when sorting large arrays.
// quadsort_prim() can be used to sort arrays of 32 and 64 bit integers
// without a comparison function or cache restrictions.

// With a 6 MB L3 cache a value of 262144 works well.

#include <cstddef>
#include <cstdlib>

#ifndef CRYPTANALYSISLIB_SORT_QUADSORT
#define CRYPTANALYSISLIB_SORT_QUADSORT

#define QUAD_CACHE (131072/2)
//#define QUAD_CACHE 131072
//#define QUAD_CACHE 262144
//#define QUAD_CACHE 524288
//#define QUAD_CACHE 4294967295

// utilize branchless ternary operations in clang

#if !defined __clang__
#define head_branchless_merge(ptd, x, ptl, ptr, cmp)  \
	x = cmp(ptl, ptr) <= 0;  \
	*ptd = *ptl;  \
	ptl += x;  \
	ptd[x] = *ptr;  \
	ptr += !x;  \
	ptd++;
#else
#define head_branchless_merge(ptd, x, ptl, ptr, cmp)  \
	*ptd++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;
#endif

#if !defined __clang__
#define tail_branchless_merge(tpd, x, tpl, tpr, cmp)  \
	y = cmp(tpl, tpr) <= 0;  \
	*tpd = *tpl;  \
	tpl -= !x;  \
	tpd--;  \
	tpd[x] = *tpr;  \
	tpr -= y;
#else
#define tail_branchless_merge(tpd, x, tpl, tpr, cmp)  \
	*tpd-- = cmp(tpl, tpr) > 0 ? *tpl-- : *tpr--;
#endif

// guarantee small parity merges are inlined with minimal overhead

#define parity_merge_two(array, swap, x, y, ptl, ptr, pts, cmp)  \
	ptl = array; ptr = array + 2; pts = swap;  \
	head_branchless_merge(pts, x, ptl, ptr, cmp);  \
	*pts = cmp(ptl, ptr) <= 0 ? *ptl : *ptr;  \
  \
	ptl = array + 1; ptr = array + 3; pts = swap + 3;  \
	tail_branchless_merge(pts, y, ptl, ptr, cmp);  \
	*pts = cmp(ptl, ptr)  > 0 ? *ptl : *ptr;

#define parity_merge_four(array, swap, x, y, ptl, ptr, pts, cmp)  \
	ptl = array + 0; ptr = array + 4; pts = swap;  \
	head_branchless_merge(pts, x, ptl, ptr, cmp);  \
	head_branchless_merge(pts, x, ptl, ptr, cmp);  \
	head_branchless_merge(pts, x, ptl, ptr, cmp);  \
	*pts = cmp(ptl, ptr) <= 0 ? *ptl : *ptr;  \
  \
	ptl = array + 3; ptr = array + 7; pts = swap + 7;  \
	tail_branchless_merge(pts, y, ptl, ptr, cmp);  \
	tail_branchless_merge(pts, y, ptl, ptr, cmp);  \
	tail_branchless_merge(pts, y, ptl, ptr, cmp);  \
	*pts = cmp(ptl, ptr)  > 0 ? *ptl : *ptr;

#define swap_branchless(pta, swap, x, y, cmp)  \
	x = cmp(pta, pta + 1) > 0;  \
	y = !x;  \
	swap = pta[y];  \
	pta[0] = pta[x];  \
	pta[1] = swap;
// the next seven functions are used for sorting 0 to 31 elements

template<typename VAR, typename CMPFUNC>
void tiny_sort(VAR *array, size_t nmemb, CMPFUNC cmp)
{
	VAR swap, *pta;
	size_t x, y;

	switch (nmemb)
	{
		case 4:
			pta = array;
			x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap; pta += 2;
			x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap; pta--;

			if (cmp(pta, pta + 1) > 0)
			{
				swap = pta[0]; pta[0] = pta[1]; pta[1] = swap; pta--;

				x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap; pta += 2;
				x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap; pta--;
				x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap;
			}
			return;
		case 3:
			pta = array;
			x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap; pta++;
			x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap;
			__attribute__ ((fallthrough));
		case 2:
			pta = array;
			x = cmp(pta, pta + 1) > 0; y = !x; swap = pta[y]; pta[0] = pta[x]; pta[1] = swap;
			__attribute__ ((fallthrough));
		case 1:
			__attribute__ ((fallthrough));
		case 0:
			return;
	}
}

// This function requires a minimum offset of 2 to work properly

template<typename VAR, typename CMPFUNC>
void twice_unguarded_insert(VAR *array, size_t offset, size_t nmemb, CMPFUNC cmp)
{
	VAR key, *pta, *end;
	size_t i, top, x, y;

	for (i = offset ; i < nmemb ; i++)
	{
		pta = end = array + i;

		if (cmp(--pta, end) <= 0)
		{
			continue;
		}

		key = *end;

		if (cmp(array + 1, &key) > 0)
		{
			top = i - 1;

			do
			{
				*end-- = *pta--;
			}
			while (--top);

			*end-- = key;
		}
		else
		{
			do
			{
				*end-- = *pta--;
				*end-- = *pta--;
			}
			while (cmp(pta, &key) > 0);

			end[0] = end[1];
			end[1] = key;
		}
		x = cmp(end, end + 1) > 0; y = !x; key = end[y]; end[0] = end[x]; end[1] = key;
	}
}

template<typename VAR, typename CMPFUNC>
void quad_swap_four(VAR *array, CMPFUNC cmp)
{
	VAR *pta, swap;
	size_t x, y;

	pta = array;

	swap_branchless(pta, swap, x, y, cmp); pta += 2;
	swap_branchless(pta, swap, x, y, cmp); pta--;

	if (cmp(pta, pta + 1) > 0)
	{
		swap = pta[0]; pta[0] = pta[1]; pta[1] = swap; pta--;

		swap_branchless(pta, swap, x, y, cmp); pta += 2;
		swap_branchless(pta, swap, x, y, cmp); pta--;
		swap_branchless(pta, swap, x, y, cmp);
	}
}

template<typename VAR, typename CMPFUNC>
void parity_swap_eight(VAR *array, VAR *swap, CMPFUNC cmp)
{
	VAR *ptl, *ptr, *pts;
	size_t x, y;

	ptl = array;

	swap_branchless(ptl, swap[0], x, y, cmp); ptl += 2;
	swap_branchless(ptl, swap[0], x, y, cmp); ptl += 2;
	swap_branchless(ptl, swap[0], x, y, cmp); ptl += 2;
	swap_branchless(ptl, swap[0], x, y, cmp);

	if (cmp(array + 1, array + 2) <= 0 && cmp(array + 3, array + 4) <= 0 && cmp(array + 5, array + 6) <= 0)
	{
		return;
	}
	parity_merge_two(array + 0, swap + 0, x, y, ptl, ptr, pts, cmp);
	parity_merge_two(array + 4, swap + 4, x, y, ptl, ptr, pts, cmp);

	parity_merge_four(swap, array, x, y, ptl, ptr, pts, cmp);
}

// left must be equal or one smaller than right

template<typename VAR, typename CMPFUNC>
void parity_merge(VAR *dest, VAR *from, size_t left, size_t right, CMPFUNC cmp)
{
	VAR *ptl, *ptr, *tpl, *tpr, *tpd, *ptd;

	ptl = from;
	ptr = from + left;
	ptd = dest;
	tpl = ptr - 1;
	tpr = tpl + right;
	tpd = dest + left + right - 1;

	if (left < right)
	{
		*ptd++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;
	}

	*ptd++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;

#if !defined cmp && !defined __clang__ // cache limit workaround for gcc
	if (left > QUAD_CACHE)
	{
		while (--left)
		{
			*ptd++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;
			*tpd-- = cmp(tpl, tpr)  > 0 ? *tpl-- : *tpr--;
		}
	}
	else
#endif
	{
		while (--left)
		{
			head_branchless_merge(ptd, x, ptl, ptr, cmp);
			tail_branchless_merge(tpd, y, tpl, tpr, cmp);
		}
	}
	*tpd = cmp(tpl, tpr)  > 0 ? *tpl : *tpr;
}


template<typename VAR, typename CMPFUNC>
void parity_swap_sixteen(VAR *array, VAR *swap, CMPFUNC cmp) {
	VAR *ptl, *ptr, *pts;
	quad_swap_four(array +  0, cmp);
	quad_swap_four(array +  4, cmp);
	quad_swap_four(array +  8, cmp);
	quad_swap_four(array + 12, cmp);

	if (cmp(array + 3, array + 4) <= 0 && cmp(array + 7, array + 8) <= 0 && cmp(array + 11, array + 12) <= 0) {
		return;
	}

	parity_merge_four(array + 0, swap + 0, (void), (void), ptl, ptr, pts, cmp);
	parity_merge_four(array + 8, swap + 8, (void), (void), ptl, ptr, pts, cmp);

	parity_merge(array, swap, 8, 8, cmp);
}

template<typename VAR, typename CMPFUNC>
void tail_swap(VAR *array, VAR *swap, size_t nmemb, CMPFUNC cmp)
{
	if (nmemb < 5)
	{
		tiny_sort(array, nmemb, cmp);
		return;
	}
	if (nmemb < 8)
	{
		quad_swap_four(array, cmp);
		twice_unguarded_insert(array, 4, nmemb, cmp);
		return;
	}
	if (nmemb < 12)
	{
		parity_swap_eight(array, swap, cmp);
		twice_unguarded_insert(array, 8, nmemb, cmp);
		return;
	}
	if (nmemb >= 16 && nmemb < 24)
	{
		parity_swap_sixteen(array, swap, cmp);
		twice_unguarded_insert(array, 16, nmemb, cmp);
		return;
	}

	size_t quad1, quad2, quad3, quad4, half1, half2;

	half1 = nmemb / 2;
	quad1 = half1 / 2;
	quad2 = half1 - quad1;

	half2 = nmemb - half1;
	quad3 = half2 / 2;
	quad4 = half2 - quad3;

	VAR *pta = array;

	tail_swap(pta, swap, quad1, cmp); pta += quad1;
	tail_swap(pta, swap, quad2, cmp); pta += quad2;
	tail_swap(pta, swap, quad3, cmp); pta += quad3;
	tail_swap(pta, swap, quad4, cmp);

	if (cmp(array + quad1 - 1, array + quad1) <= 0 &&
	    cmp(array + half1 - 1, array + half1) <= 0 &&
	    cmp(pta - 1, pta) <= 0)
	{
		return;
	}

	parity_merge(swap, array, quad1, quad2, cmp);
	parity_merge(swap + half1, array + half1, quad3, quad4, cmp);
	parity_merge(array, swap, half1, half2, cmp);
}

// the next three functions create sorted blocks of 32 elements

template<typename VAR>
void quad_reversal(VAR *pta, VAR *ptz)
{
	VAR *ptb, *pty, tmp1, tmp2;

	size_t loop = (ptz - pta) / 2;

	ptb = pta + loop;
	pty = ptz - loop;

	if (loop % 2 == 0)
	{
		tmp2 = *ptb; *ptb-- = *pty; *pty++ = tmp2; loop--;
	}

	loop /= 2;

	do
	{
		tmp1 = *pta; *pta++ = *ptz; *ptz-- = tmp1;
		tmp2 = *ptb; *ptb-- = *pty; *pty++ = tmp2;
	}
	while (loop--);
}

template<typename VAR, typename CMPFUNC>
void quad_swap_merge(VAR *array, VAR *swap, CMPFUNC cmp)
{
	VAR *pts, *ptl, *ptr;
	parity_merge_two(array + 0, swap + 0, (void), (void), ptl, ptr, pts, cmp);
	parity_merge_two(array + 4, swap + 4, (void), (void), ptl, ptr, pts, cmp);

	parity_merge_four(swap, array, x, y, ptl, ptr, pts, cmp);
}

template<typename VAR, typename CMPFUNC>
void tail_merge(VAR *array, VAR *swap, size_t swap_size, size_t nmemb, size_t block, CMPFUNC cmp);

template<typename VAR, typename CMPFUNC>
size_t quad_swap(VAR *array, size_t nmemb, CMPFUNC cmp)
{
	VAR swap[32];
	size_t count;
	VAR *pta, *pts;
	unsigned char v1, v2, v3, v4, x;
	pta = array;

	count = nmemb / 8;

	while (count--)
	{
		v1 = cmp(pta + 0, pta + 1) > 0;
		v2 = cmp(pta + 2, pta + 3) > 0;
		v3 = cmp(pta + 4, pta + 5) > 0;
		v4 = cmp(pta + 6, pta + 7) > 0;

		switch (v1 + v2 * 2 + v3 * 4 + v4 * 8)
		{
			case 0:
				if (cmp(pta + 1, pta + 2) <= 0 && cmp(pta + 3, pta + 4) <= 0 && cmp(pta + 5, pta + 6) <= 0)
				{
					goto ordered;
				}
				quad_swap_merge(pta, swap, cmp);
				break;

			case 15:
				if (cmp(pta + 1, pta + 2) > 0 && cmp(pta + 3, pta + 4) > 0 && cmp(pta + 5, pta + 6) > 0)
				{
					pts = pta;
					goto reversed;
				}
				__attribute__ ((fallthrough));
			default:
			not_ordered:
				x = !v1;
				swap[0] = pta[x];
				pta[0] = pta[v1];
				pta[1] = swap[0];
				pta += 2;
				x = !v2; swap[0] = pta[x]; pta[0] = pta[v2]; pta[1] = swap[0]; pta += 2;
				x = !v3; swap[0] = pta[x]; pta[0] = pta[v3]; pta[1] = swap[0]; pta += 2;
				x = !v4; swap[0] = pta[x]; pta[0] = pta[v4]; pta[1] = swap[0]; pta -= 6;

				quad_swap_merge(pta, swap, cmp);
		}
		pta += 8;

		continue;

		ordered:

		pta += 8;

		if (count--)
		{
			if ((v1 = cmp(pta + 0, pta + 1) > 0) ||
			    (v2 = cmp(pta + 2, pta + 3) > 0) ||
			    (v3 = cmp(pta + 4, pta + 5) > 0) ||
			    (v4 = cmp(pta + 6, pta + 7) > 0))
			{
				if (v1 + v2 + v3 + v4 == 4 && cmp(pta + 1, pta + 2) > 0 && cmp(pta + 3, pta + 4) > 0 && cmp(pta + 5, pta + 6) > 0)
				{
					pts = pta;
					goto reversed;
				}
				goto not_ordered;
			}
			if (cmp(pta + 1, pta + 2) <= 0 && cmp(pta + 3, pta + 4) <= 0 && cmp(pta + 5, pta + 6) <= 0)
			{
				goto ordered;
			}
			quad_swap_merge(pta, swap, cmp);
			pta += 8;
			continue;
		}
		break;

		reversed:

		pta += 8;

		if (count--)
		{
			if ((v1 = cmp(pta + 0, pta + 1) <= 0) ||
			    (v2 = cmp(pta + 2, pta + 3) <= 0) ||
			    (v3 = cmp(pta + 4, pta + 5) <= 0) ||
			    (v4 = cmp(pta + 6, pta + 7) <= 0))
			{
				// not reversed
			}
			else
			{
				if (cmp(pta - 1, pta) > 0 && cmp(pta + 1, pta + 2) > 0 && cmp(pta + 3, pta + 4) > 0 && cmp(pta + 5, pta + 6) > 0)
				{
					goto reversed;
				}
			}
			quad_reversal<VAR>(pts, pta - 1);

			if (v1 + v2 + v3 + v4 == 4 && cmp(pta + 1, pta + 2) <= 0 && cmp(pta + 3, pta + 4) <= 0 && cmp(pta + 5, pta + 6) <= 0)
			{
				goto ordered;
			}
			if (v1 + v2 + v3 + v4 == 0 && cmp(pta + 1, pta + 2)  > 0 && cmp(pta + 3, pta + 4)  > 0 && cmp(pta + 5, pta + 6)  > 0)
			{
				pts = pta;
				goto reversed;
			}

			x = !v1; swap[0] = pta[v1]; pta[0] = pta[x]; pta[1] = swap[0]; pta += 2;
			x = !v2; swap[0] = pta[v2]; pta[0] = pta[x]; pta[1] = swap[0]; pta += 2;
			x = !v3; swap[0] = pta[v3]; pta[0] = pta[x]; pta[1] = swap[0]; pta += 2;
			x = !v4; swap[0] = pta[v4]; pta[0] = pta[x]; pta[1] = swap[0]; pta -= 6;

			if (cmp(pta + 1, pta + 2) > 0 || cmp(pta + 3, pta + 4) > 0 || cmp(pta + 5, pta + 6) > 0)
			{
				quad_swap_merge(pta, swap, cmp);
			}
			pta += 8;
			continue;
		}

		switch (nmemb % 8)
		{
			case 7: if (cmp(pta + 5, pta + 6) <= 0) { break; }
				__attribute__ ((fallthrough));
			case 6: if (cmp(pta + 4, pta + 5) <= 0) { break; }
				__attribute__ ((fallthrough));
			case 5: if (cmp(pta + 3, pta + 4) <= 0) { break; }
				__attribute__ ((fallthrough));
			case 4: if (cmp(pta + 2, pta + 3) <= 0) { break; }
				__attribute__ ((fallthrough));
			case 3: if (cmp(pta + 1, pta + 2) <= 0) { break; }
				__attribute__ ((fallthrough));
			case 2: if (cmp(pta + 0, pta + 1) <= 0) { break; }
				__attribute__ ((fallthrough));
			case 1: if (cmp(pta - 1, pta + 0) <= 0) { break; }
				__attribute__ ((fallthrough));
			case 0:
				quad_reversal(pts, pta + nmemb % 8 - 1);

				if (pts == array)
				{
					return 1;
				}
				goto reverse_end;
		}
		quad_reversal(pts, pta - 1);
		break;
	}
	tail_swap(pta, swap, nmemb % 8, cmp);

	reverse_end:

	pta = array;

	for (count = nmemb / 32 ; count-- ; pta += 32)
	{
		if (cmp(pta + 7, pta + 8) <= 0 && cmp(pta + 15, pta + 16) <= 0 && cmp(pta + 23, pta + 24) <= 0)
		{
			continue;
		}
		parity_merge(swap, pta, 8, 8, cmp);
		parity_merge(swap + 16, pta + 16, 8, 8, cmp);
		parity_merge(pta, swap, 16, 16, cmp);
	}

	if (nmemb % 32 > 8)
	{
		tail_merge(pta, swap, 32, nmemb % 32, 8, cmp);
	}
	return 0;
}

// quad merge support routines

template<typename VAR, typename CMPFUNC>
void cross_merge(VAR *dest, VAR *from, size_t left, size_t right, CMPFUNC cmp)
{
	VAR *ptl, *tpl, *ptr, *tpr, *ptd, *tpd;
	size_t loop;
	ptl = from;
	ptr = from + left;
	tpl = ptr - 1;
	tpr = tpl + right;

	if (cmp(ptl + 15, ptr) > 0 && cmp(ptl, ptr + 15) <= 0 && cmp(tpl, tpr - 15) > 0 && cmp(tpl - 15, tpr) <= 0) {
		parity_merge(dest, from, left, right, cmp);
		return;
	}

	ptd = dest;
	tpd = dest + left + right - 1;

	while (tpl - ptl > 8 && tpr - ptr > 8) {
		ptl8_ptr: if (cmp(ptl + 7, ptr) <= 0) {
			loop = 8; do *ptd++ = *ptl++; while (--loop);
			
			if (tpl - ptl > 8) {goto ptl8_ptr;} break;
		}

		ptl_ptr8: if (cmp(ptl, ptr + 7) > 0) {
			loop = 8; do *ptd++ = *ptr++; while (--loop);

			if (tpr - ptr > 8) {goto ptl_ptr8;} break;
		}

		tpl_tpr8: if (cmp(tpl, tpr - 7) <= 0)
		{
			loop = 8; do *tpd-- = *tpr--; while (--loop);
			
			if (tpr - ptr > 8) {goto tpl_tpr8;} break;
		}

		tpl8_tpr: if (cmp(tpl - 7, tpr) > 0)
		{
			loop = 8; do *tpd-- = *tpl--; while (--loop);
			
			if (tpl - ptl > 8) {goto tpl8_tpr;} break;
		}

#if !defined cmp && !defined __clang__
		if (left > QUAD_CACHE)
		{
			loop = 8; do
			{
				*ptd++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;
				*tpd-- = cmp(tpl, tpr)  > 0 ? *tpl-- : *tpr--;
			}
			while (--loop);
		}
		else
#endif
		{
			loop = 8; do
			{
				head_branchless_merge(ptd, x, ptl, ptr, cmp);
				tail_branchless_merge(tpd, y, tpl, tpr, cmp);
			}
			while (--loop);
		}
	}

	if (cmp(tpl, tpr) <= 0)
	{
		while (ptl <= tpl)
		{
			*ptd++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;
		}
		while (ptr <= tpr)
		{
			*ptd++ = *ptr++;
		}
	}
	else
	{
		while (ptr <= tpr)
		{
			*ptd++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;
		}
		while (ptl <= tpl)
		{
			*ptd++ = *ptl++;
		}
	}
}

// main memory: [A][B][C][D]
// swap memory: [A  B]       step 1
// swap memory: [A  B][C  D] step 2
// main memory: [A  B  C  D] step 3

template<typename VAR, typename CMPFUNC>
void quad_merge_block(VAR *array, VAR *swap, size_t block, CMPFUNC cmp)
{
	VAR *pt1, *pt2, *pt3;
	size_t block_x_2 = block * 2;

	pt1 = array + block;
	pt2 = pt1 + block;
	pt3 = pt2 + block;

	switch ((cmp(pt1 - 1, pt1) <= 0) | (cmp(pt3 - 1, pt3) <= 0) * 2)
	{
		case 0:
			cross_merge(swap, array, block, block, cmp);
			cross_merge(swap + block_x_2, pt2, block, block, cmp);
			break;
		case 1:
			memcpy(swap, array, block_x_2 * sizeof(VAR));
			cross_merge(swap + block_x_2, pt2, block, block, cmp);
			break;
		case 2:
			cross_merge(swap, array, block, block, cmp);
			memcpy(swap + block_x_2, pt2, block_x_2 * sizeof(VAR));
			break;
		case 3:
			if (cmp(pt2 - 1, pt2) <= 0)
				return;
			memcpy(swap, array, block_x_2 * 2 * sizeof(VAR));
	}
	cross_merge(array, swap, block_x_2, block_x_2, cmp);
}

template<typename VAR, typename CMPFUNC>
size_t quad_merge(VAR *array, VAR *swap, size_t swap_size, size_t nmemb, size_t block, CMPFUNC cmp)
{
	VAR *pta, *pte;

	pte = array + nmemb;

	block *= 4;

	while (block <= nmemb && block <= swap_size)
	{
		pta = array;

		do
		{
			quad_merge_block(pta, swap, block / 4, cmp);

			pta += block;
		}
		while (pta + block <= pte);

		tail_merge(pta, swap, swap_size, pte - pta, block / 4, cmp);

		block *= 4;
	}

	tail_merge(array, swap, swap_size, nmemb, block / 4, cmp);

	return block / 2;
}

template<typename VAR, typename CMPFUNC>
void partial_forward_merge(VAR *array, VAR *swap, size_t nmemb, size_t block, CMPFUNC cmp)
{
	VAR *ptl, *ptr, *tpl, *tpr;
	size_t x, y;

	if (nmemb == block)
	{
		return;
	}

	ptr = array + block;
	tpr = array + nmemb - 1;

	if (cmp(ptr - 1, ptr) <= 0)
	{
		return;
	}

	memcpy(swap, array, block * sizeof(VAR));

	ptl = swap;
	tpl = swap + block - 1;


	while (ptl < tpl - 1 && ptr < tpr - 1)
	{
		if (cmp(ptl, ptr + 1) > 0)
		{
			*array++ = *ptr++; *array++ = *ptr++;
		}
		else if (cmp(ptl + 1, ptr) <= 0)
		{
			*array++ = *ptl++; *array++ = *ptl++;
		}
		else 
		{
			x = cmp(ptl, ptr) <= 0; y = !x; array[x] = *ptr; ptr += 1; array[y] = *ptl; ptl += 1; array += 2;
			head_branchless_merge(array, x, ptl, ptr, cmp);
		}
	}

	while (ptl <= tpl && ptr <= tpr)
	{
		*array++ = cmp(ptl, ptr) <= 0 ? *ptl++ : *ptr++;
	}

	while (ptl <= tpl)
	{
		*array++ = *ptl++;
	}
}

template<typename VAR, typename CMPFUNC>
void partial_backward_merge(VAR *array, VAR *swap, size_t nmemb, size_t block, CMPFUNC cmp)
{
	VAR *tpl, *tpa, *tpr; // tail pointer left, const_array, right
	size_t loop, x, y;

	if (nmemb == block)
	{
		return;
	}

	tpl = array + block - 1;
	tpa = array + nmemb - 1;

	if (cmp(tpl, tpl + 1) <= 0)
	{
		return;
	}

	memcpy(swap, array + block, (nmemb - block) * sizeof(VAR));

	tpr = swap + nmemb - block - 1;

	while (tpl > array + 32 && tpr > swap + 32)
	{
		tpl_tpr32: if (cmp(tpl, tpr - 31) <= 0)
		{
			loop = 32; do *tpa-- = *tpr--; while (--loop);

			if (tpr > swap + 32) {goto tpl_tpr32;} break;
		}

		tpl32_tpr: if (cmp(tpl - 31, tpr) > 0)
		{
			loop = 32; do *tpa-- = *tpl--; while (--loop);
			
			if (tpl > array + 32) {goto tpl32_tpr;} break;
		}

		loop = 16; do
		{
			if (cmp(tpl, tpr - 1) <= 0)
			{
				*tpa-- = *tpr--; *tpa-- = *tpr--;
			}
			else if (cmp(tpl - 1, tpr) > 0)
			{
				*tpa-- = *tpl--; *tpa-- = *tpl--;
			}
			else
			{
				x = cmp(tpl, tpr) <= 0; y = !x; tpa--; tpa[x] = *tpr; tpr -= 1; tpa[y] = *tpl; tpl -= 1; tpa--;
				tail_branchless_merge(tpa, y, tpl, tpr, cmp);
			}
		}
		while (--loop);
	}

	while (tpr > swap + 1 && tpl > array + 1)
	{
		if (cmp(tpl, tpr - 1) <= 0)
		{
			*tpa-- = *tpr--; *tpa-- = *tpr--;
		}
		else if (cmp(tpl - 1, tpr) > 0)
		{
			*tpa-- = *tpl--; *tpa-- = *tpl--;
		}
		else
		{
			x = cmp(tpl, tpr) <= 0; y = !x; tpa--; tpa[x] = *tpr; tpr -= 1; tpa[y] = *tpl; tpl -= 1; tpa--;
			tail_branchless_merge(tpa, y, tpl, tpr, cmp);
		}
	}

	while (tpr >= swap && tpl >= array)
	{
		*tpa-- = cmp(tpl, tpr) > 0 ? *tpl-- : *tpr--;
	}

	while (tpr >= swap)
	{
		*tpa-- = *tpr--;
	}
}

template<typename VAR, typename CMPFUNC>
void tail_merge(VAR *array, VAR *swap, size_t swap_size, size_t nmemb, size_t block, CMPFUNC cmp)
{
	VAR *pta, *pte;

	pte = array + nmemb;

	while (block < nmemb && block <= swap_size)
	{
		for (pta = array ; pta + block < pte ; pta += block * 2)
		{
			if (pta + block * 2 < pte)
			{
				partial_backward_merge(pta, swap, block * 2, block, cmp);

				continue;
			}
			partial_backward_merge(pta, swap, pte - pta, block, cmp);

			break;
		}
		block *= 2;
	}
}

// the next four functions provide in-place rotate merge support

template<typename VAR>
void trinity_rotation(VAR *array, VAR *swap, size_t swap_size, size_t nmemb, size_t left)
{
	size_t bridge, right = nmemb - left;

	if (swap_size > 65536)
	{
		swap_size = 65536;
	}

	if (left < right)
	{
		if (left <= swap_size)
		{
			memcpy(swap, array, left * sizeof(VAR));
			memmove(array, array + left, right * sizeof(VAR));
			memcpy(array + right, swap, left * sizeof(VAR));
		}
		else
		{
			VAR *pta, *ptb, *ptc, *ptd;

			pta = array;
			ptb = pta + left;

			bridge = right - left;

			if (bridge <= swap_size && bridge > 3)
			{
				ptc = pta + right;
				ptd = ptc + left;

				memcpy(swap, ptb, bridge * sizeof(VAR));

				while (left--)
				{
					*--ptc = *--ptd; *ptd = *--ptb;
				}
				memcpy(pta, swap, bridge * sizeof(VAR));
			}
			else
			{
				ptc = ptb;
				ptd = ptc + right;

				bridge = left / 2;

				while (bridge--)
				{
					*swap = *--ptb; *ptb = *pta; *pta++ = *ptc; *ptc++ = *--ptd; *ptd = *swap;
				}

				bridge = (ptd - ptc) / 2;

				while (bridge--)
				{
					*swap = *ptc; *ptc++ = *--ptd; *ptd = *pta; *pta++ = *swap;
				}

				bridge = (ptd - pta) / 2;

				while (bridge--)
				{
					*swap = *pta; *pta++ = *--ptd; *ptd = *swap;
				}
			}
		}
	}
	else if (right < left)
	{
		if (right <= swap_size)
		{
			memcpy(swap, array + left, right * sizeof(VAR));
			memmove(array + right, array, left * sizeof(VAR));
			memcpy(array, swap, right * sizeof(VAR));
		}
		else
		{
			VAR *pta, *ptb, *ptc, *ptd;

			pta = array;
			ptb = pta + left;

			bridge = left - right;

			if (bridge <= swap_size && bridge > 3)
			{
				ptc = pta + right;
				ptd = ptc + left;

				memcpy(swap, ptc, bridge * sizeof(VAR));

				while (right--)
				{
					*ptc++ = *pta; *pta++ = *ptb++;
				}
				memcpy(ptd - bridge, swap, bridge * sizeof(VAR));
			}
			else
			{
				ptc = ptb;
				ptd = ptc + right;

				bridge = right / 2;

				while (bridge--)
				{
					*swap = *--ptb; *ptb = *pta; *pta++ = *ptc; *ptc++ = *--ptd; *ptd = *swap;
				}

				bridge = (ptb - pta) / 2;

				while (bridge--)
				{
					*swap = *--ptb; *ptb = *pta; *pta++ = *--ptd; *ptd = *swap;
				}

				bridge = (ptd - pta) / 2;

				while (bridge--)
				{
					*swap = *pta; *pta++ = *--ptd; *ptd = *swap;
				}
			}
		}
	}
	else
	{
		VAR *pta, *ptb;

		pta = array;
		ptb = pta + left;

		while (left--)
		{
			*swap = *pta; *pta++ = *ptb; *ptb++ = *swap;
		}
	}
}

template<typename VAR, typename CMPFUNC>
size_t monobound_binary_first(VAR *array, VAR *value, size_t top, CMPFUNC cmp)
{
	VAR *end;
	size_t mid;

	end = array + top;

	while (top > 1)
	{
		mid = top / 2;

		if (cmp(value, end - mid) <= 0)
		{
			end -= mid;
		}
		top -= mid;
	}

	if (cmp(value, end - 1) <= 0)
	{
		end--;
	}
	return (end - array);
}

template<typename VAR, typename CMPFUNC>
void blit_merge_block(VAR *array, VAR *swap, size_t swap_size, size_t lblock, size_t right, CMPFUNC cmp)
{
	size_t left, rblock;

	if (cmp(array + lblock - 1, array + lblock) <= 0)
	{
		return;
	}

	rblock = lblock / 2;
	lblock -= rblock;

	left = monobound_binary_first(array + lblock + rblock, array + lblock, right, cmp);

	right -= left;

	if (left)
	{
		trinity_rotation(array + lblock, swap, swap_size, rblock + left, rblock);

		if (left <= swap_size)
		{
			partial_backward_merge(array, swap, lblock + left, lblock, cmp);
		}
		else if (lblock <= swap_size)
		{
			partial_forward_merge(array, swap, lblock + left, lblock, cmp);
		}
		else
		{
			blit_merge_block(array, swap, swap_size, lblock, left, cmp);
		}
	}

	if (right)
	{
		if (right <= swap_size)
		{
			partial_backward_merge(array + lblock + left, swap, rblock + right, rblock, cmp);
		}
		else if (rblock <= swap_size)
		{
			partial_forward_merge(array + lblock + left, swap, rblock + right, rblock, cmp);
		}
		else
		{
			blit_merge_block(array + lblock + left, swap, swap_size, rblock, right, cmp);
		}
	}
}

template<typename VAR, typename CMPFUNC>
void blit_merge(VAR *array, VAR *swap, size_t swap_size, size_t nmemb, size_t block, CMPFUNC cmp)
{
	VAR *pta, *pte;

	pte = array + nmemb;

	if (nmemb - block <= swap_size)
	{
		partial_backward_merge(array, swap, nmemb, block, cmp);

		return;
	}

	while (block < nmemb)
	{
		for (pta = array ; pta + block < pte ; pta += block * 2)
		{
			if (pta + block * 2 < pte)
			{
				blit_merge_block(pta, swap, swap_size, block, block, cmp);

				continue;
			}
			blit_merge_block(pta, swap, swap_size, block, pte - pta - block, cmp);

			break;
		}
		block *= 2;
	}
}

///////////////////////////////////////////////////////////////////////////////
//┌─────────────────────────────────────────────────────────────────────────┐//
//│    ██████┐ ██┐   ██┐ █████┐ ██████┐ ███████┐ ██████┐ ██████┐ ████████┐  │//
//│   ██┌───██┐██│   ██│██┌──██┐██┌──██┐██┌────┘██┌───██┐██┌──██┐└──██┌──┘  │//
//│   ██│   ██│██│   ██│███████│██│  ██│███████┐██│   ██│██████┌┘   ██│     │//
//│   ██│▄▄ ██│██│   ██│██┌──██│██│  ██│└────██│██│   ██│██┌──██┐   ██│     │//
//│   └██████┌┘└██████┌┘██│  ██│██████┌┘███████│└██████┌┘██│  ██│   ██│     │//
//│    └──▀▀─┘  └─────┘ └─┘  └─┘└─────┘ └──────┘ └─────┘ └─┘  └─┘   └─┘     │//
//└─────────────────────────────────────────────────────────────────────────┘//
///////////////////////////////////////////////////////////////////////////////

template<typename VAR, class CMPFUNC>
void quadsort(void *array, const size_t nmemb, CMPFUNC cmp)
{
	VAR *pta = (VAR *) array;

	if (nmemb < 32) {
		VAR swap[32];

		tail_swap(pta, swap, nmemb, cmp);
	}
	else if (quad_swap(pta, nmemb, cmp) == 0) {
		VAR *swap = NULL;
		size_t block, swap_size = 32;

		while (swap_size * 4 <= nmemb) {
			swap_size *= 4;
		}
		swap = (VAR *) malloc(swap_size * sizeof(VAR));

		if (swap == NULL) {
			VAR stack[512];

			tail_merge(pta, stack, 32, nmemb, 32, cmp);

			blit_merge(pta, stack, 32, nmemb, 64, cmp);

			return;
		}
		block = quad_merge(pta, swap, swap_size, nmemb, 32, cmp);

		blit_merge(pta, swap, swap_size, nmemb, block, cmp);

		free(swap);
	}
}

template<typename VAR, typename CMPFUNC>
void quadsort_swap(void *array,
                   void *swap,
                   size_t swap_size,
                   size_t nmemb, CMPFUNC cmp)
{
	VAR *pta = (VAR *) array;
	VAR *pts = (VAR *) swap;

	if (nmemb <= 96) {
		tail_swap(pta, pts, nmemb, cmp);
	}
	else if (quad_swap(pta, nmemb, cmp) == 0) {
		size_t block = quad_merge(pta, pts, swap_size, nmemb, 32, cmp);

		blit_merge(pta, pts, swap_size, nmemb, block, cmp);
	}
}
#endif
