// Created by Floyd on 05.05.22.
// src: https://github.com/scandum/crumsort/blob/main/src/crumsort.h

#ifndef SMALLSECRETLWE_CRUMSORT_H
#define SMALLSECRETLWE_CRUMSORT_H
/*
	Copyright (C) 2014-2022 Igor van den Hoven ivdhoven@gmail.com

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

	crumsort 1.1.5.2
*/

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>

#include "quadsort.h"

#define CRUM_AUX 512
#define CRUM_OUT  24

template<typename T, typename CMP>
size_t crum_analyze(T *array, T *swap, size_t swap_size, size_t nmemb, CMP *cmp)
{
	char loop, dist;
	size_t cnt, balance = 0, streaks = 0;
	T *pta, *ptb, tmp;

	pta = array;

	for (cnt = nmemb ; cnt > 16 ; cnt -= 16)
	{
		for (dist = 0, loop = 16 ; loop ; loop--)
		{
			dist += cmp(pta, pta + 1) > 0; pta++;
		}
		streaks += (dist == 0) | (dist == 16);
		balance += dist;
	}

	while (--cnt)
	{
		balance += cmp(pta, pta + 1) > 0;
		pta++;
	}

	if (balance == 0)
	{
		return 1;
	}

	if (balance == nmemb - 1)
	{
		pta = array;
		ptb = array + nmemb;

		cnt = nmemb / 2;

		do
		{
			tmp = *pta; *pta++ = *--ptb; *ptb = tmp;
		}
		while (--cnt);

		return 1;
	}

	if (streaks >= nmemb / 24) {
//	if (streaks >= nmemb / 32){
		quadsort_swap(array, swap, swap_size, nmemb, cmp);

		return 1;
	}
	return 0;
}

// The next 3 functions are used for pivot selection
template<typename T, typename CMP>
T *crum_median_of_sqrt(T *array, T *swap, size_t swap_size, size_t nmemb, CMP *cmp){
	T *pta, *piv;
	size_t cnt, sqrt, div;

	sqrt = nmemb < 65536 ? 16 : nmemb < 262144 ? 128 : 256;

	div = nmemb / sqrt;

	pta = array + nmemb - 1;
	piv = array + sqrt;

	for (cnt = sqrt ; cnt ; cnt--){
		swap[0] = *--piv; *piv = *pta; *pta = swap[0];

		pta -= div;
	}

	quadsort_swap(piv, swap, swap_size, sqrt, cmp);

	return piv + sqrt / 2;
}

template<typename T, typename CMP>
size_t crum_median_of_three(T *array, size_t v0, size_t v1, size_t v2, CMP *cmp) {
	size_t v[3] = {v0, v1, v2};
	char x, y, z;

	x = cmp(array + v0, array + v1) > 0;
	y = cmp(array + v0, array + v2) > 0;
	z = cmp(array + v1, array + v2) > 0;

	return v[(x == y) + (y ^ z)];
}

template<typename T, typename CMP>
T *crum_median_of_nine(T *array, size_t nmemb, CMP *cmp) {
	size_t x, y, z, div = nmemb / 16;

	x = crum_median_of_three(array, div * 2, div * 1, div * 4, cmp);
	y = crum_median_of_three(array, div * 8, div * 6, div * 10, cmp);
	z = crum_median_of_three(array, div * 14, div * 12, div * 15, cmp);

	return array + crum_median_of_three(array, x, y, z, cmp);
}

// As per suggestion by Marshall Lochbaum to improve generic data handling
template<typename T, typename CMP>
size_t fulcrum_reverse_partition(T *array, T *swap, T *ptx, T *piv, size_t swap_size, size_t nmemb, CMP *cmp) {
	size_t cnt, val, i, m = 0;
	T *ptl, *ptr, *pta, *tpa;

	if (nmemb <= swap_size)
	{
		cnt = nmemb / 8;

		do for (i = 8 ; i ; i--)
			{
				val = cmp(piv, ptx) > 0; swap[-m] = array[m] = *ptx++; m += val; swap++;
			}
		while (--cnt);

		for (cnt = nmemb % 8 ; cnt ; cnt--)
		{
			val = cmp(piv, ptx) > 0; swap[-m] = array[m] = *ptx++; m += val; swap++;
		}
		memcpy(array + m, swap - nmemb, (nmemb - m) * sizeof(T));

		return m;
	}

	memcpy(swap, array, 16 * sizeof(T));
	memcpy(swap + 16, array + nmemb - 16, 16 * sizeof(T));

	ptl = array;
	ptr = array + nmemb - 1;

	pta = array + 16;
	tpa = array + nmemb - 17;

	cnt = nmemb / 16 - 2;

	while (1){
		if (pta - ptl - m <= 16) {
			if (cnt-- == 0)
				break;

			for (i = 16 ; i ; i--) {
				val = cmp(piv, pta) > 0; ptl[m] = ptr[m] = *pta++; m += val; ptr--;
			}
		}
		if (pta - ptl - m > 16) {
			if (cnt-- == 0)
				break;

			for (i = 16 ; i ; i--) {
				val = cmp(piv, tpa) > 0; ptl[m] = ptr[m] = *tpa--; m += val; ptr--;
			}
		}
	}

	if (pta - ptl - m <= 16) {
		for (cnt = nmemb % 16 ; cnt ; cnt--) {
			val = cmp(piv, pta) > 0; ptl[m] = ptr[m] = *pta++; m += val; ptr--;
		}
	}
	else {
		for (cnt = nmemb % 16 ; cnt ; cnt--) {
			val = cmp(piv, tpa) > 0; ptl[m] = ptr[m] = *tpa--; m += val; ptr--;
		}
	}
	pta = swap;

	for (cnt = 32 ; cnt ; cnt--)
	{
		val = cmp(piv, pta) > 0; ptl[m] = ptr[m] = *pta++; m += val; ptr--;
	}
	return m;
}

template<typename T, typename CMP>
size_t fulcrum_default_partition(T *array, T *swap, T *ptx, T *piv, size_t swap_size, size_t nmemb, CMP *cmp) {
	size_t cnt, val, i, m = 0;
	T *ptl, *ptr, *pta, *tpa;

	if (nmemb <= swap_size) {
		cnt = nmemb / 8;

		do for (i = 8 ; i ; i--){
			val = cmp(ptx, piv) <= 0; swap[-m] = array[m] = *ptx++; m += val; swap++;
		} while (--cnt);

		for (cnt = nmemb % 8 ; cnt ; cnt--) {
			val = cmp(ptx, piv) <= 0; swap[-m] = array[m] = *ptx++; m += val; swap++;
		}
		memcpy(array + m, swap - nmemb, sizeof(T) * (nmemb - m));

		return m;
	}

	memcpy(swap, array, 16 * sizeof(T));
	memcpy(swap + 16, array + nmemb - 16, 16 * sizeof(T));

	ptl = array;
	ptr = array + nmemb - 1;

	pta = array + 16;
	tpa = array + nmemb - 17;

	cnt = nmemb / 16 - 2;

	while (1) {
		if (pta - ptl - m <= 16) {
			if (cnt-- == 0)
				break;

			for (i = 16 ; i ; i--) {
				val = cmp(pta, piv) <= 0; ptl[m] = ptr[m] = *pta++; m += val; ptr--;
			}
		}
		if (pta - ptl - m > 16) {
			if (cnt-- == 0) break;

			for (i = 16 ; i ; i--) {
				val = cmp(tpa, piv) <= 0; ptl[m] = ptr[m] = *tpa--; m += val; ptr--;
			}
		}
	}

	if (pta - ptl - m <= 16) {
		for (cnt = nmemb % 16 ; cnt ; cnt--) {
			val = cmp(pta, piv) <= 0; ptl[m] = ptr[m] = *pta++; m += val; ptr--;
		}
	}
	else {
		for (cnt = nmemb % 16 ; cnt ; cnt--) {
			val = cmp(tpa, piv) <= 0; ptl[m] = ptr[m] = *tpa--; m += val; ptr--;
		}
	}
	pta = swap;

	for (cnt = 32 ; cnt ; cnt--) {
		val = cmp(pta, piv) <= 0; ptl[m] = ptr[m] = *pta++; m += val; ptr--;
	}
	return m;
}

template<typename T, typename CMP>
void fulcrum_partition(T *array, T *swap, T *max, size_t swap_size, size_t nmemb, CMP *cmp)
{
	size_t a_size, s_size;
	T *ptp, piv;

	while (1) {
		if (nmemb <= 2048) {
			ptp = crum_median_of_nine(array, nmemb, cmp);
		}
		else {
			ptp = crum_median_of_sqrt(array, swap, swap_size, nmemb, cmp);
		}
		piv = *ptp;

		if (max && cmp(max, &piv) <= 0) {
			a_size = fulcrum_reverse_partition(array, swap, array, &piv, swap_size, nmemb, cmp);
			s_size = nmemb - a_size;

			if (s_size <= a_size / 16 || a_size <= CRUM_OUT)
			{
				return quadsort_swap(array, swap, swap_size, a_size, cmp);
			}
			nmemb = a_size; max = NULL;
			continue;
		}
		*ptp = array[--nmemb];

		a_size = fulcrum_default_partition(array, swap, array, &piv, swap_size, nmemb, cmp);
		s_size = nmemb - a_size;

		ptp = array + a_size; array[nmemb] = *ptp; *ptp = piv;

		if (a_size <= s_size / 16 || s_size <= CRUM_OUT) {
			if (s_size == 0) {
				a_size = fulcrum_reverse_partition(array, swap, array, &piv, swap_size, a_size, cmp);
				s_size = nmemb - a_size;

				if (s_size <= a_size / 16 || a_size <= CRUM_OUT) {
					return quadsort_swap(array, swap, swap_size, a_size, cmp);
				}
				return fulcrum_partition(array, swap, max, swap_size, a_size, cmp);
			}

			quadsort_swap(ptp + 1, swap, swap_size, s_size, cmp);
		}
		else {
			fulcrum_partition(ptp + 1, swap, max, swap_size, s_size, cmp);
		}

		if (s_size <= a_size / 32 || a_size <= CRUM_OUT) {
			return quadsort_swap(array, swap, swap_size, a_size, cmp);
		}
		max = ptp;
		nmemb = a_size;
	}
}

template<typename T, typename CMP>
void crumsort(T *array, size_t nmemb, CMP *cmp) {
	if (nmemb < 32){
		return tail_swap(array, nmemb, cmp);
	}
#if CRUM_AUX
	size_t swap_size = CRUM_AUX;
#else
	size_t swap_size = 32;

	while (swap_size * swap_size <= nmemb) {
		swap_size *= 4;
	}
#endif

	T swap[swap_size];

	if (crum_analyze(array, swap, swap_size, nmemb, cmp) == 0) {
		fulcrum_partition(array, swap, NULL, swap_size, nmemb, cmp);
	}
}

template<typename T, typename CMP>
void crumsort_swap(T *array, T *swap, size_t swap_size, size_t nmemb, CMP *cmp) {
	if (nmemb < 32) {
		tail_swap(array, nmemb, cmp);
	}
	else if (crum_analyze(array, swap, swap_size, nmemb, cmp) == 0){
		fulcrum_partition(array, swap, NULL, swap_size, nmemb, cmp);
	}
}
//#define cmp(a,b) (*(a) > *(b))


///////////////////////////////////////////////////////////////////////////
//┌─────────────────────────────────────────────────────────────────────┐//
//│ ██████┐██████┐ ██┐   ██┐███┐  ███┐███████┐ ██████┐ ██████┐ ████████┐│//
//│██┌────┘██┌──██┐██│   ██│████┐████│██┌────┘██┌───██┐██┌──██┐└──██┌──┘│//
//│██│     ██████┌┘██│   ██│██┌███┌██│███████┐██│   ██│██████┌┘   ██│   │//
//│██│     ██┌──██┐██│   ██│██│└█┌┘██│└────██│██│   ██│██┌──██┐   ██│   │//
//│└██████┐██│  ██│└██████┌┘██│ └┘ ██│███████│└██████┌┘██│  ██│   ██│   │//
//│ └─────┘└─┘  └─┘ └─────┘ └─┘    └─┘└──────┘ └─────┘ └─┘  └─┘   └─┘   │//
//└─────────────────────────────────────────────────────────────────────┘//
//////////////////////////////////////////////////////////////////////////

void crumsort(void *array, size_t nmemb, size_t size, CMPFUNC *cmp) {
	if (nmemb < 2) {
		return;
	}

	switch (size) {
		case sizeof(char):
			return crumsort8(array, nmemb, cmp);

		case sizeof(short):
			return crumsort16(array, nmemb, cmp);

		case sizeof(int):
			return crumsort32(array, nmemb, cmp);

		case sizeof(long long):
			return crumsort64(array, nmemb, cmp);

		case sizeof(long double):
			return crumsort128(array, nmemb, cmp);

		default:
			return assert(size == sizeof(char) || size == sizeof(short) || size == sizeof(int) || size == sizeof(long long) || size == sizeof(long double));
	}
}

#endif //SMALLSECRETLWE_CRUMSORT_H
