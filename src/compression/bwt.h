#pragma once 
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// source  https://github.com/felipelouza/bwt-lcp-in-place
#define END_MARKER '$'

/// NOTE: the input text T[0..n) must end with END_MARKER, which must be
/// 	unique and smaller than every other symbol of the text.
/// TODO docs tests and benchs

/// \param T[in]: text
/// \param n[in]: length of T
/// \param c[in]: symbol
/// \param i[in]: position
/// \return number of symbols < c in T, plus the number of c's before i
static inline
int rank(const uint8_t *T, const int n, const uint8_t c, const int i) noexcept {
	int sum=0;
	int j;
	for(j=0; j<i; j++) if(T[j] <= c) sum++;
	for(; j<n; j++) if(T[j] < c) sum++;

    return sum;
}

inline int bwt_lcp_inplace(uint8_t *T, int n, int *LCP) noexcept {

	int i, p, r=1, s;
	int p_a1, p_b1, l_a, l_b;
	LCP[n-1] = LCP[n-2] = 0;//base case

	for(s=n-3; s>=0; s--){
	
		/*steps 1 and 2*/
		p=r+1;
		for(i=s+1, r=0; T[i]!=END_MARKER; i++)if(T[i]<=T[s])r++;
		for(; i<n; i++) if(T[i]<T[s])r++;
		
		/*steps 2'*/
		p_a1=p+s-1;
		l_a=LCP[p_a1+1];
		while(T[p_a1]!=T[s])//RMQ function
			if(LCP[p_a1--]<l_a) l_a=LCP[p_a1+1];
		if(p_a1==s) l_a=0;
		else l_a++;
		
		/*steps 2''*/
		// NOTE: `p_b1` may reach `n`. Before, `T[n]` was read (the bound was
		// 	checked second) and `LCP[n]`; `l_b` is set to 0 in this case anyway.
		p_b1=p+s+1;
		l_b = (p_b1<n) ? LCP[p_b1] : 0;
		while(p_b1<n && T[p_b1]!=T[s]) { //RMQ function
			++p_b1;
			if(p_b1<n && LCP[p_b1]<l_b) l_b=LCP[p_b1];
		}
		if(p_b1==n) l_b=0;
		else l_b++;
		
		/*steps 3 and 4*/
		T[p+s]=T[s];
		for(i=s; i<s+r; i++){
			T[i]=T[i+1];
			LCP[i]=LCP[i+1];
		}
		T[s+r]=END_MARKER;
		
		/*steps 4'*/
		LCP[s+r]=l_a;
		if(s+r+1<n)//If r+1 is not the last position
			LCP[s+r+1]=l_b;
	}

    return 0;
}


/// \param T[in/out]: text of length n, ending with END_MARKER. Replaced by its BWT.
/// \param n[in]: length of T
/// \return n, the length of the BWT
inline int bwt_inplace(uint8_t *T, int n) noexcept {

	int p, r=1;
	int i, s;

	for(s=n-3; s>=0; s--){

		p = r+1;//	p = find_sentinel(&T[s]);
		r = rank(&T[s+1], n-s-1, T[s], p);
	
		T[p+s] = T[s]; //replace('$', T[s]);

		for(i=s; i<s+r; i++) 
			T[i] = T[i+1];

		T[s+r] = END_MARKER;
	}

    return n;
}

/// \param bwt[in]: BWT of length n (as computed by `bwt_inplace`)
/// \param n[in]: length of the BWT, must be > 0
/// \return the original text (n symbols + '\0'), must be freed by the caller
inline uint8_t* bwt_reverse(uint8_t *bwt, int n) noexcept {

	auto* rev = (uint8_t*) malloc((n+1)*sizeof(char));
	int p = 0;

	//backward reconstruction
	int s;
	for(s=n-2; s>=0; s--){

		rev[s] = bwt[p];
		p = rank(bwt, n, rev[s], p);
	}

	rev[n-1] = END_MARKER;
	rev[n] = '\0';

    return rev;
}
