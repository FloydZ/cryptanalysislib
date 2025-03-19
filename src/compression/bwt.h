#pragma once 
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// source  https://github.com/felipelouza/bwt-lcp-in-place
#define END_MARKER '$'


int rank(uint8_t *T, char c, int i){
	int sum=0;
	int j;
	for(j=0; j<i; j++) if(T[j] <= c) sum++;
	for(; j<(int)strlen((char *)T); j++) if(T[j] < c) sum++;

    return sum;
}

int bwt_lcp_inplace(uint8_t *T, int n, int *LCP){

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
		p_b1=p+s+1;
		l_b=LCP[p_b1];
		while(T[p_b1]!=T[s] && p_b1<n) //RMQ function
			if(LCP[++p_b1]<l_b) l_b=LCP[p_b1];
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


/// \param T[in]:
int bwt_inplace(uint8_t *T, int n){

	int p, r=1;
	int i, s;

	for(s=n-3; s>=0; s--){

		p = r+1;//	p = find_sentinel(&T[s]);
		r = rank(&T[s+1], T[s], p);
	
		T[p+s] = T[s]; //replace('$', T[s]);

		for(i=s; i<s+r; i++) 
			T[i] = T[i+1];

		T[s+r] = END_MARKER;
	}

    return 0;
}

uint8_t* bwt_reverse(uint8_t *bwt, int n){

	auto* rev = (uint8_t*) malloc((n+1)*sizeof(char));
	int p = 0;

	//backward reconstruction
	int s;
	for(s=n-2; s>=0; s--){

		rev[s] = bwt[p];
		p = rank(bwt, rev[s], p);
	}

	rev[n-1] = END_MARKER;
	rev[n] = '\0';

    return rev;
}
