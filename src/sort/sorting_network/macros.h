#ifndef CRYPTANALYSISLIB_SORT_SORTINGNETWORK_MACROS_H
#define CRYPTANALYSISLIB_SORT_SORTINGNETWORK_MACROS_H

/// creates a funtions which takes two `REG` variables, each of them storing
/// a single `MULT` type, and sorts them to a single `NEW_MULT`
/// \NEW_MULT: ex: f32x16
/// \MULT: ex: f32x8
/// \REG: ex: __m256
#define sortingnetwork_aftermerge2(NEW_MULT, MULT, REG)						\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b) { \
	REG tmp; 																\
    COEX_ ## MULT(a, b, tmp);                          						\
	a = sortingnetwork_aftermerge_ ## MULT (a); 							\
	b = sortingnetwork_aftermerge_ ## MULT (b); 							\
}


/// REG swap = (REG)_mm256_permute2f128_ps((__m256)b, (__m256)b, _MM_SHUFFLE(0, 0, 1, 1));
/// REG perm_neigh = (REG)_mm256_permute_ps((__m256)swap, _MM_SHUFFLE(0, 1, 2, 3));
/// \param a = [a0, a1, a2, a3]
/// \param b = [a0, a1, a2, a3]
#define sortingnetwork_permute_minmax2(NEW_MULT, REG, MIN_FKT, MAX_FKT )			\
static inline void sortingnetwork_permute_minmax_ ## NEW_MULT (REG &a, REG &b) noexcept { \
  	const REG swap = (REG) (__m256) __builtin_ia32_vperm2f128_ps256 ((__v8sf)b, (__v8sf)b, _MM_SHUFFLE(0, 0, 1, 1)); \
  	REG perm_neigh = (REG) (__m256) __builtin_ia32_vpermilps256 ((__v8sf)swap, _MM_SHUFFLE(0, 1, 2, 3)); 	\
	REG perm_neigh_min = MIN_FKT(a, perm_neigh);									\
	b = MAX_FKT(a, perm_neigh);														\
	a = perm_neigh_min;																\
}

#define sortingnetwork_merge_sorted2(NEW_MULT,MULT1,REG)	\
static inline void sortingnetwork_merge_sorted_ ## NEW_MULT (REG &a, REG &b) noexcept {	\
	sortingnetwork_permute_minmax_ ## NEW_MULT (a, b); 		\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
}

#define sortingnetwork_sort2(NEW_MULT,MULT1,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b) noexcept { 	\
	a = sortingnetwork_sort_ ## MULT1 (a); 				\
	b = sortingnetwork_sort_ ## MULT1 (b); 				\
	sortingnetwork_merge_sorted_ ## NEW_MULT(a, b); 	\
}


#define sortingnetwork_merge_sorted3(NEW_MULT,MULT2,MULT1,REG)	\
static inline void sortingnetwork_merge_sorted_ ## NEW_MULT (REG &a, REG &b, REG &c) noexcept {\
 	REG tmp;                                                    \
	sortingnetwork_permute_minmax_ ## MULT2(b, c); 				\
	COEX_ ## MULT1(a, b, tmp);									\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 				\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 				\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 				\
}

#define sortingnetwork_aftermerge_sorted3(NEW_MULT,MULT1,REG)	\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT (REG &a, REG &b, REG &c) { \
	REG tmp;                                                    \
	COEX_ ## MULT1 (a, c, tmp); 								\
	COEX_ ## MULT1 (a, b, tmp); 								\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 				\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 				\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 				\
}


#define sortingnetwork_sort3(NEW_MULT,DOUBLE_MULT,MULT1,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c) noexcept {\
	sortingnetwork_sort_ ## DOUBLE_MULT(a, b); 					\
	c = sortingnetwork_sort_ ## MULT1 (c); 						\
	sortingnetwork_merge_sorted_ ## NEW_MULT (a, b, c);			\
}


#define sortingnetwork_merge_sorted4(NEW_MULT,DOUBLE_MULT,SINGLE_MULT,REG)	\
static inline void sortingnetwork_merge_sorted ## NEW_MULT (REG &a, REG &b, REG &c, REG &d) noexcept {\
	REG tmp;                                                \
	sortingnetwork_permute_minmax_ ## DOUBLE_MULT (a, d); 	\
	sortingnetwork_permute_minmax_ ## DOUBLE_MULT (b, c); 	\
	COEX_ ## SINGLE_MULT(a, b, tmp); 						\
	COEX_ ## SINGLE_MULT(c, d, tmp); 						\
	a = sortingnetwork_aftermerge_ ## SINGLE_MULT(a); 		\
	b = sortingnetwork_aftermerge_ ## SINGLE_MULT(b); 		\
	c = sortingnetwork_aftermerge_ ## SINGLE_MULT(c); 		\
	d = sortingnetwork_aftermerge_ ## SINGLE_MULT(d); 		\
}

#define sortingnetwork_aftermerge_sorted4(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d) {\
	REG tmp;                                                \
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1 (d); 			\
}

#define sortingnetwork_aftermerge_sorted5(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e) { \
	REG tmp;                                                \
	COEX_ ## MULT1 (a, e, tmp); 							\
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1 (d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1 (e); 			\
}

#define sortingnetwork_aftermerge_sorted6(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f){\
	REG tmp;                                                \
	COEX_ ## MULT1 (a, e, tmp); 							\
	COEX_ ## MULT1 (b, f, tmp); 							\
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	COEX_ ## MULT1 (e, f, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1 (d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1 (e); 			\
	f = sortingnetwork_aftermerge_ ## MULT1 (f); 			\
}

#define sortingnetwork_aftermerge_sorted7(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG& e, REG& f, REG &g) { \
	REG tmp;                                                \
	COEX_ ## MULT1 (a, e, tmp); 							\
	COEX_ ## MULT1 (b, f, tmp); 							\
	COEX_ ## MULT1 (c, g, tmp); 							\
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	COEX_ ## MULT1 (e, g, tmp); 							\
	COEX_ ## MULT1 (e, f, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1(a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1(b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1(c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1(d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1(e); 			\
	f = sortingnetwork_aftermerge_ ## MULT1(f); 			\
	g = sortingnetwork_aftermerge_ ## MULT1(g); 			\
}



#define sortingnetwork_sort4(NEW_MULT,DOUBLE_MULT,SINGLE_MULT,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d) noexcept { \
	sortingnetwork_sort_ ## DOUBLE_MULT(a, b); 				\
	sortingnetwork_sort_ ## DOUBLE_MULT(c, d); 				\
	sortingnetwork_merge_sorted ## NEW_MULT (a, b, c, d); 	\
}

#define sortingnetwork_sort5(NEW_MULT,MULT4,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e) { \
	REG tmp;                                                \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 				\
	e = sortingnetwork_sort_ ## MULT1(e); 					\
	sortingnetwork_permute_minmax_ ## MULT2(d, e);			\
	COEX_ ## MULT1(a, c, tmp); 								\
	COEX_ ## MULT1(b, d, tmp); 								\
	COEX_ ## MULT1(a, b, tmp); 								\
	COEX_ ## MULT1(c, d, tmp); 								\
	a = sortingnetwork_aftermerge_ ## MULT1(a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1(b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1(c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1(d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1(e); 			\
}

#define sortingnetwork_sort6(NEW_MULT,MULT4,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f){ \
	REG tmp;                                        \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 		\
	sortingnetwork_sort_ ## MULT2(e, f);			\
	sortingnetwork_permute_minmax_ ## MULT2(c, f); 	\
	sortingnetwork_permute_minmax_ ## MULT2(d, e); 	\
	COEX_ ## MULT1(a, c, tmp); 						\
	COEX_ ## MULT1(b, d, tmp); 						\
	COEX_ ## MULT1(a, b, tmp); 						\
	COEX_ ## MULT1(c, d, tmp); 						\
	COEX_ ## MULT1(e, f, tmp); 						\
	a = sortingnetwork_aftermerge_ ## MULT1(a);		\
	b = sortingnetwork_aftermerge_ ## MULT1(b);		\
	c = sortingnetwork_aftermerge_ ## MULT1(c);		\
	d = sortingnetwork_aftermerge_ ## MULT1(d);		\
	e = sortingnetwork_aftermerge_ ## MULT1(e);		\
	f = sortingnetwork_aftermerge_ ## MULT1(f);		\
}

#define sortingnetwork_sort7(NEW_MULT,MULT4,MULT3,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g){ \
	REG tmp;                                        \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 		\
	sortingnetwork_sort_ ## MULT3(e, f, g); 		\
	sortingnetwork_permute_minmax_ ## MULT2(c, f); 	\
	sortingnetwork_permute_minmax_ ## MULT2(d, e); 	\
	sortingnetwork_permute_minmax_ ## MULT2(b, g); 	\
	COEX_ ## MULT1(a, c, tmp); 						\
	COEX_ ## MULT1(b, d, tmp); 						\
	COEX_ ## MULT1(a, b, tmp); 						\
	COEX_ ## MULT1(c, d, tmp); 						\
	COEX_ ## MULT1(e, g, tmp); 						\
	COEX_ ## MULT1(e, f, tmp); 						\
	a = sortingnetwork_aftermerge_ ## MULT1(a);		\
	b = sortingnetwork_aftermerge_ ## MULT1(b);		\
	c = sortingnetwork_aftermerge_ ## MULT1(c);		\
	d = sortingnetwork_aftermerge_ ## MULT1(d);		\
	e = sortingnetwork_aftermerge_ ## MULT1(e);		\
	f = sortingnetwork_aftermerge_ ## MULT1(f);		\
	g = sortingnetwork_aftermerge_ ## MULT1(g);		\
}

#define sortingnetwork_aftermerge8(NEW_MULT,MULT2,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h){ \
	REG tmp;                                    \
	COEX_ ## MULT1(a, e, tmp); 					\
	COEX_ ## MULT1(b, f, tmp); 					\
	COEX_ ## MULT1(c, g, tmp); 					\
	COEX_ ## MULT1(d, h, tmp); 					\
	COEX_ ## MULT1(a, c, tmp); 					\
	COEX_ ## MULT1(b, d, tmp); 					\
	COEX_ ## MULT1(a, b, tmp); 					\
	COEX_ ## MULT1(c, d, tmp); 					\
	COEX_ ## MULT1(e, g, tmp); 					\
	COEX_ ## MULT1(f, h, tmp); 					\
	COEX_ ## MULT1(e, f, tmp); 					\
	COEX_ ## MULT1(g, h, tmp); 					\
	a = sortingnetwork_aftermerge_ ## MULT1(a); \
	b = sortingnetwork_aftermerge_ ## MULT1(b); \
	c = sortingnetwork_aftermerge_ ## MULT1(c); \
	d = sortingnetwork_aftermerge_ ## MULT1(d); \
	e = sortingnetwork_aftermerge_ ## MULT1(e); \
	f = sortingnetwork_aftermerge_ ## MULT1(f); \
	g = sortingnetwork_aftermerge_ ## MULT1(g); \
	h = sortingnetwork_aftermerge_ ## MULT1(h); \
}

#define sortingnetwork_sort8(NEW_MULT,MULT4,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h) { \
	REG tmp;                                        \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 		\
	sortingnetwork_sort_ ## MULT4(e, f, g, h); 		\
	sortingnetwork_permute_minmax_ ## MULT2(a, h); 	\
	sortingnetwork_permute_minmax_ ## MULT2(b, g); 	\
	sortingnetwork_permute_minmax_ ## MULT2(c, f); 	\
	sortingnetwork_permute_minmax_ ## MULT2(d, e); 	\
	COEX_ ## MULT1(a, c, tmp); 						\
	COEX_ ## MULT1(b, d, tmp); 						\
	COEX_ ## MULT1(a, b, tmp); 						\
	COEX_ ## MULT1(c, d, tmp); 						\
	COEX_ ## MULT1(e, g, tmp); 						\
	COEX_ ## MULT1(f, h, tmp); 						\
	COEX_ ## MULT1(e, f, tmp); 						\
	COEX_ ## MULT1(g, h, tmp); 						\
	a = sortingnetwork_aftermerge_ ## MULT1(a);		\
	b = sortingnetwork_aftermerge_ ## MULT1(b);		\
	c = sortingnetwork_aftermerge_ ## MULT1(c);		\
	d = sortingnetwork_aftermerge_ ## MULT1(d);		\
	e = sortingnetwork_aftermerge_ ## MULT1(e);		\
	f = sortingnetwork_aftermerge_ ## MULT1(f);		\
	g = sortingnetwork_aftermerge_ ## MULT1(g);		\
	h = sortingnetwork_aftermerge_ ## MULT1(h);		\
}

#define sortingnetwork_sort9(NEW_MULT,MULT8,MULT2,MULT1,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i) noexcept { \
	sortingnetwork_sort_ ## MULT8 (a, b, c, d, e, f, g, h); 		\
	i = sortingnetwork_sort_ ## MULT1(i); 							\
	sortingnetwork_permute_minmax_ ## MULT2(h, i);					\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);	\
	i = sortingnetwork_aftermerge_ ## MULT1(i); 					\
}

#define sortingnetwork_sort10(NEW_MULT,MULT8,MULT2,REG)				\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j) { \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 			\
	sortingnetwork_sort_ ## MULT2(i, j); 							\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 					\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 					\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h); 	\
	sortingnetwork_aftermerge_ ## MULT2(i, j); 						\
}

#define sortingnetwork_sort11(NEW_MULT,MULT8,MULT3,MULT2,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k) { \
	sortingnetwork_sort_ ## MULT8 (a, b, c, d, e, f, g, h); 		\
	sortingnetwork_sort_ ## MULT3 (i, j, k); 						\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 					\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 					\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 					\
	sortingnetwork_aftermerge_ ## MULT8 (a, b, c, d, e, f, g, h); 	\
	sortingnetwork_aftermerge_ ## MULT3 (i, j, k); 					\
}

#define sortingnetwork_sort12(NEW_MULT,MULT8,MULT4,MULT2,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG   &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l) { \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 			\
	sortingnetwork_sort_ ## MULT4(i, j, k, l); 						\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 					\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 					\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 					\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 					\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h); 	\
	sortingnetwork_aftermerge_ ## MULT4(i, j, k, l); 				\
}

#define sortingnetwork_sort13(NEW_MULT,MULT8,MULT5,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m) { \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 		\
	sortingnetwork_sort_ ## MULT5(i, j, k, l, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(d, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 				\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 				\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 				\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 				\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);\
	sortingnetwork_aftermerge_ ## MULT5(i, j, k, l, m); 		\
}

#define sortingnetwork_sort14(NEW_MULT,MULT8,MULT6,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m, REG &n) { \
	sortingnetwork_sort_ ## MULT8 (a, b, c, d, e, f, g, h); 	\
	sortingnetwork_sort_ ## MULT6 (i, j, k, l, m, n); 			\
	sortingnetwork_permute_minmax_ ## MULT2(c, n); 				\
	sortingnetwork_permute_minmax_ ## MULT2(d, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 				\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 				\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 				\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 				\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);\
	sortingnetwork_aftermerge_ ## MULT6 (i, j, k, l, m, n); 	\
}

#define sortingnetwork_sort15(NEW_MULT,MULT8,MULT7,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m, REG &n, REG &o){ \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 		\
	sortingnetwork_sort_ ## MULT7(i, j, k, l, m, n, o); 		\
	sortingnetwork_permute_minmax_ ## MULT2(b, o); 				\
	sortingnetwork_permute_minmax_ ## MULT2(c, n); 				\
	sortingnetwork_permute_minmax_ ## MULT2(d, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 				\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 				\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 				\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 				\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);\
	sortingnetwork_aftermerge_ ## MULT7(i, j, k, l, m, n, o);	\
}

#define sortingnetwork_sort16(NEW_MULT,MULT8,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m, REG &n, REG &o, REG &p){ \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); \
	sortingnetwork_sort_ ## MULT8(i, j, k, l, m, n, o, p); \
	sortingnetwork_permute_minmax_ ## MULT2(a, p); \
	sortingnetwork_permute_minmax_ ## MULT2(b, o); \
	sortingnetwork_permute_minmax_ ## MULT2(c, n); \
	sortingnetwork_permute_minmax_ ## MULT2(d, m); \
	sortingnetwork_permute_minmax_ ## MULT2(e, l); \
	sortingnetwork_permute_minmax_ ## MULT2(f, k); \
	sortingnetwork_permute_minmax_ ## MULT2(g, j); \
	sortingnetwork_permute_minmax_ ## MULT2(h, i); \
	sortingnetwork_aftermerge_ ## MULT8 (a, b, c, d, e, f, g, h); \
	sortingnetwork_aftermerge_ ## MULT8 (i, j, k, l, m, n, o, p); \
}
#endif
