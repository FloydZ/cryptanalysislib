#ifndef CRYPTANALYSISLIB_HASHMAP_COMMON_H
#define CRYPTANALYSISLIB_HASHMAP_COMMON_H

#if __cplusplus > 201709L

#include <cstdint>
template<typename HashMap>
concept HashMapAble = requires(HashMap hm) {
	/// needed typedefs
	typename HashMap::T;
	typename HashMap::LoadType;
	typename HashMap::IndexType;

	requires requires(const typename HashMap::T d,
	                  const typename HashMap::T &dd,
					  const typename HashMap::T *ddd,
	                  // NOTE: cannot be const, as this values must be
	                  // non const.
	                  typename HashMap::LoadType l,
					  typename HashMap::LoadType &ll,
					  typename HashMap::LoadType *lll,
	                  const typename HashMap::IndexType i,
					  const typename HashMap::IndexType &ii,
					  const typename HashMap::IndexType *iii,
                      const uint32_t tid) {
		hm.hash(d);
		hm.insert(d, iii, tid);

		// NOTE: this seems to be an internal compiler bug. It
		// seems that you cannot name function `find()`.
		hm._find(d, ll);
	};
};
#endif

#endif//CRYPTANALYSISLIB_HASHMAP_COMMON_H
