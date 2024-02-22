#ifndef CRYPTANALYSISLIB_HASHMAP_COMMON_H
#define CRYPTANALYSISLIB_HASHMAP_COMMON_H

#if __cplusplus > 201709L

#include <cstdint>
template<typename HashMap>
concept HashMapAble = requires(HashMap hm) {
	/// needed typedefs
	typename HashMap::data_type;
	typename HashMap::key_type;
	typename HashMap::index_type;
	typename HashMap::load_type;

	requires requires(const typename HashMap::key_type k,
	                  const typename HashMap::key_type &kk,
					  const typename HashMap::key_type *kkk,
	                  // NOTE: cannot be const, as this values must be
	                  // non const.
	                  typename HashMap::load_type l,
					  typename HashMap::load_type &ll,
					  typename HashMap::load_type *lll,
					  const typename HashMap::index_type i,
					  const typename HashMap::index_type &ii,
					  const typename HashMap::index_type *iii,
	                  const typename HashMap::data_type d,
					  const typename HashMap::data_type &dd,
					  const typename HashMap::data_type *ddd,
                      const uint32_t tid) {
		hm.ptr();
		hm.ptr(i);

		hm.hash(k);
		hm.insert(k, dd, tid);
		hm.insert(k, dd);

		// NOTE: this seems to be an internal compiler bug. It
		// seems that you cannot name function `find()`.
		hm.find(k, ll);
	};
};
#endif

#endif//CRYPTANALYSISLIB_HASHMAP_COMMON_H
