#ifndef CRYPTANALYSISLIB_COMMON_H
#define CRYPTANALYSISLIB_COMMON_H

template<typename HashMap>
concept HashMapAble = requires(HashMap hm) {
	/// needed typedefs
	typename HashMap::T;
	typename HashMap::LoadType;
	typename HashMap::IndexType;

	requires requires(const typename HashMap::T d,
	                  const typename HashMap::T &ddd,
	                  const typename HashMap::LoadType l,
					  const typename HashMap::LoadType *ll,
					  const typename HashMap::LoadType *lll,
	                  const typename HashMap::IndexType i,
					  const typename HashMap::IndexType *ii,
                      const uint32_t tid) {
		hm.hash(d);
		hm.insert(d, ii, tid);
		// TODO kp geth nicht hm.find(ddd, lll);
	};
};

#endif//CRYPTANALYSISLIB_COMMON_H
