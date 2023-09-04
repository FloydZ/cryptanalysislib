#ifndef CRYPTANALYSISLIB_LIB_ENUMERATION_H
#define CRYPTANALYSISLIB_LIB_ENUMERATION_H
#include <cstddef>

/// list enumeration type
enum ListIteration {
	SingleFullLength,       // Only one symbol on full length
	MultiFullLength,        // Two Symbols on the full length
	SinglePartialSingle,    // [xxxx0000|xx000000], [0000xxxx|00xx0000], [xxxx0000|0000xx00], [0000xxxx|000000xx]
	EnumSinglePartialSingle,// [xxxx0000|xx000000], [0000xxxx|00xx0000], [xxxx0000|0000xx00], [0000xxxx|000000xx] for all p=1, .., nr1
	MultiDisjointBlock,     // [xyxyxyxy|xy000000], [xyxyxyxy|00xy0000], [xyxyxyxy|0000xy00], [xyxyxyxy|000000xy]
	MITMSingle,             // [xxxxxxxx|00000000] and [xxxxxxxx|00000000]
	MITMMulti,              // [xyxyxyxy|00000000] and [00000000|xyxyxyxy]
	MITMEnumSingle,         // same as MITMSingle, but enumerating every weight
};


#if __cplusplus > 201709L
///
/// \tparam List
template<class List>
concept ListEnumeration_ListAble = requires(List c) {
	/// should return the List
	c.size() -> size_t;
	c.clear();
};

///
template<class List>
concept ListEnumeration_ChangeListAble = requires(List c) {
	/// should return the ChangeList
	c.size() -> size_t;
	c.clear();
};
#endif

#include "list/enumeration/fq.h"
// TODO #include "list/enumeration/ternary.h"

#endif//CRYPTANALYSISLIB_ENUMERATION_H
