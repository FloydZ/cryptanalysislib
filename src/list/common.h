#ifndef DECODING_LIST_COMMON_H
#define DECODING_LIST_COMMON_H

#include <cstdint>
#include "element.h"

#if __cplusplus > 201709L

/// This concept enforces the needed
/// 	- functions
/// 	- typedefs
/// 	- elements
/// each list needs
/// \tparam Element
template<class Element>
concept ListElementAble = requires(Element a) {
	typename Element::ValueType;
	typename Element::LabelType;
	typename Element::MatrixType;

	typename Element::ValueContainerType;
	typename Element::LabelContainerType;

	typename Element::ValueDataType;
	typename Element::LabelDataType;

	requires ElementAble<typename Element::ValueType,
						 typename Element::LabelType,
						 typename Element::MatrixType
	                     >;

	// function requirements
	// TODO be more precise:
	requires requires(const uint32_t a1, const uint32_t a2) { a.print(a1, a2); };
	requires requires() { a.zero(); a.get_label(); a.get_value(); };
	requires requires(const uint32_t b1, const uint32_t b2) { Element::add(a, a, a, b1, b1, b2); };
};


/// this concept enforces the needed
/// 	- functions
/// 	- typdefs
/// a list must implement
/// \tparam List
template<class List>
concept ListAble = requires(List l) {
	/// size stuff
	requires requires(const uint32_t i) {
		/// returns the size
		l.size();
		/// returns the size each thread needs to enumerate
		l.size(i);

		/// start/end pos of each block for each thread
		l.start_pos(i);
		l.end_pos(i);

		l.get_load();
		l.set_load(i);
	};

	/// access functions
	requires requires(const uint32_t i) {
		l.data_value();
		l.data_label();
		l.data_value(i);
		l.data_label(i);
		l[i];
		l.at(i);
	};

	/// printing stuff
	requires requires(const uint32_t i) {
		/// print single elements
		l.print_binary(i,i,i,i,i);
		l.print(i,i,i,i,i);

		/// print parts of the list
		l.print_binary(i,i,i,i,i,i);
		l.print(i,i,i,i,i,i);
	};

	/// arithmetic/algorithm stuff
	requires requires (const uint32_t i) {
		l.sort();
		/// i = thread id
		l.zero(i);

		l.bytes();
	};
};
#endif
#endif //DECODING_LIST_COMMON_H
