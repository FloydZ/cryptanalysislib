#ifndef CRYPTANALYSISLIB_COMBINATION_ENUMERATOR_H
#define CRYPTANALYSISLIB_COMBINATION_ENUMERATOR_H

#include <concepts>

/// The common interface of all enumerators in `src/combination`:
/// 	- after the construction (or `first()`) the enumerator is at the first
/// 		element of its sequence
/// 	- `current()`: the current element
/// 	- `next()`: moves to the next element. Returns `false` (and does not
/// 		change the current element) if the current element is the last one.
/// 		Infinite sequences always return `true`.
/// 	- `previous()`: moves to the previous element. Returns `false` (and
/// 		does not change the current element) if the current element is the
/// 		first one. Enumerators, which cannot go back, `assert(false)` and
/// 		return `false`.
/// Usage:
/// \code
/// 	enumeration_colex<uint64_t, 10, 3> e;
/// 	do { use(e.current()); } while (e.next());
/// \endcode
/// Tests: `tests/combination/enumerator.cpp`
/// \tparam Enumerator
template<typename Enumerator>
concept EnumeratorAble = requires(Enumerator e, const Enumerator ce) {
	ce.current();
	{ e.next() } -> std::same_as<bool>;
	{ e.previous() } -> std::same_as<bool>;
};
#endif//CRYPTANALYSISLIB_COMBINATION_ENUMERATOR_H
