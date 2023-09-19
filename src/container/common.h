#ifndef CRYPTANALYSISLIB_CONTAINER_COMMON_H
#define CRYPTANALYSISLIB_CONTAINER_COMMON_H

/// Concept for base data type.
/// \tparam T
template<typename T>
concept kAryContainerAble =
requires(T t) {
	t ^ t;
	T(0);
};
#endif//CRYPTANALYSISLIB_CONTAINER_COMMON_H
