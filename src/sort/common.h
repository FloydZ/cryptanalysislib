#ifndef CRYPTANALYSISLIB_COMMON_H
#define CRYPTANALYSISLIB_COMMON_H

#include <cstdlib>

#include "algorithm/swap.h"

/// src: https://github.com/scandum/crumsort/tree/main
/// \tparam T
/// \param array
/// \param head
/// \param tail
/// \return
template<typename T>
size_t hoare_partition(T array[],
                    size_t head,
                    size_t tail) {
	// NOTE: `pivot` is an index, not a value
	const size_t pivot = head++;

	while (true) {
		while (array[head] <= array[pivot] && head < tail) {
			head++;
		}

		while (array[tail] > array[pivot]) {
			tail--;
		}

		if (head >= tail) {
			cryptanalysislib::swap(array[pivot], array[tail]);
			return tail;
		}

		cryptanalysislib::swap(array[head], array[tail]);
	}
}

/// SRC: https://github.com/scandum/crumsort/tree/main
/// \tparam T
/// \param array
/// \param head
/// \param tail
/// \return
template<typename T>
size_t fulcrum_partition(T array[],
                         size_t head,
                         size_t tail) {
	T pivot = array[head];

	while (true) {
		if (array[tail] > pivot)
		{
			tail--;
			continue;
		}

		if (head >= tail)
		{
			array[head] = pivot;
			return head;
		}
		array[head++] = array[tail];

		while (true)
		{
			if (head >= tail)
			{
				array[head] = pivot;
				return head;
			}

			if (array[head] <= pivot)
			{
				head++;
				continue;
			}
			array[tail--] = array[head];
			break;
		}
	}
}
#endif//CRYPTANALYSISLIB_COMMON_H
