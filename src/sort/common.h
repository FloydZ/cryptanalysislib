#ifndef CRYPTANALYSISLIB_COMMON_H
#define CRYPTANALYSISLIB_COMMON_H

#include <cstdlib>

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
	T pivot = head++;
	int swap;

	while (true) {
		while (array[head] <= array[pivot] && head < tail) {
			head++;
		}

		while (array[tail] > array[pivot]) {
			tail--;
		}

		if (head >= tail) {
			swap = array[pivot]; array[pivot] = array[tail]; array[tail] = swap;

			return tail;
		}

		swap = array[head]; array[head] = array[tail]; array[tail] = swap;
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
