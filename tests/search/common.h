#include <vector>
#include <algorithm>

#include "random.h"
#include "helper.h"

/// generates a random sorted list.
/// \tparam T base element type
/// \param data output list
/// \param solution_index output: solution index
/// \param size nr of elements to generate
/// \param nr_sols if > 1 the `nr_sols` elements from `solution_index` are also solutions
/// \return the element to search for
template<typename T>
T random_data(std::vector<T> &data,
              size_t &solution_index,
              const size_t size,
              const size_t nr_sols=1,
              const T __mask=0) {
	const T mask = __mask == 0 ? T(-1ull) : __mask;

	data.resize(size);
	for (uint64_t i = 0; i < size; ++i) {
		data[i] = fastrandombytes_uint64() & mask;
	}

	std::sort(data.begin(), data.end(),
		[](const auto &e1, const auto &e2) {
			return e1 < e2;
		}
	);

	ASSERT(std::is_sorted(data.begin(), data.end()));
	solution_index = fastrandombytes_uint64() % size;

	if (nr_sols > 1) {
		for (size_t i = 1; i < nr_sols; ++i) {
			data[solution_index + i] = data[solution_index];
		}
	}

	return data[solution_index];
}
