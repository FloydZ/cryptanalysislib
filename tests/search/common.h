#include <vector>
#include <algorithm>

#include "helper.h"
#include "random.h"

/// generates a random sorted list.
/// \tparam T base element type
/// \param data output list
/// \param solution_index output: solution index
/// \param size nr of elements to generate
/// \param nr_sols if > 1 the `nr_sols` elements from `solution_index` are also solutions
/// \return the element to search for: the solution
template<typename L, typename T>
T random_data(L &data,
              size_t &solution_index,
              const size_t size,
              const size_t nr_sols=1,
              const T __mask=0) {

	data.resize(size);
	for (uint64_t i = 0; i < size; ++i) {
		if constexpr (std::is_integral_v<T>) {
			const T mask = __mask == 0 ? T(-1ull) : __mask;
			data[i] = fastrandombytes_uint64() & mask;
		} else {
			data[i].random();
		}
	}

	std::sort(data.begin(), data.end(),
		[](const auto &e1, const auto &e2) {
			return e1 < e2;
		}
	);

	ASSERT(std::is_sorted(data.begin(), data.end()));
	solution_index = fastrandombytes_uint64() % size;

	if (nr_sols > 1ull) {
		for (size_t i = 1u; i < nr_sols; ++i) {
			if (solution_index + i >= size) { break; }
			data[solution_index + i] = data[solution_index];
		}
	}

	return data[solution_index];
}
