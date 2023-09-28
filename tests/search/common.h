#include <vector>
#include <algorithm>

#include "random.h"


template<typename T>
T random_data(std::vector<T> &data,
              size_t &solution_index,
              const size_t size,
              const T __mask=0) {
	const T mask = __mask == 0 ? -1ull : __mask;

	data.resize(size);
	for (uint64_t i = 0; i < size; ++i) {
		data[i] = fastrandombytes_uint64() & mask;
	}

	std::sort(data.begin(), data.end(),
			  [](const auto &e1, const auto &e2) {
				return e1 < e2;
			  }
	);

	assert(std::is_sorted(data.begin(), data.end()));
	solution_index = fastrandombytes_uint64() & size;
	return data[solution_index];
}
