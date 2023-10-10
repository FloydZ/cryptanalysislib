#ifndef CRYPTANALYSISLIB_ALIGN_H
#define CRYPTANALYSISLIB_ALIGN_H

#include <cstdint>

/// TODO use this
struct alignment_config {
public:
	// alignment in bytes
	const uint32_t alignment;

	constexpr alignment_config(const uint32_t alignment) noexcept :
			alignment(alignment) {}
};
#endif
