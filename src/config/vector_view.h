#ifndef CRYPTANALYSISLIB_CONTAINER_VECTOR_VIEW
#define CRYPTANALYSISLIB_CONTAINER_VECTOR_VIEW

#include <cstdint>
#include <cstdlib>

#include "helper.h"
#include "element.h"


template<class Container>
#if __cplusplus > 201709L
    requires ElementDataAble<Container>
#endif
class ContainerView {
private:
    constexpr static uint32_t length = Container::length;
    using DataType = Container::DataType;

	constexpr ContainerView() = default;

	const size_t start;
	const size_t end;
	const Container *container;

public:

	constexpr ContainerView(const ContainerView *container,
						    const size_t start,
						    const size_t end) :
			start(start), end(end), container(container) {
		ASSERT(start < end);
		ASSERT(end < length);
	}

	[[nodiscard]] constexpr inline size_t size() const noexcept {
		return end - start;
	}

	[[nodiscard]] constexpr inline DataType &operator[](const size_t i) noexcept {
		return container->get(start + i);
	}

	[[nodiscard]] constexpr inline const DataType &operator[](const size_t i) const noexcept {
		return container->get(start + i);
	}
};



#endif
