#ifndef CRYPTANALYSISLIB_CONTAINER_VECTOR_VIEW
#define CRYPTANALYSISLIB_CONTAINER_VECTOR_VIEW

#include <cstdint>
#include <cstdlib>
#include <cassert>

// needed for `ElementDataAble`
#include "element.h"


/// just a simple view on a vector
/// both const and non const
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


    /// \param container[in]: 
    /// \param start[in]: 
    /// \param end[in]: 
	constexpr ContainerView(const ContainerView *container,
						    const size_t start,
						    const size_t end) :
			start(start), end(end), container(container) {
		assert(start < end);
		assert(end <= length);
	}

    /// \return the size=#of elements in the vector
	[[nodiscard]] constexpr inline size_t size() const noexcept {
		return end - start;
	}

    /// \param i[in]: 
	[[nodiscard]] constexpr inline DataType &operator[](const size_t i) noexcept {
		return container->get(start + i);
	}

    /// \param i[in]: 
	[[nodiscard]] constexpr inline const DataType &operator[](const size_t i) const noexcept {
		return container->get(start + i);
	}
};



#endif
