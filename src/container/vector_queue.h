#ifndef CONTAINER_VECTOR_QUEUE_H
#define CONTAINER_VECTOR_QUEUE_H

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "helper.h"

/// NOTE: not thread safe
/// NOTE: const means; its not resizable
/// \tparam T base type
/// \tparam V vector type, only [], needed
template<class T>
class ConstVectorQueue {
private:
    constexpr static size_t _max_size = 4096;

    // NOTE: int32_t: max capacity is 2**32, which should be enough
    // NOTE: int32_t: signed integers needed to make the signed
    //  operations easy
    int32_t _front = 0, _back = 1;
    alignas(64) T __data[_max_size];

public:
    // typedef typename	V::value_type		value_type;
    // typedef typename	V::reference		reference;
    // typedef typename	V::const_reference	const_reference;
    // typedef typename	V::size_type		size_type;
    // typedef		        V			        container_type;

    using container_type = T[];
    using const_reference = const T&;
    using reference = T&;
    using value_type = T;
    using size_type = size_t;

	/// \return the current element at the fotn
    [[nodiscard]] inline const_reference front() const noexcept {
        return __data[_front];
    }

	/// \return
    [[nodiscard]] inline const_reference back() const noexcept {
        return __data[_back];
    }

	/// @return
    [[nodiscard]] inline bool empty() const noexcept {
        return _front == (_back - 1);
    }

	/// @return
    [[nodiscard]] inline size_t max_size() const noexcept {
        return _max_size;
    }

	/// @return
    [[nodiscard]] inline size_t size() const noexcept {
        return std::abs(_back - _front - 1);
    }

	/// @param value
    inline void push(const value_type &value) noexcept {
        if ((_back-1) == _max_size) {
            // we silently overwrite stuff
            // if (_front == 0) {
            //     // in this case we silently fail
            //     ASSERT(false);
            //     return;
            // }

            // wrap around
            _back = 1;
        }

        if (_back == _front) {
            //this means the queue is full
            ASSERT(false);
            return;
        }

        __data[_back - 1] = value;
        _back += 1;
    }

    /// @param value
    inline void push(value_type &&value) noexcept {
        if ((_back-1) == _max_size) {
            // we silently overwrite stuff
            // if (_front == 0) {
            //     // in this case we silently fail
            //     ASSERT(false);
            //     return;
            // }

            // wrap around
            _back = 1;
        }

        if (_back == _front) {
            //this means the queue is full
            ASSERT(false);
            return;
        }

        __data[_back - 1] = std::move(value);
        _back += 1;
    }

    ///
    inline void pop() noexcept {
        _front += 1;
        if (_front == _max_size) { _front = 0; }
    }

    /// just some debugging
    void into() noexcept {
        std::cout << "{ \"name\": \"ConstVectorQueue\"";
        std::cout << ", \"_front\": " << _front;
        std::cout << ", \"_back\": " << _back;
        std::cout << ", \"_data\": ";
        for (size_t i = _front; i<(size_t)(_back-1); i++) {
            std::cout << __data[i] << " ";
        }
        std::cout << "}" << std::endl;
    }
};
#endif //VECTOR_QUEUE_H
