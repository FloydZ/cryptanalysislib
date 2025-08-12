#ifndef CONTAINER_VECTOR_QUEUE_H
#define CONTAINER_VECTOR_QUEUE_H

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

/// NOTE: not thread safe
/// NOTE: const means; its not resizable
/// \tparam T base type
template<class T,
         const size_t _max_size=4096>
class ConstVectorQueue {
private:
    // NOTE: int32_t: max capacity is 2**32, which should be enough
    // NOTE: int32_t: signed integers needed to make the signed
    //  operations easy
    int32_t _front = 0, _back = 1;
    alignas(64) T __data[_max_size];

public:
    using container_type = T[];
    using const_reference = const T&;
    using reference = T&;
    using value_type = T;
    using size_type = size_t;

	/// \return the current element at the fotn
    [[nodiscard]] constexpr inline const_reference front() const noexcept {
        return __data[_front];
    }

	/// \return the last element in the back of the queue
    [[nodiscard]] constexpr inline const_reference back() const noexcept {
        return __data[_back];
    }

	/// \return
    [[nodiscard]] constexpr inline bool empty() const noexcept {
        return _front == (_back - 1);
    }

	/// \return max size the queue can handle. NOTE: its not resizable
    [[nodiscard]] constexpr inline size_t max_size() const noexcept {
        return _max_size;
    }

	/// \return current number of elements in the queue
    [[nodiscard]] constexpr inline size_t size() const noexcept {
        return std::abs(_back - _front - 1);
    }

	/// \param value[in]
    [[nodiscard]] constexpr inline bool push(const value_type &value) noexcept {
        if ((_back-1) == _max_size) {
            // wrap around
            _back = 1;
        }

        if (_back == _front) {
            //this means the queue is full
            return false;
        }

        __data[_back - 1] = value;
        _back += 1;
        return true;
    }

    /// \param value[in]: get moved into the queue
    [[nodiscard]] constexpr inline bool push(value_type &&value) noexcept {
        if ((_back-1) == _max_size) {
            // wrap around
            _back = 1;
        }

        if (_back == _front) {
            //this means the queue is full
            return false;
        }

        __data[_back - 1] = std::move(value);
        _back += 1;
        return true;
    }

    /// incremenets the front counter
    constexpr inline void pop() noexcept {
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
