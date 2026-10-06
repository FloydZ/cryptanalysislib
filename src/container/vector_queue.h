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
    // NOTE: ring buffer: `_front` is the position of the first element,
    //  `_size` the number of elements
    size_t _front = 0, _size = 0;
    alignas(64) T __data[_max_size];

    /// \return the position after `i` in the ring buffer
    [[nodiscard]] constexpr static inline size_t next(const size_t i) noexcept {
        return (i + 1u == _max_size) ? 0u : i + 1u;
    }

    /// \return the position of the `i`-th element (counted from the front)
    [[nodiscard]] constexpr inline size_t pos(const size_t i) const noexcept {
        const size_t p = _front + i;
        return (p >= _max_size) ? p - _max_size : p;
    }

public:
    using container_type = T[];
    using const_reference = const T&;
    using reference = T&;
    using value_type = T;
    using size_type = size_t;

	/// \return the current element at the front
    [[nodiscard]] constexpr inline const_reference front() const noexcept {
        return __data[_front];
    }

	/// \return the last element in the back of the queue
    [[nodiscard]] constexpr inline const_reference back() const noexcept {
        return __data[pos(_size - 1u)];
    }

	/// \return
    [[nodiscard]] constexpr inline bool empty() const noexcept {
        return _size == 0;
    }

	/// \return max size the queue can handle. NOTE: its not resizable
    [[nodiscard]] constexpr inline size_t max_size() const noexcept {
        return _max_size;
    }

	/// \return current number of elements in the queue
    [[nodiscard]] constexpr inline size_t size() const noexcept {
        return _size;
    }

	/// \param value[in]
	/// \return false if the queue is full
    [[nodiscard]] constexpr inline bool push(const value_type &value) noexcept {
        if (_size == _max_size) {
            return false;
        }

        __data[pos(_size)] = value;
        _size += 1;
        return true;
    }

    /// \param value[in]: get moved into the queue
	/// \return false if the queue is full
    [[nodiscard]] constexpr inline bool push(value_type &&value) noexcept {
        if (_size == _max_size) {
            return false;
        }

        __data[pos(_size)] = static_cast<value_type &&>(value);
        _size += 1;
        return true;
    }

    /// removes the front element. Does nothing if the queue is empty.
    constexpr inline void pop() noexcept {
        if (_size == 0) {
            return;
        }

        _front = next(_front);
        _size -= 1;
    }

    /// just some debugging
    void into() noexcept {
        std::cout << "{ \"name\": \"ConstVectorQueue\"";
        std::cout << ", \"_front\": " << _front;
        std::cout << ", \"_size\": " << _size;
        std::cout << ", \"_data\": ";
        for (size_t i = 0; i < _size; i++) {
            std::cout << __data[pos(i)] << " ";
        }
        std::cout << "}" << std::endl;
    }
};
#endif //VECTOR_QUEUE_H
