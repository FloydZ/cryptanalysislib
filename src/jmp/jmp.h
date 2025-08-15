#pragma once 

#include <cassert>
#include <cstdint>

// TODO really needed
// #pragma GCC system_header

extern "C" int getpagesize() throw();
extern "C" int mprotect(void *, __SIZE_TYPE__, int) throw();
extern "C" void *__start___jmp [[gnu::section("__jmp")]] [[gnu::weak]];
extern "C" void *__stop___jmp [[gnu::section("__jmp")]] [[gnu::weak]];


namespace cryptanalysislib::jmp {

namespace internal {
/// simple helper class for an array
/// \param T
/// \param N
template<class T,
         const size_t N>
struct array {
    ///
	[[nodiscard]] static constexpr auto size() noexcept {
        return N; 
    }
    ///
	[[nodiscard]] constexpr const auto &operator[](const size_t index) const noexcept {
        assert(index < N);
		return data[index];
	}

	T data[N]{};
};


/// TODO doc
template<size_t N>
struct [[gnu::packed]] entry {
	uint64_t size{};        /// sizeof(entry<N>)
	void *code{};      /// code memory (to be patched)
	uint64_t len{};         /// code length
	const void *self{};/// self identifier
	uint32_t offsets[N]{};  /// jmp offsets
};

}; // end namespace internal

/// TODO doc
template<class T, T...>
struct static_branch;

}; // end namespace cryptanalyslib::jmp


#if defined(__x86_64__)
#include "x86.h" 
#endif

#if defined(__arm__)
#include "arm.h" 
#endif

#if defined(__riscv__)
#include "riscv.h" 
#endif



namespace cryptanalysislib::jmp {

/// Makes required pages writable for code patching
/// Note: Must be called before changing the branch value (`branch = ...`)
///       Should be called once at the startup
/// \param page_size page size (default: getpagesize())
/// \param permissions protect permissions (default: PROT_READ | PROT_WRITE | PROT_EXEC)
/// \return true if succesful, false on error (errno is set to indicate the error)
[[nodiscard]] static inline auto init(const uint64_t page_size = getpagesize(),
                                      const uint64_t permissions = 0b111) noexcept -> bool {
	using entry_t = internal::entry<0u>;
	auto data = uint64_t(&__start___jmp);
	while (data != uint64_t(&__stop___jmp)) {
		const auto *entry = (const entry_t *) data;
		data += entry->size;
		if (const auto memory = uint64_t(entry->code) & ~(page_size - 1u);
		    mprotect((void *) memory, uint64_t(entry->code) - memory + entry->len, permissions)) {
			return false;
		}
	}
	return true;
}
}; // end namespace cryptanalyslib::jmp
