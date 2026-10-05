#pragma once

#include "jmp.h"

/// idea: https://docs.kernel.org/staging/static-keys.html
/// Source: https://raw.githubusercontent.com/qlibs/jmp/refs/heads/main/jmp
/// TODO: arm, riscv

namespace cryptanalysislib::jmp { 


/// TODO doc
template<>
class static_branch<bool> final {

    // sizeof(nop/jmp)
	using instr_t = internal::array<uint8_t, 5u>;
	using entry_t = internal::entry<2u>;
    
    // https://www.felixcloutier.com/x86/nop
	static constexpr instr_t NOP {
        0x0f, 0x1f, 0x44, 0x00, 0x00
    };

    // https://www.felixcloutier.com/x86/jmp
	static constexpr instr_t JMP {
        0xe9, 0x00, 0x00, 0x00, 0x00
    };
	static_assert(sizeof(NOP) == sizeof(JMP));

public:
	constexpr explicit(false) static_branch(const bool value) noexcept {
		void failed();
		if (value) failed();/// { false: nop, true: jmp }
	}
	constexpr static_branch(const static_branch &) noexcept = delete;
	constexpr static_branch(static_branch &&) noexcept = delete;
	constexpr static_branch &operator=(const static_branch &) noexcept = delete;
	constexpr static_branch &operator=(static_branch &&) noexcept = delete;

	inline const auto &operator=(const bool value) const noexcept {
		struct [[gnu::packed]] {
			uint8_t op{JMP[0u]};
			uint32_t offset{};
		} jmp{};
		const instr_t *ops[]{&NOP, (const instr_t *) &jmp};
		auto data = uint64_t(&__start___jmp);
		while (data != uint64_t(&__stop___jmp)) {
			const auto *entry = (const entry_t *) data;
			data += entry->size;
			if (entry->self != this) continue;
			jmp.offset = entry->offsets[size_t(value)];
			*static_cast<instr_t *>(entry->code) = *ops[size_t(value)];
		}
		return *this;
	}

	[[gnu::always_inline]] [[nodiscard]] inline explicit(false) operator bool() const noexcept {
		asm volatile goto(
		        "0: \n"
		        ".byte %c0, %c1, %c2, %c3, %c4 \n"
		        ".pushsection __jmp, \"aw\" \n"
		        ".quad %c7, 0b, %c5, %c6 \n"
		        ".long 0, %l[_true] - (0b + %c5) \n"
		        ".popsection \n"
		        : : "i"(NOP[0]), "i"(NOP[1]), "i"(NOP[2]), "i"(NOP[3]), "i"(NOP[4]),
		            "i"(sizeof(instr_t)),
		            "i"(this),
		            "i"(sizeof(entry_t))
		        : : _true);
		return false;
	_true:
		return true;
	}
};

template<class T, 
         const T Min,
         const T Max>
    requires requires(T t) {
        reinterpret_cast<T>(t); 
    } and (Max - Min >= 2 and Max - Min <= 7)
class static_branch<T, Min, Max> final {
	using entry_t = internal::entry<(Max - Min) + T(1)>;

    // https://www.felixcloutier.com/x86/jmp
	static constexpr uint8_t JMP[] {
        0xe9, 0x00, 0x00, 0x00, 0x00
    };

public:
	constexpr explicit(false) static_branch(const T value) noexcept {
		void failed();
		if (value != Min) failed();
	}
	constexpr static_branch(const static_branch &) noexcept = delete;
	constexpr static_branch(static_branch &&) noexcept = delete;
	constexpr static_branch &operator=(const static_branch &) noexcept = delete;
	constexpr static_branch &operator=(static_branch &&) noexcept = delete;

	inline const auto &operator=(const T value) const noexcept {
		auto data = uint64_t(&__start___jmp);
		while (data != uint64_t(&__stop___jmp)) {
			const auto *entry = (const entry_t *) data;
			data += entry->size;
			if (entry->self != this) continue;
			*(uint32_t *) (entry->code) = entry->offsets[value - Min];
		}
		return *this;
	}

	[[gnu::always_inline]] [[nodiscard]] inline explicit(false) operator T() const noexcept
	    requires(Max - Min == T(2))
	{
		asm volatile goto(
		        "0: \n"
		        ".byte %c0, %c1, %c2, %c3, %c4 \n"
		        ".pushsection __jmp, \"aw\" \n"
		        ".quad %c7, 1 + 0b, %c5, %c6 \n"
		        ".long 0 \n"
		        ".long %l[_1] - (1 + 0b + %c5) \n"
		        ".long %l[_2] - (1 + 0b + %c5) \n"
		        ".popsection \n"
		        : : "i"(JMP[0]), "i"(JMP[1]), "i"(JMP[2]), "i"(JMP[3]), "i"(JMP[4]),
		            "i"(sizeof(uint32_t)),
		            "i"(this),
		            "i"(sizeof(entry_t))
		        : : _1, _2);
		return T() + Min;
	_1:
		return T(1) + Min;
	_2:
		return T(2) + Min;
	}

	[[gnu::always_inline]] [[nodiscard]] inline explicit(false) operator T() const noexcept
	    requires(Max - Min == T(3))
	{
		asm volatile goto(
		        "0: \n"
		        ".byte %c0, %c1, %c2, %c3, %c4 \n"
		        ".pushsection __jmp, \"aw\" \n"
		        ".quad %c7, 1 + 0b, %c5, %c6 \n"
		        ".long 0 \n"
		        ".long %l[_1] - (1 + 0b + %c5) \n"
		        ".long %l[_2] - (1 + 0b + %c5) \n"
		        ".long %l[_3] - (1 + 0b + %c5) \n"
		        ".popsection \n"
		        : : "i"(JMP[0]), "i"(JMP[1]), "i"(JMP[2]), "i"(JMP[3]), "i"(JMP[4]),
		            "i"(sizeof(uint32_t)),
		            "i"(this),
		            "i"(sizeof(entry_t))
		        : : _1, _2, _3);
		return T() + Min;
	_1:
		return T(1) + Min;
	_2:
		return T(2) + Min;
	_3:
		return T(3) + Min;
	}

	[[gnu::always_inline]] [[nodiscard]] inline explicit(false) operator T() const noexcept
	    requires(Max - Min == T(4))
	{
		asm volatile goto(
		        "0: \n"
		        ".byte %c0, %c1, %c2, %c3, %c4 \n"
		        ".pushsection __jmp, \"aw\" \n"
		        ".quad %c7, 1 + 0b, %c5, %c6 \n"
		        ".long 0 \n"
		        ".long %l[_1] - (1 + 0b + %c5) \n"
		        ".long %l[_2] - (1 + 0b + %c5) \n"
		        ".long %l[_3] - (1 + 0b + %c5) \n"
		        ".long %l[_4] - (1 + 0b + %c5) \n"
		        ".popsection \n"
		        : : "i"(JMP[0]), "i"(JMP[1]), "i"(JMP[2]), "i"(JMP[3]), "i"(JMP[4]),
		            "i"(sizeof(uint32_t)),
		            "i"(this),
		            "i"(sizeof(entry_t))
		        : : _1, _2, _3, _4);
		return T() + Min;
	_1:
		return T(1) + Min;
	_2:
		return T(2) + Min;
	_3:
		return T(3) + Min;
	_4:
		return T(4) + Min;
	}

	[[gnu::always_inline]] [[nodiscard]] inline explicit(false) operator T() const noexcept
	    requires(Max - Min == T(5))
	{
		asm volatile goto(
		        "0: \n"
		        ".byte %c0, %c1, %c2, %c3, %c4 \n"
		        ".pushsection __jmp, \"aw\" \n"
		        ".quad %c7, 1 + 0b, %c5, %c6 \n"
		        ".long 0 \n"
		        ".long %l[_1] - (1 + 0b + %c5) \n"
		        ".long %l[_2] - (1 + 0b + %c5) \n"
		        ".long %l[_3] - (1 + 0b + %c5) \n"
		        ".long %l[_4] - (1 + 0b + %c5) \n"
		        ".long %l[_5] - (1 + 0b + %c5) \n"
		        ".popsection \n"
		        : : "i"(JMP[0]), "i"(JMP[1]), "i"(JMP[2]), "i"(JMP[3]), "i"(JMP[4]),
		            "i"(sizeof(uint32_t)),
		            "i"(this),
		            "i"(sizeof(entry_t))
		        : : _1, _2, _3, _4, _5);
		return T() + Min;
	_1:
		return T(1) + Min;
	_2:
		return T(2) + Min;
	_3:
		return T(3) + Min;
	_4:
		return T(4) + Min;
	_5:
		return T(5) + Min;
	}

	[[gnu::always_inline]] [[nodiscard]] inline explicit(false) operator T() const noexcept
	    requires(Max - Min == T(6))
	{
		asm volatile goto(
		        "0: \n"
		        ".byte %c0, %c1, %c2, %c3, %c4 \n"
		        ".pushsection __jmp, \"aw\" \n"
		        ".quad %c7, 1 + 0b, %c5, %c6 \n"
		        ".long 0 \n"
		        ".long %l[_1] - (1 + 0b + %c5) \n"
		        ".long %l[_2] - (1 + 0b + %c5) \n"
		        ".long %l[_3] - (1 + 0b + %c5) \n"
		        ".long %l[_4] - (1 + 0b + %c5) \n"
		        ".long %l[_5] - (1 + 0b + %c5) \n"
		        ".long %l[_6] - (1 + 0b + %c5) \n"
		        ".popsection \n"
		        : : "i"(JMP[0]), "i"(JMP[1]), "i"(JMP[2]), "i"(JMP[3]), "i"(JMP[4]),
		            "i"(sizeof(uint32_t)),
		            "i"(this),
		            "i"(sizeof(entry_t))
		        : : _1, _2, _3, _4, _5, _6);
		return T() + Min;
	_1:
		return T(1) + Min;
	_2:
		return T(2) + Min;
	_3:
		return T(3) + Min;
	_4:
		return T(4) + Min;
	_5:
		return T(5) + Min;
	_6:
		return T(6) + Min;
	}

	[[gnu::always_inline]] [[nodiscard]] inline explicit(false) operator T() const noexcept
	    requires(Max - Min == T(7))
	{
		asm volatile goto(
		        "0: \n"
		        ".byte %c0, %c1, %c2, %c3, %c4 \n"
		        ".pushsection __jmp, \"aw\" \n"
		        ".quad %c7, 1 + 0b, %c5, %c6 \n"
		        ".long 0 \n"
		        ".long %l[_1] - (1 + 0b + %c5) \n"
		        ".long %l[_2] - (1 + 0b + %c5) \n"
		        ".long %l[_3] - (1 + 0b + %c5) \n"
		        ".long %l[_4] - (1 + 0b + %c5) \n"
		        ".long %l[_5] - (1 + 0b + %c5) \n"
		        ".long %l[_6] - (1 + 0b + %c5) \n"
		        ".long %l[_7] - (1 + 0b + %c5) \n"
		        ".popsection \n"
		        : : "i"(JMP[0]), "i"(JMP[1]), "i"(JMP[2]), "i"(JMP[3]), "i"(JMP[4]),
		            "i"(sizeof(uint32_t)),
		            "i"(this),
		            "i"(sizeof(entry_t))
		        : : _1, _2, _3, _4, _5, _6, _7);
		return T() + Min;
	_1:
		return T(1) + Min;
	_2:
		return T(2) + Min;
	_3:
		return T(3) + Min;
	_4:
		return T(4) + Min;
	_5:
		return T(5) + Min;
	_6:
		return T(6) + Min;
	_7:
		return T(7) + Min;
	}
}; // end class

}; // end namespace cryptanalysisilib::jmp
