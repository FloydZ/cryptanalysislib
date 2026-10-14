#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace cryptanalysislib {
	/// the default completion function of a `barrier`: nothing
	struct barrier_noop_completion {
		constexpr void operator()() const noexcept {}
	};

	/// Reusable thread barrier (like `std::barrier`): every phase waits until
	/// `expected` arrivals happened, then the last arriving thread calls the
	/// completion function and starts the next phase, which unblocks all
	/// waiting threads.
	/// Usage:
	/// \code
	/// 	barrier b(nthreads);
	/// 	// in every thread:
	/// 	for (...) { work(); b.arrive_and_wait(); }
	/// \endcode
	/// \tparam CompletionFunction called once per phase by the last arriving
	/// 	thread, before any thread of the phase is unblocked
	template<class CompletionFunction = barrier_noop_completion>
	class barrier {
		static_assert(std::is_nothrow_invocable_v<CompletionFunction &>);

	public:
		/// identifies the phase of an arrival, see `wait`
		class arrival_token {
			friend class barrier;
			uint32_t phase;
			constexpr explicit arrival_token(const uint32_t p) noexcept : phase(p) {}
		};

		/// \return the maximal number of participants
		static constexpr ptrdiff_t max() noexcept { return PTRDIFF_MAX; }

		/// \param expected[in]: number of participating threads
		/// \param f[in]: completion function
		constexpr explicit barrier(const ptrdiff_t expected,
		                           CompletionFunction f = CompletionFunction()) noexcept
		    : _M_expected(expected), _M_count(expected), _M_phase(0), _M_f(std::move(f)) {}

		~barrier() = default;
		barrier(const barrier &) = delete;
		barrier &operator=(const barrier &) = delete;

		/// arrives at the current phase, without waiting
		/// \param update[in]: number of arrivals
		/// \return token to wait for the end of the phase
		[[nodiscard]] inline arrival_token arrive(const ptrdiff_t update = 1) noexcept {
			// the phase cannot end before this arrival, so it is still current
			const uint32_t phase = _M_phase.load(std::memory_order_acquire);
			if (_M_count.fetch_sub(update, std::memory_order_acq_rel) == update) {
				// last arrival: complete the phase, start the next one
				_M_f();
				_M_count.store(_M_expected.load(std::memory_order_relaxed), std::memory_order_relaxed);
				_M_phase.store(phase + 1u, std::memory_order_release);
				_M_phase.notify_all();
			}

			return arrival_token(phase);
		}

		/// blocks until the phase of `token` ended
		/// \param token[in]: returned by `arrive` of this phase
		inline void wait(arrival_token &&token) const noexcept {
			uint32_t cur;
			while ((cur = _M_phase.load(std::memory_order_acquire)) == token.phase) {
				_M_phase.wait(cur, std::memory_order_acquire);
			}
		}

		/// arrives and blocks until the end of the current phase
		inline void arrive_and_wait() noexcept {
			wait(arrive());
		}

		/// arrives at the current phase and removes the calling thread from
		/// all following phases
		inline void arrive_and_drop() noexcept {
			_M_expected.fetch_sub(1, std::memory_order_relaxed);
			(void)arrive();
		}

	private:
		/// number of participants of the next phases
		std::atomic<ptrdiff_t> _M_expected;
		/// arrivals missing in the current phase
		std::atomic<ptrdiff_t> _M_count;
		/// current phase, waited on
		std::atomic<uint32_t> _M_phase;
		CompletionFunction _M_f;
	};
} // end namespace cryptanalysislib
