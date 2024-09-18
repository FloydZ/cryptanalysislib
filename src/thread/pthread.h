#ifndef CRYPTANALYSISLIB_THREAD_PTHREAD_H
#define CRYPTANALYSISLIB_THREAD_PTHREAD_H

#ifndef __APPLE__
#include <pthread.h>
#include <type_traits>
#include <functional>

#include "helper.h"


namespace cryptanalysislib {
namespace internal {

	// small helper function for wrapping a function pointer
	// and it arguments in the same function, which fullfills
	// the signature requirements of pthreads
	void *pthread_helper(void *p) {
		static void *t = nullptr;
		if (t == nullptr) {
			if (p != nullptr) {
				t = p;
			}

			return nullptr;
		}

		if (p != nullptr) {
			auto tt = (void *(*) (void *) ) t;
			tt(p);
		}

		return t;
	}
} // namespace: internal


/// just a plain wrapper around pthread
/// maybe its only good for benchmarking
struct pthread {
private:
	pthread_t t;
public:
	pthread() noexcept = default;
	pthread(pthread &&t) noexcept = default;
	pthread(const pthread&) = delete;

	template< class F, class... Args >
	explicit pthread(F&& f, Args&&... args) noexcept : t(0) {
		//static const std::tuple<Args&&...> _args = std::forward_as_tuple(args...);
		static std::tuple<Args...> _args = { args... };
		//static auto _args = std::forward<Args>(args);
		static F fn = std::forward<F>(f);
		auto kek = [&] (void *ptr) -> void* {
			if constexpr (internal::lambda_details<decltype(fn)>::argument_count > 0) {
				//std::invoke<F, Args...>(std::forward<F>(fn), std::get<Args...>(_args));
				std::apply(std::forward<F>(fn),
				           std::forward<decltype(_args)>(_args));
			} else {
				fn();
			}

			return ptr;
		};

		auto *ptr = CallableMetadata<decltype(kek)>::generatePointer(kek);
		pthread_create(&t, nullptr, ptr, (void *) nullptr);
	}

	[[nodiscard]] constexpr inline bool joinable() const noexcept {
		return true;
	}

	[[nodiscard]] inline uint32_t get_id() const noexcept {
		return t;
	}

	[[nodiscard]] constexpr inline static uint32_t hardware_concurrency() noexcept {
		return std::thread::hardware_concurrency();
	}

	void join() const noexcept {
		pthread_join(t, nullptr);
	}

	void detach() const noexcept {
		pthread_detach(t);
	}

	// only for jthread
	bool request_stop() noexcept {
		return true;// TODO
	}
};
} // namespace: cryptanalysislib

#endif // __APPLE__
#endif
