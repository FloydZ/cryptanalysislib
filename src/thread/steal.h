#ifndef CRYPTANALYISLIB_THREAD_SCHEDULER_H
#define CRYPTANALYISLIB_THREAD_SCHEDULER_H


#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <thread>
#include <type_traits>
#include <version>

#include "atomic/annotated_mutex.h"
#include "container/queue.h"
#include "performance.h"


namespace cryptanalysislib {
	/// \tparam ThreadType
	/// \tparam FunctionType
	/// \tparam config
	template<typename ThreadType = details::default_thread_type,
	         typename FunctionType = details::default_function_type,
	         const SchedulerConfig &config = schedulerConfig>
#if __cplusplus > 201709L
	    requires std::invocable<FunctionType> &&
	             std::is_same_v<void, std::invoke_result_t<FunctionType>>
#endif
	class StealingScheduler {
	private:
		constexpr static bool enable_try_block = config.enable_try_block;
		constexpr static bool enable_remote_view = config.enable_remote_view;
		SchedulerPerformanceManager *schedulerPerformance;

	public:
		/// TODO use the MOVE operator from SimpleScheduler
		/// \tparam InitializationFunction
		/// \param number_of_threads
		/// \param init
		template<typename InitializationFunction = std::function<void(std::size_t)>>
		    requires std::invocable<InitializationFunction, std::size_t> &&
		             std::is_same_v<void, std::invoke_result_t<InitializationFunction, std::size_t>>
		explicit StealingScheduler(const unsigned int &number_of_threads = std::thread::hardware_concurrency(),
                                    InitializationFunction init = [](std::size_t) {}) noexcept 
    : tasks_(number_of_threads) {
			std::size_t current_id = 0;
			if constexpr (enable_remote_view) {
				schedulerPerformance = new SchedulerPerformanceManager{false};
				schedulerPerformance->resize(number_of_threads);
			}

			/// create all threads
			for (std::size_t i = 0; i < number_of_threads; ++i) {
				priority_queue_.push_back(size_t(current_id));
				threads_.emplace_back([&, i, id = current_id, init](const std::stop_token &stop_tok) -> int {
					(void) i;
					/// invoke the init function on the thread
					if constexpr (enable_try_block) {
						try {
							std::invoke(init, id);
						} catch (...) { return 0; }
					} else {
						std::invoke(init, id);
					}

					do {
						// wait until signaled
						tasks_[id].signal.acquire();
						if constexpr (enable_remote_view) {
							// not nice but easy
							if (i == 0) {
								schedulerPerformance->gather(get_num_running_tasks(),
								                             get_num_queued_tasks());
							} else {
								schedulerPerformance->gather(i);
							}
						}

						do {
							// invoke the task
							while (auto task = tasks_[id].tasks.pop_front()) {
								// decrement the unassigned tasks as the task is now going
								// to be executed
								unassigned_tasks_.fetch_sub(1, std::memory_order_release);
								// invoke the task
								std::invoke(std::move(task.value()));
								// the above task can push more work onto the pool, so we
								// only decrement the in flights once the task has been
								// executed because now it's now longer "in flight"
								in_flight_tasks_.fetch_sub(1, std::memory_order_release);
							}

							// try to steal a task
							for (std::size_t j = 1; j < tasks_.size(); ++j) {
								const std::size_t index = (id + j) % tasks_.size();
								if (auto task = tasks_[index].tasks.steal()) {
									// steal a task
									unassigned_tasks_.fetch_sub(1, std::memory_order_release);
									std::invoke(std::move(task.value()));
									in_flight_tasks_.fetch_sub(1, std::memory_order_release);
									// stop stealing once we have invoked a stolen task
									break;
								}
							}
							// check if there are any unassigned tasks before rotating to the
							// front and waiting for more work
						} while (unassigned_tasks_.load(std::memory_order_acquire) > 0);

						priority_queue_.rotate_to_front(id);
						// check if all tasks are completed and release the barrier (binary
						// semaphore)
						if (in_flight_tasks_.load(std::memory_order_acquire) == 0) {
							threads_complete_signal_.store(true, std::memory_order_release);
							threads_complete_signal_.notify_one();
						}

					} while (!stop_tok.stop_requested());

					return 0;
				});
				// increment the thread id
				++current_id;
			}
		}

		~StealingScheduler() noexcept {
			wait_for_tasks();

			// stop all threads
			for (std::size_t i = 0; i < threads_.size(); ++i) {
				threads_[i].request_stop();
				tasks_[i].signal.release();
				threads_[i].join();
			}
		}

		/// thread pool is non-copyable
		StealingScheduler(const StealingScheduler &) noexcept = delete;
		StealingScheduler &operator=(const StealingScheduler &) noexcept = delete;

		/// \brief submit a task into the thread pool that returns a result.
		/// \details Note that task execution begins once the task is submitd.
		/// \tparam Function An invokable type.
		/// \tparam Args Argument parameter pack
		/// \tparam ReturnType The return type of the Function
		/// \param f The callable function
		/// \param args The parameters that will be passed (copied) to the function.
		/// \return A std::future<ReturnType> that can be used to retrieve the returned value.
		template<typename Function,
                 typename... Args,
		         typename ReturnType = std::invoke_result_t<Function &&, Args &&...>>
		    requires std::invocable<Function, Args...>
		[[nodiscard]] std::future<ReturnType> submit(Function &&f,
		                                              Args&&... args) noexcept {
#ifdef __cpp_lib_move_only_function
			// we can do this in C++23 because we now have support for move only functions
			std::promise<ReturnType> promise;
			auto future = promise.get_future();
			auto task = [func = std::move(f), ... largs = std::move(args),
			             promise = std::move(promise)]() mutable {
				try {
					if constexpr (std::is_same_v<ReturnType, void>) {
						func(largs...);
						promise.set_value();
					} else {
						promise.set_value(func(largs...));
					}
				} catch (...) {
					promise.set_exception(std::current_exception());
				}
			};
			submit_task(std::move(task));
			return future;
#else
			/*
             * use shared promise here so that we don't break the promise later (until C++23)
             *
             * with C++23 we can do the following:
             *
             * std::promise<ReturnType> promise;
             * auto future = promise.get_future();
             * auto task = [func = std::move(f), ...largs = std::move(args),
                              promise = std::move(promise)]() mutable {...};
             */
			auto shared_promise = std::make_shared<std::promise<ReturnType>>();
			auto task = [func = std::move(f), ... largs = std::move(args),
				         promise = shared_promise]() __attribute__((always_inline)) {
				if constexpr (enable_try_block) {
					try {
						if constexpr (std::is_same_v<ReturnType, void>) {
							func(largs...);
							promise->set_value();
						} else {
							promise->set_value(func(largs...));
						}
					} catch (...) { promise->set_exception(std::current_exception()); }
				} else {
					if constexpr (std::is_same_v<ReturnType, void>) {
						func(largs...);
						promise->set_value();
					} else {
						promise->set_value(func(largs...));
					}
				}
			};

			// get the future before enqueuing the task
			auto future = shared_promise->get_future();
			// submit the task
			submit_task(std::move(task));
			return future;
#endif
		}

        /// @brief submit a task to be executed in the thread pool.
        /// Any return value of the function
        /// will be ignored.
        /// @tparam Function An invokable type.
        /// \tparam Args Argument parameter pack for Function
        /// \param func The callable to be executed
        /// \param args Arguments that will be passed to the function.
		template<typename Function,
		         typename... Args>
		    requires std::invocable<Function, Args...>
		void submit_detach(Function &&func,
		                    Args &&...args) {
			submit_task(std::move([f = std::forward<Function>(func),
			                        ... largs =
			                                std::forward<Args>(args)]() mutable -> decltype(auto) {
				if constexpr (std::is_same_v<void, std::invoke_result_t<Function &&, Args &&...>>) {
					std::invoke(f, largs...);
				} else {
					// the function returns an argument, but can be ignored
					std::ignore = std::invoke(f, largs...);
				}
			}));
		}

		/// \brief Wait for all tasks to finish.
		/// \details This function will block until all tasks have been completed.
		void wait_for_tasks() noexcept {
			if (in_flight_tasks_.load(std::memory_order_acquire) > 0) {
				// wait for all tasks to finish
				threads_complete_signal_.wait(false);
			}
		}

		/// pausing executing enqueud tasks
		void inline pause() noexcept {
			pool_paused = true;
		}

		/// Resume executing queued tasks.
		void unpause() noexcept {
			pool_paused = false;
		}

		/// Check whether the pool is paused.
		/// \return true if pause() has been called without an
		///         intervening unpause().
		[[nodiscard]] constexpr bool inline is_paused() const noexcept {
			return pool_paused;
		}

		/// \brief Makes best-case attempt to clear all tasks from the thread_pool
		/// \details Note that this does not guarantee that all tasks will be cleared, as currently
		/// running tasks could add additional tasks. Also a thread could steal a task from another
		/// in the middle of this.
		/// @return number of tasks cleared
		[[nodiscard]] inline size_t clear_tasks() noexcept {
			size_t removed_task_count{0};
			for (auto &task_list: tasks_) {
				removed_task_count += task_list.tasks.clear();
			}
			in_flight_tasks_.fetch_sub(removed_task_count, std::memory_order_release);
			unassigned_tasks_.fetch_sub(removed_task_count, std::memory_order_release);

			return removed_task_count;
		}

		/// Get number of enqueued tasks.
		/// \return: Number of tasks that have been enqueued but not yet started.
		[[nodiscard]] constexpr size_t get_num_queued_tasks() const {
			return tasks_.size();
		}

		/// Get number of in-progress tasks.
		/// \return Approximate number of tasks currently being processed by
		///     worker threads.
		[[nodiscard]] constexpr size_t get_num_running_tasks() const noexcept {
			return in_flight_tasks_.load();
		}

		/// Get total number of tasks in the pool.
		/// \return Approximate number of tasks both enqueued and running.
		[[nodiscard]] constexpr size_t get_num_tasks() const noexcept {
			return tasks_.size() + in_flight_tasks_.load();
		}

		/// brief Returns the number of threads in the pool.
		/// \return std::size_t The number of threads in the pool.
		[[nodiscard]] constexpr inline auto size() const noexcept { return threads_.size(); }
		[[nodiscard]] constexpr inline auto get_num_threads() const noexcept { return size(); }

        [[nodiscard]] uint32_t set_num_threads(uint32_t num_threads) noexcept {
            // not implemented
            return num_threads;
        }
	private:
		/// \tparam Function
		/// \param f function to enqueue
		template<typename Function>
		void submit_task(Function &&f) noexcept {
			auto i_opt = priority_queue_.copy_front_and_rotate_to_back();
			if (!i_opt.has_value()) {
				// would only be a problem if there are zero threads
				return;
			}

			// get the index
			auto i = *(i_opt);

			// increment the unassigned tasks and in flight tasks
			unassigned_tasks_.fetch_add(1, std::memory_order_release);
			const auto prev_in_flight = in_flight_tasks_.fetch_add(1, std::memory_order_release);

			// reset the in flight signal if the list was previously empty
			if (prev_in_flight == 0) {
				threads_complete_signal_.store(false, std::memory_order_release);
			}

			// assign work
			tasks_[i].tasks.push_back(std::forward<Function>(f));
			tasks_[i].signal.release();
		}

		///
		struct task_item {
			thread_safe_queue<FunctionType> tasks{};
			std::binary_semaphore signal{0};
		};

		std::vector<ThreadType> threads_;
		std::vector<task_item> tasks_;
		thread_safe_queue<std::size_t> priority_queue_;

		// guarantee these get zero-initialized
		std::atomic_int_fast64_t unassigned_tasks_{0}, in_flight_tasks_{0};
		std::atomic_bool threads_complete_signal_{false};
		std::atomic_bool pool_paused{false};
	};
}// namespace cryptanalysislib

#endif
