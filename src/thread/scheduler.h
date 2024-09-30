#ifndef CRYPTANALYISLIB_THREAD_SCHEDULER_H
#define CRYPTANALYISLIB_THREAD_SCHEDULER_H

// apple doesnt provide jthread!
#ifndef __APPLE__

#include <atomic>
#include <concepts>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <semaphore>
#include <thread>
#include <type_traits>
#include <version>
#include <algorithm>
#include <concepts>
#include <deque>
#include <mutex>
#include <optional>

// TODO only available on unix
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <assert.h>


#include "container/queue.h"
#include "atomic/annotated_mutex.h"
#include "pthread.h"

#include <rfl.hpp>
#include <rfl/json.hpp>

namespace cryptanalysislib {
	namespace details {

#ifdef __cpp_lib_move_only_function
		using default_function_type = std::move_only_function<void()>;
#else
		using default_function_type = std::function<void()>;
#endif

		/// TODO: apple does not suport jthread
#ifdef __APPLE__
		using default_thread_type = std::jthread;
#else 
		using default_thread_type = std::jthread;
#endif
	}  // namespace details

	class SchedulerConfig {
	public:
		constexpr static bool enable_try_block = false;
		constexpr static bool enable_remote_view = true;
	} schedulerConfig;

	// TODO: this is linux only
	// NOTE: this is needed, as the real internal `rusage`
	// uses strange unions, which are not supported by
	// `reflect-cpp`
	struct rusage__ {
		struct timeval ru_utime;
		struct timeval ru_stime;
		long int ru_maxrss;
		long int ru_ixrss;
		long int ru_idrss;
		long int ru_isrss;
		long int ru_minflt;
		long int ru_majflt;
		long int ru_nswap;
		long int ru_inblock;
		long int ru_oublock;
		long int ru_msgsnd;
		long int ru_msgrcv;
		long int ru_nsignals;
		long int ru_nvcsw;
		long int ru_nivcsw;
	};


	class SchedulerThreadLoad {
	public:
		rusage__ data;

		///// get the needed values
		void gather() noexcept {
			if (getrusage(RUSAGE_THREAD, (rusage *)&data) != 0) {
				ASSERT(false);
			}
		}

		/// \return maximum resident set size
		[[nodiscard]] constexpr inline uint64_t maxrss() const noexcept {
			return data.ru_maxrss;
		}

		/// \return integral shared memory size
		[[nodiscard]] constexpr inline uint64_t ixrss() const noexcept {
			return data.ru_ixrss;
		}

		/// \return integral unshared data size
		[[nodiscard]] constexpr inline uint64_t idrss() const noexcept {
			return data.ru_idrss;
		}

		/// \return integral unshared stack size
		[[nodiscard]] constexpr inline uint64_t isrss() const noexcept {
			return data.ru_isrss;
		}

		/// \return page reclaims (soft page faults)
		[[nodiscard]] constexpr inline uint64_t minflt() const noexcept {
			return data.ru_maxrss;
		}

		/// \return page faults (hard page faults )
		[[nodiscard]] constexpr inline uint64_t majflt() const noexcept {
			return data.ru_maxrss;
		}

		/// \return number of swaps
		[[nodiscard]] constexpr inline uint64_t nswaps() const noexcept {
			return data.ru_maxrss;
		}

		/// \return block input operations
		[[nodiscard]] constexpr inline uint64_t inblock() const noexcept {
			return data.ru_maxrss;
		}

		/// \return block output operations
		[[nodiscard]] constexpr inline uint64_t oublock() const noexcept {
			return data.ru_maxrss;
		}

		/// \return IPC messages sent
		[[nodiscard]] constexpr inline uint64_t msgsnd() const noexcept {
			return data.ru_maxrss;
		}

		/// \return IPC messages received
		[[nodiscard]] constexpr inline uint64_t msgrcv() const noexcept {
			return data.ru_maxrss;
		}

		/// \return signal received
		[[nodiscard]] constexpr inline uint64_t nsignals() const noexcept {
			return data.ru_maxrss;
		}

		/// \return voluntary context switches
		[[nodiscard]] constexpr inline uint64_t nvcsw() const noexcept {
			return data.ru_maxrss;
		}

		/// \return involuntary context switches
		[[nodiscard]] constexpr inline uint64_t nivcsw() const noexcept {
			return data.ru_maxrss;
		}

		/// \return user time in microseconds
		[[nodiscard]] constexpr inline uint64_t usertime() const noexcept {
			return data.ru_utime.tv_sec * 1000000 + data.ru_utime.tv_usec;
		}

		/// \return system time
		[[nodiscard]] constexpr inline uint64_t systime() const noexcept {
			return data.ru_stime.tv_sec * 1000000 + data.ru_stime.tv_usec;
		}
	};


	/// NOTE: just the data container. To access it use the Manager
	/// class. This is needed for the reflect-cpp framework
	struct SchedulerPerformance {
		// per thread information
		std::vector <SchedulerThreadLoad> schedulerThreadLoad;

		// global information
		int number_active_threads = 0;
		int number_enqueud_tasks = 0;
	};

	/// Manager class
	struct SchedulerPerformanceManager {
	private:
		SchedulerPerformance schedulerPerformance;

		const bool server;

		// socket communication
		constexpr static char *socket_path = "/tmp/cryptanalysislib_scheduler.socket4";
		constexpr static size_t buffer_size = 8096;
		int sockfd;
		std::thread server_thread;
	public:

		// SchedulerPerformanceManager() noexcept = default;

		/// either create a server or client instance. The client is sending data.
		/// The server is receiving data.
		/// \param server
		explicit SchedulerPerformanceManager(const bool server) noexcept : server(server) {
			if ((sockfd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
				std::cout << "ERROR: Creating socket" << std::endl;
				exit(1);
			}

			struct sockaddr_un serv_addr;
			bzero((char *) &serv_addr, sizeof(serv_addr));
			serv_addr.sun_family = AF_UNIX;
			strcpy(serv_addr.sun_path, socket_path);
			const int servlen = strlen(serv_addr.sun_path) + sizeof(serv_addr.sun_family);

			if (server) {
				if (bind(sockfd, (struct sockaddr *) &serv_addr, servlen) < 0) {
					std::cout << "ERROR: binding socket" << std::endl;
					exit(1);
				}

				std::cout << "starting server" << std::endl;
				auto server_worker = [&]() ->void {
					if (listen(sockfd, 5) != 0) {
						std::cout << "ERROR: listen: " << strerror(errno) << std::endl;
						exit(1);
					}

					struct sockaddr_un cli_addr;
					char buf[buffer_size];
					socklen_t clilen = sizeof(cli_addr);

					int newsockfd = accept((int)sockfd, (struct sockaddr *) &cli_addr, &clilen);
					if (newsockfd < 0) {
						std::cout << "ERROR: accepting: " << strerror(errno) << std::endl;
						exit(1);
					}

					// TODO with fork etc we can handle multiple streams, but for now
					// its fine
					while (true) {
						memset((void *)buf, 0, buffer_size);
						const uint32_t n = read(newsockfd, buf, buffer_size);
						if (n == 0) { break; }
						assert(n < buffer_size);
						std::cout << "recv: " << std::endl;
						std::cout << buf << std::endl;

						// const auto r = rfl::json::read<Shapes>(buf).value();
						// rfl::visit(handle_shapes, r);
						// write(newsockfd, "I got your message\n", 19);
					}

					std::cout << "server closing" << std::endl;
					close(newsockfd);
				};
				server_thread = std::thread(server_worker);
			} else {
				if (connect(sockfd, (struct sockaddr *) &serv_addr, servlen) < 0) {
					std::cout << "ERROR Connecting: " << strerror(errno) << std::endl;
					exit(1);
				}
			}
		}

		~SchedulerPerformanceManager() {
			std::cout << "closing socket" << std::endl;
			close(sockfd);
		}

		/// \param nr_threads
		/// \return nothing
		constexpr void resize(const uint32_t nr_threads) noexcept {
			ASSERT(nr_threads);
			schedulerPerformance.schedulerThreadLoad.resize(nr_threads);
		}

		/// write the
		void send() noexcept {
			const auto data = rfl::json::write(schedulerPerformance);
			std::cout << "sending data:" << std::endl;
			std::cout << data << std::endl;

			const int k = write(sockfd, data.data(), data.size());
			if (k < 0) {
				std::cout << "Error writing" << std:: endl;
			}
		}

		void serve() noexcept {
			ASSERT(server);
			server_thread.join();
		}

		/// \param i
		/// \return
		constexpr inline void gather(const uint32_t i) noexcept {
			ASSERT(i < schedulerPerformance.schedulerThreadLoad.size());
			schedulerPerformance.schedulerThreadLoad[i].gather();
		}

		///
		/// \param number_active_threads
		/// \param number_enqueud_tasks
		/// \return nothing
		constexpr inline void gather(const uint32_t number_active_threads,
									 const uint32_t number_enqueud_tasks) noexcept {
			schedulerPerformance.schedulerThreadLoad[0].gather();
			schedulerPerformance.number_active_threads = number_active_threads;
			schedulerPerformance.number_enqueud_tasks = number_enqueud_tasks;
			send();
		}

		/// \param i
		/// \return
		constexpr inline SchedulerThreadLoad& operator[](const uint32_t i) noexcept {
			ASSERT(i < schedulerPerformance.schedulerThreadLoad.size());
			return schedulerPerformance.schedulerThreadLoad[i];
		}
	};

	/// TODO config
	/// TODO implement pause
	/// 	std::vector as queue
	template <typename ThreadType = std::jthread,
	          typename FunctionType = details::default_function_type,
	          const SchedulerConfig &config=schedulerConfig>
#if __cplusplus > 201709L
	    requires std::invocable<FunctionType> &&
	             std::is_same_v<void, std::invoke_result_t<FunctionType>>
#endif
	class scheduler {
	private:
		constexpr static bool enable_try_block = config.enable_try_block;
		constexpr static bool enable_remote_view = config.enable_remote_view;
		SchedulerPerformanceManager schedulerPerformance{false};

	public:

		template <typename InitializationFunction = std::function<void(std::size_t)>>
		    requires std::invocable<InitializationFunction, std::size_t> &&
		             std::is_same_v<void, std::invoke_result_t<InitializationFunction, std::size_t>>
		explicit scheduler(const unsigned int &number_of_threads = std::thread::hardware_concurrency(),
		                   InitializationFunction init = [](std::size_t) {}) noexcept
		    : tasks_(number_of_threads) {
			std::size_t current_id = 0;
			if constexpr (enable_remote_view) {
				schedulerPerformance.resize(number_of_threads);
			}

			/// create all
			for (std::size_t i = 0; i < number_of_threads; ++i) {
				priority_queue_.push_back(size_t(current_id));
				threads_.emplace_back([&, i, id = current_id, init]
			                     (const std::stop_token &stop_tok) -> int {
					(void)i;
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
								schedulerPerformance.gather(get_num_running_tasks(),
								                            get_num_queued_tasks());
							} else {schedulerPerformance.gather(i); }
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

				    return 0 ;
				});
				// increment the thread id
				++current_id;
			}
		}

		~scheduler() noexcept {
			wait_for_tasks();

			// stop all threads
			for (std::size_t i = 0; i < threads_.size(); ++i) {
				threads_[i].request_stop();
				tasks_[i].signal.release();
				threads_[i].join();
			}
		}

		/// thread pool is non-copyable
		scheduler(const scheduler &) noexcept = delete;
		scheduler &operator=(const scheduler &) noexcept = delete;

        /// @brief Enqueue a task into the thread pool that returns a result.
        /// @details Note that task execution begins once the task is enqueued.
        /// @tparam Function An invokable type.
        /// @tparam Args Argument parameter pack
        /// @tparam ReturnType The return type of the Function
        /// @param f The callable function
        /// @param args The parameters that will be passed (copied) to the function.
        /// @return A std::future<ReturnType> that can be used to retrieve the returned value.
		template <typename Function, typename... Args,
		         typename ReturnType = std::invoke_result_t<Function &&, Args &&...>>
		    requires std::invocable<Function, Args...>
		[[nodiscard]] std::future<ReturnType> enqueue(Function f,
		                                              Args... args) noexcept {
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
			enqueue_task(std::move(task));
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
			             promise = shared_promise]() {
				if constexpr (enable_try_block) {
					try {
						if constexpr (std::is_same_v<ReturnType, void>) {
							func(largs...);
							promise->set_value();
						} else { promise->set_value(func(largs...)); }
					} catch (...) { promise->set_exception(std::current_exception()); }
				} else {
					if constexpr (std::is_same_v<ReturnType, void>) {
						func(largs...);
						promise->set_value();
					} else { promise->set_value(func(largs...)); }
				}
			};

			// get the future before enqueuing the task
			auto future = shared_promise->get_future();
			// enqueue the task
			enqueue_task(std::move(task));
			return future;
#endif
		}

		/**
         * @brief Enqueue a task to be executed in the thread pool. Any return value of the function
         * will be ignored.
         * @tparam Function An invokable type.
         * @tparam Args Argument parameter pack for Function
         * @param func The callable to be executed
         * @param args Arguments that will be passed to the function.
         */
		template <typename Function,
		          typename... Args>
		    requires std::invocable<Function, Args...>
		void enqueue_detach(Function &&func,
		                    Args &&...args) {
			enqueue_task(std::move([f = std::forward<Function>(func),
			                        ... largs =
			                                std::forward<Args>(args)]() mutable -> decltype(auto) {
				// suppress exceptions
				//try {
					if constexpr (std::is_same_v<void,std::invoke_result_t<Function &&, Args &&...>>) {
						std::invoke(f, largs...);
					} else {
						// the function returns an argument, but can be ignored
						std::ignore = std::invoke(f, largs...);
					}
				//} catch (...) {
				//}
			}));
		}

        /// @brief Wait for all tasks to finish.
        /// @details This function will block until all tasks have been completed.
		void wait_for_tasks() noexcept {
			if (in_flight_tasks_.load(std::memory_order_acquire) > 0) {
				// wait for all tasks to finish
				threads_complete_signal_.wait(false);
			}
		}

        /// @brief Makes best-case attempt to clear all tasks from the thread_pool
        /// @details Note that this does not guarantee that all tasks will be cleared, as currently
        /// running tasks could add additional tasks. Also a thread could steal a task from another
        /// in the middle of this.
        /// @return number of tasks cleared
		[[nodiscard]] inline size_t clear_tasks() noexcept {
			size_t removed_task_count{0};
			for (auto &task_list : tasks_) {
				removed_task_count += task_list.tasks.clear();
			}
			in_flight_tasks_.fetch_sub(removed_task_count, std::memory_order_release);
			unassigned_tasks_.fetch_sub(removed_task_count, std::memory_order_release);

			return removed_task_count;
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
        /// @return true if pause() has been called without an 
        /// intervening unpause().
		[[nodiscard]] constexpr bool inline is_paused() const noexcept {
			return pool_paused;
		}

        /// Get number of enqueued tasks.
        /// @return Number of tasks that have been enqueued but not yet started.
		[[nodiscard]] constexpr size_t get_num_queued_tasks() const {
			return tasks_.size();
		}

        /// Get number of in-progress tasks.
        /// @return Approximate number of tasks currently being processed by 
        ///     worker threads.
		[[nodiscard]] constexpr size_t get_num_running_tasks() const noexcept {
			return in_flight_tasks_.load();
		}

        /// Get total number of tasks in the pool.
        /// @return Approximate number of tasks both enqueued and running.
		[[nodiscard]] constexpr size_t get_num_tasks() const noexcept {
			return tasks_.size() + in_flight_tasks_.load();
		}

        /// @brief Returns the number of threads in the pool.
        /// @return std::size_t The number of threads in the pool.
		[[nodiscard]] constexpr inline auto size() const { return threads_.size(); }

	private:

		/// \tparam Function
		/// \param f function to enqueue
		template <typename Function>
		void enqueue_task(Function &&f) noexcept {
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
}

#endif
#endif
