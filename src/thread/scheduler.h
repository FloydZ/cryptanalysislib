#ifndef CRYPTANALYISLIB_THREAD_SCHEDULER_H
#define CRYPTANALYISLIB_THREAD_SCHEDULER_H

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

#include "container/queue.h"
#include "pthread.h"

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

	/// TODO config
	/// TODO implement pause
	/// 	std::vector as queue
	template <typename ThreadType = std::jthread,
	          typename FunctionType = details::default_function_type>
	    requires std::invocable<FunctionType> &&
	             std::is_same_v<void, std::invoke_result_t<FunctionType>>
	class scheduler {
	public:

		template <typename InitializationFunction = std::function<void(std::size_t)>>
		    requires std::invocable<InitializationFunction, std::size_t> &&
		             std::is_same_v<void, std::invoke_result_t<InitializationFunction, std::size_t>>
		explicit scheduler(const unsigned int &number_of_threads = std::thread::hardware_concurrency(),
		                   InitializationFunction init = [](std::size_t) {}) noexcept
		    : tasks_(number_of_threads) {
			std::size_t current_id = 0;
			for (std::size_t i = 0; i < number_of_threads; ++i) {
				priority_queue_.push_back(size_t(current_id));
				//try {
					threads_.emplace_back([&, id = current_id,
					                       init](const std::stop_token &stop_tok) -> int {
						//// invoke the init function on the thread
						try {
							std::invoke(init, id);
						} catch (...) {
							// suppress exceptions
						    return 0;
						}

						do {
							// wait until signaled
							tasks_[id].signal.acquire();

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

				//} catch (...) {
				//	// catch all

				//	// remove one item from the tasks
				//	tasks_.pop_back();

				//	// remove our thread from the priority queue
				//	std::ignore = priority_queue_.pop_back();
				//}
			}
		}

		~scheduler() {
			wait_for_tasks();

			// stop all threads
			for (std::size_t i = 0; i < threads_.size(); ++i) {
				threads_[i].request_stop();
				tasks_[i].signal.release();
				threads_[i].join();
			}
		}

		/// thread pool is non-copyable
		scheduler(const scheduler &) = delete;
		scheduler &operator=(const scheduler &) = delete;

		/**
         * @brief Enqueue a task into the thread pool that returns a result.
         * @details Note that task execution begins once the task is enqueued.
         * @tparam Function An invokable type.
         * @tparam Args Argument parameter pack
         * @tparam ReturnType The return type of the Function
         * @param f The callable function
         * @param args The parameters that will be passed (copied) to the function.
         * @return A std::future<ReturnType> that can be used to retrieve the returned value.
         */
		template <typename Function, typename... Args,
		         typename ReturnType = std::invoke_result_t<Function &&, Args &&...>>
		    requires std::invocable<Function, Args...>
		[[nodiscard]] std::future<ReturnType> enqueue(Function f, Args... args) {
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
				//try {
					if constexpr (std::is_same_v<ReturnType, void>) {
						func(largs...);
						promise->set_value();
					} else {
						promise->set_value(func(largs...));
					}

				// } catch (...) {
				// 	promise->set_exception(std::current_exception());
				// }
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
		template <typename Function, typename... Args>
		    requires std::invocable<Function, Args...>
		void enqueue_detach(Function &&func, Args &&...args) {
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

		/**
         * @brief Wait for all tasks to finish.
         * @details This function will block until all tasks have been completed.
         */
		void wait_for_tasks() {
			if (in_flight_tasks_.load(std::memory_order_acquire) > 0) {
				// wait for all tasks to finish
				threads_complete_signal_.wait(false);
			}
		}

		/**
         * @brief Makes best-case attempt to clear all tasks from the thread_pool
         * @details Note that this does not guarantee that all tasks will be cleared, as currently
         * running tasks could add additional tasks. Also a thread could steal a task from another
         * in the middle of this.
         * @return number of tasks cleared
         */
		[[nodiscard]] inline size_t clear_tasks() noexcept {
			size_t removed_task_count{0};
			for (auto &task_list : tasks_) {
				removed_task_count += task_list.tasks.clear();
			}
			in_flight_tasks_.fetch_sub(removed_task_count, std::memory_order_release);
			unassigned_tasks_.fetch_sub(removed_task_count, std::memory_order_release);

			return removed_task_count;
		}

		void inline pause() noexcept {
			pool_paused = true;
		}

		/**
         * Resume executing queued tasks.
         */
		void unpause() noexcept {
			pool_paused = false;
			//task_cv.notify_all();
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
		/// \param f
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


    class HeartbeatSchedulerConfig {
    public:
        /// The number of background workers. If `null` this chooses a sensible
        /// default based on your system (i.e. number of cores).
        constexpr static uint32_t background_worker_count = 0;
        /// How often a background thread is interrupted to find more work.
        constexpr static size_t heartbeat_interval = 100*1000; // NS_PERD_US 1000L
    } heartbeatSchedulerConfig;


    ///
    ///
	template <typename ThreadType = std::jthread,
	          typename FunctionType = details::default_function_type,
              typename Mutex = std::mutex,
              const HeartbeatSchedulerConfig &config = heartbeatSchedulerConfig>
	    requires std::invocable<FunctionType> &&
	             std::is_same_v<void, std::invoke_result_t<FunctionType>>
	class HeartbeatScheduler {
    private:
        /// forward decleration
        struct Task;
        struct Worker;

        ///
        Mutex mutex;
        /// List of all workers.
        std::vector<Worker *> workers; 

        /// List of all background workers.
        std::vector<ThreadType> background_threads; 

        /// The background thread which beats.
        ThreadType heartbeat_thread;

        /// A pool for the JobExecuteState, to minimize allocations.
        //execute_state_pool: std.heap.MemoryPool(JobExecuteState),
    
        /// This is used to signal that more jobs are now ready.
        /// job_ready: std.Thread.Condition = .{},
        std::condition_variable job_ready;

        /// This is used to wait for the background workers to be available initially.
        // workers_ready: std.Thread.Semaphore = .{},
        std::counting_semaphore<1> workers_ready{0};

        /// This is set to true once we're trying to stop.
        bool is_stopping = false;

        constexpr static std::size_t max_result_words = 4;
        constexpr static uint32_t background_worker_count = config.background_worker_count;
        constexpr static size_t heartbeat_interval = config.heartbeat_interval;

        enum class JobState {
            pending,
            queued,
            executing
        };


        struct JobExecuteState {
            // A condition variable as an approximation of a thread synchronization event
            std::condition_variable done;
            std::mutex mtx;
            bool is_done = false;
        
            using ResultType = std::array<uint64_t, max_result_words>;
            ResultType result;
        
            // Get a pointer to the result of type T.
            template<typename T>
            T* resultPtr() {
                // Ensure T is small enough to fit in ResultType
                static_assert(sizeof(T) <= sizeof(ResultType), "value is too big to be returned by background thread");
        
                // Return a pointer to the result, casted to the desired type T
                return reinterpret_cast<T*>(result.data());
            }
        
            // Helper to notify when the job is done (using a condition variable)
            void notify_done() {
                std::unique_lock<std::mutex> lock(mtx);
                is_done = true;
                done.notify_all();
            }
        
            // Helper to wait until the job is done
            void wait_until_done() {
                std::unique_lock<std::mutex> lock(mtx);
                done.wait(lock, [this]() { return is_done; });
            }
        };


        // A job represents something which potentially could be executed on a different thread.
        // The jobs form a doubly-linked list: You call `push` to append a job and `pop` to remove it.
        struct Job {
            using JobHandler = std::function<void(Task*, Job*)>;
        
            JobHandler *handler;
            Job *prev_or_null;
            void *next_or_state;
        
            // Returns a new job which can be used for the head of a list.
            constexpr static inline Job head() noexcept {
                return Job{
                    nullptr,
                    nullptr,
                    nullptr
                };
            }
        
            // Returns a pending job.
            constexpr inline static Job pending() noexcept {
                return Job{
                    nullptr,
                    nullptr,
                    nullptr,
                };
            }
        
            // Determines the current state of the job.
            constexpr inline JobState state() const noexcept {
                if (handler == nullptr) { return JobState::pending; }
                if (prev_or_null != nullptr) { return JobState::queued; }
                return JobState::executing;
            }
        
            // Checks if this is the tail of the list.
            constexpr inline bool isTail() const noexcept {
                return next_or_state == nullptr;
            }
        
            // Gets the JobExecuteState if the job is executing.
            constexpr inline JobExecuteState* getExecuteState() const noexcept {
                ASSERT(state() == JobState::State::executing);
                ASSERT(next_or_state);
                return (JobExecuteState*)next_or_state;
            }
        
            // Sets the execution state of the job.
            constexpr inline void setExecuteState(JobExecuteState* execute_state) noexcept {
                ASSERT(state() == JobState::State::executing);
                ASSERT(execute_state);
                next_or_state = execute_state;
            }
        
            // Pushes the job onto a stack.
            constexpr inline void push(Job** tail,
                                       JobHandler *new_handler) noexcept {
                ASSERT(state() == JobState::State::pending);
                handler = new_handler;
                
                (*tail)->next_or_state = this;  // tail->next = this
                prev_or_null = *tail;           // this->prev = tail
                next_or_state = nullptr;        // this->next = null
                *tail = this;                   // tail = this
                
                ASSERT(state() == JobState::State::queued);
            }
        
            // Pops the job from the stack.
            constexpr inline void pop(Job** tail) noexcept {
                assert(state() == JobState::State::queued);
                assert(*tail == this);
        
                const Job* prev = (Job *)prev_or_null;
                prev->next_or_state = nullptr;  // prev->next = null
                *tail = prev;                   // tail = prev
                // TODO *this = Job();                  // Reset the current job.
            }
        
            // Shifts the job to executing state if it has a next job.
            constexpr inline Job* shift() noexcept {
                if (next_or_state == nullptr) { return nullptr; }
        
                Job* job = (Job *)next_or_state;
                ASSERT(job->state() == JobState::State::queued);
        
                auto next = (Job *)(job->next_or_state);
                // Now we have: self -> job -> next.

                // If there is no `next` then it means that `tail` actually points to `job`.
                // In this case we can't remove `job` since we're not able to also update the tail.
                if (next == nullptr) {
                    return nullptr; 
                }
                
        
                next->prev_or_null = this;          // next->prev = this
                next_or_state = next;               // this->next = next
        
                job->prev_or_null = nullptr;       // job->prev = null
                job->next_or_state = nullptr;      // job->next_or_state = undefined
        
                ASSERT(job->state() == JobState::State::executing);
                return job;
            }
        };


        struct Worker {
            HeartbeatScheduler *pool;
            Job job_head = Job::head();
        
            /// A job (guaranteed to be in executing state) which other workers can pick up.
            Job *shared_job = nullptr;
        
            // Time when the job was shared, used for job prioritization
            std::size_t job_time = 0;
        
            // Heartbeat value, initially set to true to signal heartbeat action
            std::atomic<bool> heartbeat = true;
        
            // Function to begin a new task
            Task begin() {
                assert(job_head.isTail());
        
                return Task{
                    .worker = this,
                    .job_tail = &job_head
                };
            }
        
            // Function to execute a job, passing a task and the job to the job's handler
            void executeJob(Job* job) {
                Task t = begin();
                if (job->handler) {
                    (*(job->handler))(&t, job);
                }
            }
        };

        // Task struct that contains a Worker and a Job
        struct Task {
            Worker* worker;
            Job* job_tail;
        
            // Inline function to check and call heartbeat logic
            inline void tick() noexcept{
                if (worker->heartbeat.load()) {
                    worker->pool.heartbeat(worker);
                }
            }
        
            // Templated function to call a function with Task's context
            template<typename T, typename Func, typename Arg>
            inline T call(Func func, Arg arg) {
                return callWithContext(worker, job_tail, func, arg);
            }
        };


        // The following function's signature is actually extremely critical. We take in all of
        // the task state (worker, last_heartbeat, job_tail) as parameters. The reason for this
        // is that Zig/LLVM is really good at passing parameters in registers, but struggles to
        // do the same for "fields in structs". In addition, we then return the changed value
        // of last_heartbeat and job_tail.
        template <typename T, typename Func, typename Arg>
        inline static T callWithContext(Worker* worker, 
                                        Job* job_tail, 
                                        Func func, 
                                        Arg arg) noexcept {
            Task t{ worker, job_tail };
            t.tick();
            
            // Call the function with the task and argument, returning a value of type T
            return std::invoke(func, &t, arg);
        }

	public:

        // Start the thread pool with the given configuration
        HeartbeatScheduler() noexcept {
            const auto actual_count = config.background_worker_count !=0 ? 
                                      config.background_worker_count : 
                                      std::thread::hardware_concurrency();

            // Reserve space for background threads and workers
            background_threads.reserve(actual_count);
            workers.reserve(actual_count);

            // Spawn background workers
            for (std::size_t i = 0; i < actual_count; ++i) {
                background_threads.emplace_back([this]()__attribute__((always_inline)){backgroundWorker();});
            }

            // Spawn heartbeat thread
            heartbeat_thread = ThreadType([this]()__attribute__((always_inline)){heartbeatWorker();});

            // Wait for workers to be ready
            for (std::size_t i = 0; i < actual_count; ++i) {
                workers_ready.acquire();
            }
        }

        // De-initialize and stop the thread pool
        ~HeartbeatScheduler() noexcept{ 
            // Lock and signal stopping
            {
                std::lock_guard lock(mutex);
                is_stopping = true;
                job_ready.notify_all();
            }

            // Join background threads
            for (auto& thread : background_threads) {
                thread.join();
            }

            // Join heartbeat thread
			heartbeat_thread.join();

            // Clean up workers and other resources
            workers.clear();
            background_threads.clear();
        }

        // Worker background processing loop
        void backgroundWorker() noexcept {
            Worker w{.pool = this};
            bool first = true;

            // Lock and add the worker to the list
            mutex.lock();
            workers.push_back(&w);

            while (true) {
                if (is_stopping) { break; }

                if (auto job = _popReadyJob()) {
                    mutex.unlock();
                    w.executeJob(job);
                    mutex.lock();
                    continue; // Attempt to find more work
                }

                if (first) {
                    // Register worker as ready
                    workers_ready.release();
                    first = false;
                }

				// TODO?
				std::unique_lock lock(mutex);
                job_ready.wait(lock);
            }

            mutex.unlock();
        }

        // Heartbeat worker processing loop
        void heartbeatWorker() noexcept {
            std::size_t i = 0;

            while (true) {
                size_t to_sleep = heartbeat_interval;
                {
                    std::lock_guard lock(mutex);
                    if (is_stopping) { break;}

                    if (!workers.empty()) {
                        i %= workers.size();
                        workers[i]->heartbeat.store(true);
                        i += 1;
                        to_sleep /= workers.size();
                    }
                }
                
                std::this_thread::sleep_for(std::chrono::nanoseconds(to_sleep));
            }
        }

        // Create an one-off worker:
        template <typename T, typename Func, typename Arg>
        T call(Func func, Arg arg) noexcept {
            Worker worker{.pool = this};

            {
                std::lock_guard lock(mutex);
                workers.push_back(&worker);
            }
            
            Task t = worker.begin();
            const T ret = t.call(func, arg);
            
            std::lock_guard lock(mutex);
            for (uint32_t i = 0; i < workers.size(); i++) {
                if (workers[i] == &worker) {
                    workers.erase(workers.begin() + i);
                    break;
                }
            }
            return ret;
        }

        // Core heartbeat logic for workers
        void heartbeat(Worker* worker) noexcept {
            std::lock_guard lock(mutex);

            if (worker->shared_job == nullptr) {
                if (auto job = worker->job_head.shift()) {
                    // Allocate an execute state for the job
                    // TODO memory allocator
                    auto exec_state = new JobExecuteState();
                    job->setExecuteState(exec_state);

                    worker->shared_job = job;
                    // TODO this seams wrong
                    worker->job_time = (size_t)time(nullptr);
                    worker->job_time += 1;
                    job_ready.notify_one(); // wake up one thread
                }
            }

            worker->heartbeat.store(false, std::memory_order_relaxed);
        }

        // Wait for a shared job to be completed
        [[nodiscard]] bool waitForJob(Worker* worker, 
                                      Job* job) noexcept {
            const auto exec_state = job->getExecuteState();

            {
                mutex.lock();

                if (worker->shared_job == job) {
                    // This is the job we attempted to share with someone else, but before someone picked it up.
                    worker->shared_job = nullptr;
                    // TODO better ways.
                    delete exec_state;
                    return false;
                }

                // Help with other work while waiting
                while (!exec_state->done.is_set()) {
                    if (auto other_job = _popReadyJob()) {
                        mutex.unlock();
                        worker->executeJob(other_job);
                        mutex.lock();
                    } else {
                        break;
                    }
                }

                mutex.unlock();
            }

            exec_state->done.wait();
            return true;
        }

        // Finds and returns a job ready for execution
        Job* _popReadyJob() noexcept {
            Worker* best_worker = nullptr;

            for (auto *other_worker : workers) {
                if (other_worker->shared_job) {
                    if (best_worker) {
                        if (other_worker->job_time < best_worker->job_time) {
                            // Pick this one instead if it's older.
                            best_worker = other_worker; 
                        }
                    } else {
                        best_worker = other_worker;
                    }
                }
            }

            if (best_worker) {
                auto job = best_worker->shared_job;
                // TODO best_worker->shared_job->reset();
                return job;
            }

            return nullptr;
        }
    };
}

#endif
#endif
