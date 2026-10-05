#ifndef CRYPTANALYSISLIB_THREAD_SIMPLESCHEDULER_H
#define CRYPTANALYSISLIB_THREAD_SIMPLESCHEDULER_H

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

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
    class SimpleScheduler {
	private:
		constexpr static bool enable_try_block = config.enable_try_block;
		constexpr static bool enable_remote_view = config.enable_remote_view;
		SchedulerPerformanceManager *schedulerPerformance;

    public:
        /// Create a SimpleScheduler and start worker threads.
        /// \param num_threads Number of worker threads. If 0 then number of 
        ///     threads is equal to the number of physical cores on the machine, 
        ///     as given by std::thread::hardware_concurrency().
        explicit SimpleScheduler(uint32_t num_threads = 0) noexcept {
            if (num_threads == 0) {
                num_threads = std::thread::hardware_concurrency();
                if (num_threads < 1) { num_threads = 1; }
            
            }
			if constexpr (enable_remote_view) {
				schedulerPerformance = new SchedulerPerformanceManager{false};
				schedulerPerformance->resize(num_threads);
			}

            start_threads(num_threads);
        }

        /// Finish all tasks left in the queue then shut down worker threads.
        /// If the pool is currently paused then it is resumed.
        ~SimpleScheduler() noexcept {
            unpause();
            wait_for_queued_tasks();
            stop_all_threads();
        }

        /// Submit a Callable for the pool to execute and return a std::future.
        /// \param func The Callable to execute. Can be a function, a lambda, std::packaged_task, std::function, etc.
        /// \param args Arguments for func. Optional.
        /// \return std::future that can be used to get func's return value or thrown exception.
        template <typename F, 
                  typename... A,
                  typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>
        >
        [[nodiscard]] std::future<R> submit(F&& func, A&&... args) {
            std::packaged_task<R()> task(std::bind(std::forward<F>(func), std::forward<A>(args)...));
            auto ret = task.get_future();
            submit_detach(std::move(task));
            return ret;
        }

        /// Submit a zero-argument Callable for the pool to execute.
        /// \param func The Callable to execute. Can be a function, a lambda, 
        /// std::packaged_task, std::function, etc.
        template <typename F>
        void submit_detach(F&& func) {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            tasks.emplace(std::forward<F>(func));
            task_cv.notify_one();
        }

        /// Submit a Callable with arguments for the pool to execute.
        /// /param func The Callable to execute. Can be a function, a lambda,
        /// std::packaged_task, std::function, etc.
        template <typename F, typename... A>
        void submit_detach(F&& func, A&&... args) {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            tasks.emplace(std::bind(std::forward<F>(func), std::forward<A>(args)...));
            task_cv.notify_one();
        }

        /// Block until the task queue is empty.
        /// Some tasks may be in-progress when this method returns.
        void wait_for_queued_tasks() {
            std::unique_lock<std::mutex> tasks_lock(task_mutex);
            notify_task_finish = true;
            task_finished_cv.wait(tasks_lock, [&] { return tasks.empty(); });
            notify_task_finish = false;
        }

        /// Block until all tasks have finished.
        void wait_for_tasks() {
            std::unique_lock<std::mutex> tasks_lock(task_mutex);
            notify_task_finish = true;
            task_finished_cv.wait(tasks_lock, [&] { return tasks.empty() && num_inflight_tasks == 0; });
            notify_task_finish = false;
        }

        /// Stop executing queued tasks. Use `unpause()` to resume. Note: 
        /// Destroying the pool will implicitly unpause.
        /// Any in-progress tasks continue executing.
        void pause() {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            pool_paused = true;
        }

        /// Resume executing queued tasks.
        void unpause() {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            pool_paused = false;
            task_cv.notify_all();
        }

        /// Check whether the pool is paused.
        /// \return true if pause() has been called without an intervening unpause().
        [[nodiscard]] bool is_paused() const {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            return pool_paused;
        }

        /// Drop all tasks that have been submitted but not yet started by a worker.
        /// Tasks already in progress continue executing.
        void clear_tasks() noexcept {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            tasks = {};
        }

        /// Get number of enqueued tasks.
        /// \return Number of tasks that have been enqueued but not yet started.
        [[nodiscard]] size_t get_num_queued_tasks() const noexcept {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            return tasks.size();
        }

        /// Get number of in-progress tasks.
        /// \return Approximate number of tasks currently being processed by 
        /// worker threads.
        [[nodiscard]] size_t get_num_running_tasks() const noexcept  {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            return num_inflight_tasks;
        }

        /// Get total number of tasks in the pool.
        /// \return Approximate number of tasks both enqueued and running.
        [[nodiscard]] size_t get_num_tasks() const noexcept {
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            return tasks.size() + num_inflight_tasks;
        }

        /// Get number of worker threads.
        /// \return Number of worker threads.
        [[nodiscard]] unsigned int get_num_threads() const noexcept {
            const std::lock_guard<std::recursive_mutex> threads_lock(thread_mutex);
            return static_cast<unsigned int>(threads.size());
        }

        /// Set number of worker threads. Will start or stop worker threads as necessary.
        /// \param num_threads Number of worker threads. If 0 then number of 
        ///  threads is equal to the number of physical cores on the machine, 
        ///  as given by std::thread::hardware_concurrency().
        /// \return previous number of threads
        [[nodiscard]] uint32_t set_num_threads(uint32_t num_threads) noexcept {
            const std::lock_guard<std::recursive_mutex> threads_lock(thread_mutex);
            unsigned int previous_num_threads = get_num_threads();

            if (num_threads < 1) {
                num_threads = std::thread::hardware_concurrency();
            }

            if (previous_num_threads <= num_threads) {
                // expanding the thread pool
                start_threads(num_threads - previous_num_threads);
            } else {
                // contracting the thread pool
                stop_all_threads();
                {
                    const std::lock_guard<std::mutex> tasks_lock(task_mutex);
                    pool_running = true;
                }
                start_threads(num_threads);
            }

            return previous_num_threads;
        }

    protected:

        /// Main function for worker threads.
        void worker_main(const uint32_t id) noexcept {
            bool finished_task = false;

            while (true) {
                std::unique_lock<std::mutex> tasks_lock(task_mutex);

                if (finished_task) {
                    --num_inflight_tasks;
                    if (notify_task_finish) {
                        task_finished_cv.notify_all();
                    }
                }

                task_cv.wait(tasks_lock, [&]() {
                    return !pool_running || (!pool_paused && !tasks.empty());
                });

                if (!pool_running) {
                    break;
                }

				if constexpr (enable_remote_view) {
					// not nice but easy
					if (id == 0) {
						schedulerPerformance->gather(get_num_running_tasks(),
						                             get_num_queued_tasks());
					} else {
						schedulerPerformance->gather(id);
					}
				}

                // Must mean that (!pool_paused && !tasks.empty()) is true
                auto task{std::move(tasks.front())};
                tasks.pop();
                ++num_inflight_tasks;
                tasks_lock.unlock();

                // try {
                    task();
                // } catch (...) { }

                finished_task = true;
            }
        }

        /// Start worker threads.
        /// \param num_threads How many threads to start.
        void start_threads(const unsigned int num_threads) {
            const std::lock_guard<std::recursive_mutex> threads_lock(thread_mutex);

            for (uint32_t i = 0; i < num_threads; ++i) {
                threads.emplace_back(&SimpleScheduler::worker_main, this, i);
            }
        }

        /// Stop, join, and destroy all worker threads.
        void stop_all_threads() noexcept {
            const std::lock_guard<std::recursive_mutex> threads_lock(thread_mutex);

            {
                const std::lock_guard<std::mutex> tasks_lock(task_mutex);
                pool_running = false;
                task_cv.notify_all();
            }

            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }

            threads.clear();
        }

        /// The worker threads.
        /// Access protected by thread_mutex
        std::vector<std::thread> threads;

        /**
         * A mutex for methods that start/stop threads.
         */
        mutable std::recursive_mutex thread_mutex;

        /// TODO: replace with `ConstVectorQueue`
        /// tasks: Access protected by task_mutex.
        std::queue<std::packaged_task<void()>> tasks = {};

        /**
         * A mutex for all variables related to tasks.
         */
        mutable std::mutex task_mutex;

        /**
         * Used to notify changes to the task queue, such as a new task added, pause/unpause, etc.
         */
        std::condition_variable task_cv;

        /**
         * Used to notify of finished tasks.
         */
        std::condition_variable task_finished_cv;

        /**
         * A signal for worker threads that the pool is either running or shutting down.
         *
         * Access protected by task_mutex.
         */
        bool pool_running = true;

        /**
         * A signal for worker threads to not pull new tasks from the queue.
         *
         * Access protected by task_mutex.
         */
        bool pool_paused = false;

        /**
         * A signal for worker threads that they should notify task_finished_cv when they finish a task.
         *
         * Access protected by task_mutex.
         */
        bool notify_task_finish = false;

        /**
         * A counter of the number of tasks in-progress by worker threads.
         * Incremented when a task is popped off the task queue and decremented when that task is complete.
         *
         * Access protected by task_mutex.
         */
        int num_inflight_tasks = 0;
    };
}

#endif
