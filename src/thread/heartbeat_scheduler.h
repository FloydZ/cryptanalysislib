#ifndef CRYPTANALYSISLIB_HEARTBEAT_SCHEDULER_H
#define CRYPTANALYSISLIB_HEARTBEAT_SCHEDULER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <deque>
#include <thread>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <variant>
#include <cassert>

#include "thread/scheduler.h"

namespace cryptanalysislib {
	class HeartbeatSchedulerConfig {
	public:
		/// The number of background workers. If `null` this chooses a sensible
		/// default based on your system (i.e. number of cores).
		constexpr static uint32_t background_worker_count = 0;
		/// How often a background thread is interrupted to find more work.
		constexpr static size_t heartbeat_interval = 100 * 1000;// NS_PERD_US 1000L
	} heartbeatSchedulerConfig;


	///
	/// \tparam ThreadType
	/// \tparam FunctionType
	/// \tparam Mutex
	/// \tparam config
	template<typename ThreadType = std::jthread,
	         typename FunctionType = details::default_function_type,
	         typename Mutex = std::mutex,
	         const HeartbeatSchedulerConfig &config = heartbeatSchedulerConfig>
#if __cplusplus > 201709L
	    requires std::invocable<FunctionType> &&
	             std::is_same_v<void, std::invoke_result_t<FunctionType>>
#endif
	class HeartbeatScheduler {
	public:
		/// forward decleration
		struct Task;
		struct Worker;

	private:
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
		std::condition_variable job_ready{};

		/// This is used to wait for the background workers to be available initially.
		// workers_ready: std.Thread.Semaphore = .{},
		std::counting_semaphore<config.background_worker_count> workers_ready{0};

		/// This is set to true once we're trying to stop.
		bool is_stopping = false;

		constexpr static std::size_t max_result_words = 4;
		constexpr static uint32_t background_worker_count = config.background_worker_count;
		constexpr static size_t heartbeat_interval = config.heartbeat_interval;

	public:
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
			T *resultPtr() {
				// Ensure T is small enough to fit in ResultType
				static_assert(sizeof(T) <= sizeof(ResultType), "value is too big to be returned by background thread");

				// Return a pointer to the result, cast to the desired type T
				return reinterpret_cast<T *>(result.data());
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
			using JobHandler = std::function<void(Task *, Job *)>;

			JobHandler *handler;
			Job *prev_or_null;
			void *next_or_state;

			// Returns a new job which can be used for the head of a list.
			constexpr static inline Job head() noexcept {
				return Job{
				        nullptr,
				        nullptr,
				        nullptr};
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
			constexpr inline JobExecuteState *getExecuteState() const noexcept {
				ASSERT(state() == JobState::State::executing);
				ASSERT(next_or_state);
				return (JobExecuteState *) next_or_state;
			}

			// Sets the execution state of the job.
			constexpr inline void setExecuteState(JobExecuteState *execute_state) noexcept {
				ASSERT(state() == JobState::executing);
				ASSERT(execute_state);
				next_or_state = execute_state;
			}

			// Pushes the job onto a stack.
			constexpr inline void push(Job **tail,
			                           JobHandler *new_handler) noexcept {
				ASSERT(state() == JobState::State::pending);
				handler = new_handler;

				(*tail)->next_or_state = this;// tail->next = this
				prev_or_null = *tail;         // this->prev = tail
				next_or_state = nullptr;      // this->next = null
				*tail = this;                 // tail = this

				ASSERT(state() == JobState::State::queued);
			}

			// Pops the job from the stack.
			constexpr inline void pop(Job **tail) noexcept {
				assert(state() == JobState::State::queued);
				assert(*tail == this);

				const Job *prev = (Job *) prev_or_null;
				prev->next_or_state = nullptr;// prev->next = null
				*tail = prev;                 // tail = prev
				                              // TODO *this = Job();                  // Reset the current job.
			}

			// Shifts the job to executing state if it has a next job.
			constexpr inline Job *shift() noexcept {
				if (next_or_state == nullptr) { return nullptr; }

				Job *job = (Job *) next_or_state;
				ASSERT(job->state() == JobState::queued);

				auto next = (Job *) (job->next_or_state);
				// Now we have: self -> job -> next.

				// If there is no `next` then it means that `tail` actually points to `job`.
				// In this case we can't remove `job` since we're not able to also update the tail.
				if (next == nullptr) {
					return nullptr;
				}


				next->prev_or_null = this;// next->prev = this
				next_or_state = next;     // this->next = next

				job->prev_or_null = nullptr; // job->prev = null
				job->next_or_state = nullptr;// job->next_or_state = undefined

				ASSERT(job->state() == JobState::executing);
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
				        .job_tail = &job_head};
			}

			// Function to execute a job, passing a task and the job to the job's handler
			void executeJob(Job *job) {
				Task t = begin();
				if (job->handler) {
					(*(job->handler))(&t, job);
				}
			}
		};

		// Task struct that contains a Worker and a Job
		struct Task {
			Worker *worker;
			Job *job_tail;

			// Inline function to check and call heartbeat logic
			inline void tick() noexcept {
				if (worker->heartbeat.load()) {
					worker->pool->heartbeat(worker);
				}
			}

			// Templated function to call a function with Task's context
			template<typename T,
			         typename Func,
			         typename Arg>
			inline T call(Func &func, Arg &arg) noexcept __attribute__((always_inline)) {
				return callWithContext<T, Func, Arg>(worker, job_tail, func, arg);
			}
		};


		// The following function's signature is actually extremely critical. We take in all
		// the task state (worker, last_heartbeat, job_tail) as parameters. The reason for this
		// is that Zig/LLVM is really good at passing parameters in registers, but struggles to
		// do the same for "fields in structs". In addition, we then return the changed value
		// of last_heartbeat and job_tail.
		template<typename T, typename Func, typename Arg>
		inline static T callWithContext(Worker *worker,
		                                Job *job_tail,
		                                Func &func,// TODO correct reference
		                                Arg arg) noexcept __attribute__((always_inline)) {
			Task t{worker, job_tail};
			t.tick();
			// TODO
			// return std::invoke(func, &t, arg);
			return std::invoke(func, arg);
		}

		// Start the thread pool with the given configuration
		HeartbeatScheduler() noexcept {
			const uint32_t actual_count = background_worker_count != 0 ? background_worker_count : std::thread::hardware_concurrency();

			// Reserve space for background threads and workers
			background_threads.reserve(actual_count);
			workers.reserve(actual_count);

			// Spawn background workers
			for (std::size_t i = 0; i < actual_count; ++i) {
				background_threads.emplace_back([this]() __attribute__((always_inline)) { backgroundWorker(); });
				//backgroundWorker();
			}

			// Spawn heartbeat thread
			heartbeat_thread = ThreadType([this]() __attribute__((always_inline)) { heartbeatWorker(); });

			// Wait for workers to be ready
			for (std::size_t i = 0; i < actual_count; ++i) {
				workers_ready.acquire();
			}
		}

		// De-initialize and stop the thread pool
		~HeartbeatScheduler() noexcept {
			// Lock and signal stopping
			{
				std::lock_guard lock(mutex);
				is_stopping = true;
				job_ready.notify_all();
			}

			// Join background threads
			for (auto &thread: background_threads) {
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
			mutex.unlock();

			while (true) {
				if (is_stopping) { break; }

				if (first) [[unlikely]] {
					// Register worker as ready
					workers_ready.release();
					first = false;
				}

				if (auto job = _popReadyJob()) {
					mutex.unlock();
					w.executeJob(job);
					mutex.lock();
					continue;// Attempt to find more work
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
					if (is_stopping) { break; }

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
		template<typename T, typename Func, typename Arg>
		inline T call(Func &func, Arg arg) noexcept __attribute__((always_inline)) {
			Worker worker{.pool = this};

			{
				std::lock_guard lock(mutex);
				workers.push_back(&worker);
			}

			Task t = worker.begin();
			const T ret = t.template call<T, Func, Arg>(func, arg);

			// clean up stuff
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
		void heartbeat(Worker *worker) noexcept {
			std::lock_guard lock(mutex);

			if (worker->shared_job == nullptr) {
				if (auto job = worker->job_head.shift()) {
					// Allocate an execute state for the job
					// TODO memory allocator
					auto exec_state = new JobExecuteState();
					job->setExecuteState(exec_state);

					worker->shared_job = job;
					// TODO this seams wrong
					worker->job_time = (size_t) time(nullptr);
					worker->job_time += 1;
					job_ready.notify_one();// wake up one thread
				}
			}

			worker->heartbeat.store(false, std::memory_order_relaxed);
		}

		// Wait for a shared job to be completed
		[[nodiscard]] bool waitForJob(Worker *worker,
		                              Job *job) noexcept {
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
		Job *_popReadyJob() noexcept {
			Worker *best_worker = nullptr;

			for (auto *other_worker: workers) {
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

		// Destroy a job's execute state
		void destroyExecuteState(JobExecuteState *exec_state) noexcept {
			// not needed, as this is done curretnly with `new`
			// std::lock_guard lock(mutex);
			ASSERT(exec_state);
			delete exec_state;
		}

		// `Future` function template to create a type that handles asynchronous execution.
		template<typename Input, typename Output>
		struct Future {
			Job job;                   // Represents the job associated with the future
			std::optional<Input> input;// Input passed to the asynchronous job

			// Initializes the `Future` with a pending job
			static Future init() {
				return Future{Job::pending(), std::nullopt};
			}

			// Schedules work to be executed by another thread.
			// After calling this, `join` or `tryJoin` must be called to get the result.
			template<typename Func>
			void fork(Task *task, Func func, Input input_value) {
				// Define the handler as a lambda
				auto handler = [this, func](Task *t, Job *job) {
					auto fut = this;// Get a pointer to this future instance
					auto exec_state = job->getExecuteState();
					// Call the provided function and store the result
					Output value = t->call(func, fut->input.value());
					exec_state->template resultPt<Output>() = value;
					exec_state->done.set();
				};

				// Store the input and push the job into the task queue
				input = input_value;
				job.push(&task->job_tail, handler);
			}

			// Waits for the result of `fork`.
			// Safe to call only if `fork` was actually called.
			std::optional<Output> join(Task *task) {
				assert(job.state() != JobState::Pending);
				return tryJoin(task);
			}

			// Waits for the result of `fork`.
			// This function is safe to call even if `fork` wasn't called.
			std::optional<Output> tryJoin(Task *task) {
				switch (job.state()) {
					case JobState::Pending:
						return std::nullopt;

					case JobState::Queued:
						job.pop(&task->job_tail);
						return std::nullopt;

					case JobState::Executing:
						return joinExecuting(task);

					default:
						return std::nullopt;
				}
			}

		private:
			// Helper function to handle the execution of the job and wait for its result
			std::optional<Output> joinExecuting(Task *task) {
				const Worker *worker = task->worker;
				HeartbeatScheduler<> *pool = worker->pool;
				JobExecuteState *exec_state = job.getExecuteState();

				if (pool->waitForJob(worker, &job)) {
					Output result = exec_state->result();
					pool->destroyExecuteState(exec_state);
					return result;
				}

				return std::nullopt;
			}
		};
	};
}
#endif//CRYPTANALYSISLIB_HEARTBEAT_SCHEDULER_H
