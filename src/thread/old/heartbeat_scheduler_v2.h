#ifndef CRYPTANALYSISLIB_THREAD_SIMPLESCHEDULER_H
#define CRYPTANALYSISLIB_THREAD_SIMPLESCHEDULER_H

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>

// translation of: https://github.com/dragostis/chili/blob/main/src/lib.rs#L284

namespace cryptanalysislib {
    // forward declaration
    template<typename T>
    class Scope;
    class ThreadPool;


    template <typename F = void(*)()>
    class JobStack {
    private:
        // Encapsulates the stored callable and provides manual control over its lifetime
        struct ManuallyDrop {
            F func;
            bool taken = false;

            explicit ManuallyDrop(F&& f) : func(std::move(f)) {}

            ManuallyDrop(const ManuallyDrop&) = delete;
            ManuallyDrop& operator=(const ManuallyDrop&) = delete;

            ManuallyDrop(ManuallyDrop&& other) noexcept
                : func(std::move(other.func)), taken(other.taken) {
                other.taken = true;
            }

            ManuallyDrop& operator=(ManuallyDrop&& other) noexcept {
                if (this != &other) {
                    func = std::move(other.func);
                    taken = other.taken;
                    other.taken = true;
                }
                return *this;
            }

            inline F take() noexcept {
                assert(!taken && "Function already taken");
                taken = true;
                return std::move(func);
            }

            ~ManuallyDrop() = default; // Manages destruction automatically
        };

        mutable ManuallyDrop func_;

    public:
        explicit JobStack(F&& func) : func_(std::move(func)) {}

        // SAFETY:
        // This method should only be called once.
        F take_once() const {
            // Ensures the callable is taken only once
            return func_.take();
        }
    };


	///
    /// @tparam T should be funciton type
    template <typename T = void(*)()>
    class Job {
    public:
        using FutureType = std::future<T>;

    private:

        std::shared_ptr<JobStack<T>> stack_;
        std::atomic<FutureType*> fut_{nullptr};

    public:
        template <typename F>
        requires std::is_invocable_r_v<T, F, Scope*> && std::is_move_constructible_v<F>
        static Job create(F&& func) {
            auto stack = std::make_shared<JobStack<F>>(std::forward<F>(func));
            return Job(stack);
        }

        bool is_waiting() const {
            return fut_.load() == nullptr;
        }

        bool equals(const Job& other) const {
            return stack_ == other.stack_;
        }

        void set_future(FutureType* future) {
            fut_.store(future, std::memory_order_relaxed);
        }

        FutureType* get_future() const {
            return fut_.load(std::memory_order_relaxed);
        }

        bool poll() const {
            auto fut = get_future();
            if (!fut) {
                return false;
            }

            auto status = fut->wait_for(std::chrono::seconds(0));
            return status == std::future_status::ready;
        }

        std::optional<T> wait() {
            auto fut = get_future();
            if (!fut) {
                return std::nullopt;
            }

            auto result = fut->get();
            delete fut; // Cleanup future after it has been waited upon
            fut_.store(nullptr, std::memory_order_relaxed);
            return result;
        }

        void drop() {
            auto fut = get_future();
            if (fut) {
                delete fut; // Ensure future is cleaned up
                fut_.store(nullptr, std::memory_order_relaxed);
            }
        }

        void execute(Scope<T>* scope) {
            assert(stack_);
            stack_->execute(scope);
        }

    private:
        explicit Job(std::shared_ptr<JobStack<T>> stack) : stack_(std::move(stack)) {}
    };


    template<typename T>
    using JobQueue = std::deque<Job<T>>;

    using ThreadJobQueue = JobQueue;
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;

    class ThreadPool;

    struct Heartbeat {
        /// TODO weak probably not needed?
        /// Das Problem ist, dass Arc keine kreise in den referencen erlaubt
        /// deswegen mussdas mit weak aufgebrochen werdn
        std::weak_ptr<std::atomic<bool>> is_set;
        TimePoint last_heartbeat;
    };

	///
    /// @tparam T
    template<typename T>
    class LockContext {
    public:
        LockContext() :
            time(0), is_stopping(false), heartbeat_index(0) {}

        std::shared_ptr<std::atomic<bool>> new_heartbeat() noexcept {
            auto is_set = std::make_shared<std::atomic<bool>>(true);
            Heartbeat heartbeat{
                .is_set = std::weak_ptr(is_set),
                .last_heartbeat = Clock::now()
            };

            heartbeats[heartbeat_index] = heartbeat;
            heartbeat_index += 1;

            return is_set;
        }

        std::optional<Job<T>> pop_earliest_shared_job() noexcept {
            if (shared_jobs.empty()) {
                return std::nullopt;
            }

            auto it = shared_jobs.begin();
            auto job = it->second.second;
            shared_jobs.erase(it);
            return job;
        }

        uint64_t time;
        bool is_stopping;
        std::map<size_t, std::pair<uint64_t, Job<T>>> shared_jobs;
        std::map<uint64_t, Heartbeat> heartbeats;
        uint64_t heartbeat_index;
    };

    template<typename T>
    struct Context {
        std::mutex lock;
        LockContext<T> data;
        std::condition_variable job_is_ready;
        std::condition_variable scope_created_from_thread_pool;
    };

    template<typename T>
    void execute_worker(std::shared_ptr<Context<T>> context) {
        JobQueue job_queue;

        while (true) {
            std::shared_ptr<Job<T>> job;
            {
                std::unique_lock<std::mutex> lk(context->lock);
                job = context->lock_context->pop_earliest_shared_job();
                if (job) {
                    // Execute job
                    job->execute();
                }
                if (context->lock_context->is_stopping) break;
                context->job_is_ready.wait(lk);
            }
        }
    }

    template<typename T>
    void execute_heartbeat(std::shared_ptr<Context<T>> context,
                           Duration heartbeat_interval,
                           size_t num_workers) noexcept {
        while (true) {
            {
                std::unique_lock<std::mutex> lk(context->lock);
                if (context->lock_context->is_stopping) break;

                auto now = std::chrono::steady_clock::now();
                for (auto it = context->lock_context->heartbeats.begin(); it != context->lock_context->heartbeats.end();) {
                    if (auto is_set = it->second.is_set.lock()) {
                        if (now - it->second.last_heartbeat >= heartbeat_interval) {
                            is_set->store(true);
                            it->second.last_heartbeat = now;
                        }
                    } else {
                        it = context->lock_context->heartbeats.erase(it);
                    }
                }
            }
            std::this_thread::sleep_for(heartbeat_interval);
        }
    }

    class ThreadPool {
    public:
        ThreadPool(size_t thread_count, std::chrono::microseconds heartbeat_interval) {
            for (size_t i = 0; i < thread_count; ++i) {
                workers.emplace_back(execute_worker, context);
            }
            heartbeat_thread = std::thread(execute_heartbeat, context, heartbeat_interval, thread_count);
        }

        ~ThreadPool() {
            {
                std::lock_guard<std::mutex> lock(context->lock);
                context->data.is_stopping = true;
            }

            context->job_is_ready.notify_all();
            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }

            if (heartbeat_thread.joinable()) {
                heartbeat_thread.join();
            }
        }
        // TODO
        using T = uint64_t;

        std::shared_ptr<JobQueue<T>> job_queue{};

        std::shared_ptr<Context<T>> context;
        std::vector<std::thread> workers;
        std::thread heartbeat_thread;
    };

    /// A `Scope`d object that you can run fork-join workloads on.
    /// @tparam T
    template<typename T>
    class Scope {
    public:
        static Scope global() {
            auto global_thread_pool = ThreadPool::global();
            return global_thread_pool.scope();
        }

        Scope(ThreadPool &p)
        : context_(p.) {

        }

        // new from workter
        Scope(std::shared_ptr<Context<T>> context, ThreadJobQueue job_queue)
            : context_(std::move(context)),
              job_queue_(std::move(job_queue)),
              heartbeat_(context_->new_heartbeat()),
              join_count_(0) {}

        template <typename A, typename B, typename RA, typename RB>
        std::pair<RA, RB> join(A a, B b) {
            return join_with_heartbeat_every<64, A, B, RA, RB>(std::move(a), std::move(b));
        }

        template <uint8_t TIMES, typename A, typename B, typename RA, typename RB>
        std::pair<RA, RB> join_with_heartbeat_every(A a, B b) {
            join_count_ = (join_count_ + 1) % TIMES;

            if (join_count_ == 0 || job_queue_.len() < 3) {
                return join_heartbeat<A, B, RA, RB>(std::move(a), std::move(b));
            } else {
                return join_seq<A, B, RA, RB>(std::move(a), std::move(b));
            }
        }

    private:
        std::shared_ptr<Context<T>> context_;
        ThreadJobQueue job_queue_;
        std::shared_ptr<std::atomic<bool>> heartbeat_;
        uint8_t join_count_;

        template <typename A, typename B, typename RA, typename RB>
        std::pair<RA, RB> join_seq(A a, B b) {
            auto rb = b(this);
            auto ra = a(this);

            return {ra, rb};
        }

        template <typename A, typename B, typename RA, typename RB>
        std::pair<RA, RB> join_heartbeat(A a, B b) {
            auto job_stack = std::make_shared<JobStack<A>>(std::move(a));
            auto job = std::make_shared<Job<>>(job_stack);

            job_queue_.push_back(job);

            auto rb = b(this);

            if (job->is_waiting()) {
                job_queue_.pop_back();
                return {job_stack->take_once()(this), rb};
            } else {
                auto ra = wait_for_sent_job<RA>(job);
                return {ra.value_or(job_stack->take_once()(this)), rb};
            }
        }

        std::optional<T> wait_for_sent_job(const std::shared_ptr<Job<T>>& job) {
            // Placeholder for waiting on a sent job
            return std::nullopt;
        }
    };

    class HeartBeatScheduler_v2 {
    public:

    };
}

#endif
