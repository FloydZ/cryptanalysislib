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

// translation of:https://github.com/dragostis/chili/blob/main/src/lib.rs#L284

namespace cryptanalysislib {
    template<typename T>
    class Job {

    };

    template<typename T>
    using JobQueue = std::deque<Job<T>>;


    using ThreadJobQueue = JobQueue;
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;

    class ThreadPool;

    struct Heartbeat {
        std::weak_ptr<std::atomic<bool>> is_set;
        TimePoint last_heartbeat;
    };

    template<typename T>
    class LockContext {
    public:
        LockContext() :
            time(0), is_stopping(false), heartbeat_index(0) {}

        std::shared_ptr<std::atomic<bool>> new_heartbeat() {
            auto is_set = std::make_shared<std::atomic<bool>>(true);
            Heartbeat heartbeat{
                .is_set = std::weak_ptr<std::atomic<bool>>(is_set),
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
    class Scope {
    private:
        std::shared_ptr<Context<T>> context;
        ThreadJobQueue *job_queue;
        std::shared_ptr<std::atomic<bool>> heartbeat;
        uint8_t join_count=0;
    public:
        Scope(std::shared_ptr<Context<T>> context, ThreadJobQueue *job_queue) :
            context(std::move(context)),
            job_queue(job_queue) {

            std::lock_guard lock(context->lock);
            heartbeat = context->data.new_heartbeat();
        }

        Scope(ThreadPool &pool) {
            // TODO
        }

        //Scope(std::shared_ptr<Context> context, JobQueue *job_queue {
        //    // TODO
        //}

        size_t hearbeat_id() noexcept {
            return (size_t)(heartbeat.get());
        }
        template<typename T>
        std::optional<std::future<T>> wait_for_sent_job(Job<T> &job) {
            std::lock_guard lock(context->lock);
            const auto hid = hearbeat_id();
            if (context->data.shared_jobs.contains()(hid)) {
                context->data.shared_jobs.erase(hid);
            }


        }
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

    private:
        std::shared_ptr<Context<T>> context;
        std::vector<std::thread> workers;
        std::thread heartbeat_thread;
    };

    class HeartBeatScheduler_v2 {
    public:

    };
}

#endif
