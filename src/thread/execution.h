#ifndef CRYPTANALYSISLIB_THREAD_EXECUTION_H
#define CRYPTANALYSISLIB_THREAD_EXECUTION_H

#include "helper.h"
#include "thread/heartbeat_scheduler.h"
#include "thread/steal_scheduler.h"
#include "thread/simple_scheduler.h"


namespace cryptanalysislib {

// using pool_type_ = HeartbeatScheduler<>;
// using pool_type_ = StealingScheduler<>;
using pool_type_ = SimpleScheduler;
using pool_type = pool_type_*;

namespace internal {
    /// Holds the thread pool used by par.
    inline std::shared_ptr<pool_type_> get_default_pool() noexcept {
        static std::shared_ptr<pool_type_> pool;
        static std::once_flag flag;
        std::call_once(flag, [&](){ 
            pool = std::make_shared<pool_type_>(); 
        });
        return pool;
    }
}; // end namespace cryptanalysislib::internal

// base class for the execution policy
struct execution_policy {
    // TODO verstehen warum warnung [[nodiscard]] constexpr virtual pool_type pool() const noexcept;
    // TODO verstehen warum warnung [[nodiscard]] constexpr virtual bool par_allowed() const noexcept;
};

/// A sequential policy that simply forwards to the non-policy overload.
struct sequenced_policy : public execution_policy {
    [[nodiscard]] constexpr inline pool_type pool() const noexcept {
        ASSERT("requested thread pool for seq policy.");
        return nullptr;
    }

    [[nodiscard]] constexpr inline bool par_allowed() const noexcept {
        return false;
    }
};


/// A parallel policy that can use a user-specified thread pool 
/// or a default one.
struct parallel_policy : public execution_policy {
    parallel_policy() = default;

    /// \return 
    explicit parallel_policy(pool_type on_pool, 
                             const bool par_ok): 
        on_pool(on_pool), par_ok(par_ok) {}
    
    ///
    [[nodiscard]] constexpr inline parallel_policy on(pool_type pool) const noexcept {
        return parallel_policy{pool, par_ok};
    }

    ///
    [[nodiscard]] parallel_policy par_if(const bool call_par) const noexcept {
        return parallel_policy{on_pool, call_par};
    }

    ///
    [[nodiscard]] pool_type pool() const noexcept {
        if (on_pool != nullptr) {
            return on_pool;
        } else {
            return internal::get_default_pool().get();
        }
    }

    /// \return if parallisation is enabled
    [[nodiscard]] constexpr inline bool par_allowed() const noexcept {
        return par_ok;
    }

protected:
    pool_type on_pool = nullptr;
    bool par_ok = true;
};


/// EXPERIMENTAL: Subject to significant changes or removal.
/// Use pure threads for each operation instead of a shared thread pool.
/// 
/// Advantage:
///  - Fewer symbols (no packaged_task with its operators, destructors, vtable, etc) means smaller binary
///    which can mean a lot when there are many calls.
///  - No thread pool to manage.
/// 
/// Disadvantages:
///  - Threads are started and joined for every operation, so it is harder to amortize that cost.
///  - Barely any algorithms are supported.
struct pure_threads_policy : public execution_policy {
    /// \param num_threads can be 0, then `std::thread::hardware_concurrency`
    ///     many threads are used.
    explicit pure_threads_policy(const uint32_t num_threads, 
                                 const bool par_ok) noexcept :
        num_threads(num_threads),
        par_ok(par_ok) {}

    /// \return the number of thread available
    [[nodiscard]] constexpr inline uint32_t get_num_threads() const noexcept {
        if (num_threads == 0) {
            return std::thread::hardware_concurrency();
        }

        return num_threads;
    }

    /// \return
    [[nodiscard]] constexpr inline bool par_allowed() const noexcept {
        return par_ok;
    }

protected:
    unsigned int num_threads = 1;
    bool par_ok = true;
};

constexpr sequenced_policy seq{};
constexpr parallel_policy par{};

/// Choose parallel or sequential at runtime.
/// \param call_par Whether to use a parallel policy.
/// \return par if call_par is true, else a sequential policy (like `seq`).
constexpr inline static parallel_policy par_if(const bool call_par) noexcept {
    return parallel_policy{nullptr, call_par};
}

/// Choose parallel or sequential at runtime, with pool selection.
/// \param call_par Whether to use a parallel policy.
/// \return `par.on(pool)` if call_par is true, else a sequential 
///     policy (like `seq`).
constexpr inline static parallel_policy par_if(const bool call_par,
                                         pool_type_ &pool) {
    return parallel_policy{&pool, call_par};
}

/// \return true if the policy enforces sequential execution
template <class ExecPolicy>
#if __cplusplus > 201709L
    // TODO ExecPolicy concept
#endif
[[nodiscard]] constexpr inline static bool is_seq(const ExecPolicy& policy) noexcept {
    return !policy.par_allowed();
}


namespace internal {
    /// \return returns the number of elements in each chunk a single 
    ///     thread must process
    [[nodiscard]] inline constexpr std::size_t get_chunk_size(const std::size_t num_steps, 
                                                const uint32_t num_threads) noexcept {
        return (num_steps / num_threads) + ((num_steps % num_threads) > 0 ? 1 : 0);
    }
    
    /// \return returns the number of elements in each chunk a single 
    ///     thread must process
    template<typename Iterator>
#if __cplusplus > 201709L
    // TODO concept und dann dieses dofe typedef weg
#endif
    constexpr typename std::iterator_traits<Iterator>::difference_type
    get_chunk_size(const Iterator &first, 
                   const Iterator &last,
                   const uint32_t num_threads) noexcept {
        using diff_t = typename std::iterator_traits<Iterator>::difference_type;
        return static_cast<diff_t>(get_chunk_size((std::size_t)std::distance(first, last), num_threads));
    }
   
    /// min between 
    template<typename Iterator>
#if __cplusplus > 201709L
    // TODO concept und dann dieses dofe typedef weg
#endif
    constexpr typename std::iterator_traits<Iterator>::difference_type
    get_iter_chunk_size(const Iterator& iter,
                        const Iterator& last,
                        const typename std::iterator_traits<Iterator>::difference_type chunk_size) noexcept {
        return std::min(chunk_size, std::distance(iter, last));
    }
    
    template<typename Iterator>
#if __cplusplus > 201709L
    // TODO concept und dann dieses dofe typedef weg
#endif
    constexpr static Iterator advanced(Iterator iter, typename std::iterator_traits<Iterator>::difference_type offset) noexcept {
        Iterator ret = iter;
        std::advance(ret, offset);
        return ret;
    }

    /// waits for everything
    template <class Container>
#if __cplusplus > 201709L
    // TODO is iteratable
#endif
    static inline void get_futures(Container& futures) noexcept {
        for (auto &future: futures) {
            future.get();
        }
    }

    /**
  * An iterator wrapper that calls std::future<>::get().
  * @tparam Iterator
  */
    template<typename Iterator>
    class getting_iter : public Iterator {
    public:
        using value_type = decltype((*std::declval<Iterator>()).get());
        using difference_type = typename std::iterator_traits<Iterator>::difference_type;
        using pointer = value_type*;
        using reference = value_type&;
        explicit getting_iter(Iterator iter) : iter(iter) {}

        getting_iter operator++() { ++iter; return *this; }
        getting_iter operator++(int) { getting_iter ret(*this); ++iter; return ret; }

        value_type operator*() { return (*iter).get(); }
        value_type operator[](difference_type offset) { return iter[offset].get(); }

        bool operator==(const getting_iter<Iterator> &other) const { return iter == other.iter; }
        bool operator!=(const getting_iter<Iterator> &other) const { return iter != other.iter; }

    protected:
        Iterator iter;
    };

    template<typename Iterator>
    getting_iter<Iterator> get_wrap(Iterator iter) noexcept {
        return getting_iter<Iterator>(iter);
    }


    template <class ExecPolicy,
              class RandIt,
              class Chunk,
              class ChunkRet,
              typename... A>
    void parallel_chunk_for_1_wait(ExecPolicy &&policy,
                                   RandIt first,
                                   RandIt last,
                                   Chunk chunk,
                                   ChunkRet*,
                                   const uint32_t extra_split_factor,
                                   const uint32_t nthreads,
                                   A&&... chunk_args) noexcept {
        std::vector<std::thread> threads;
        const uint32_t t = nthreads == 0 ? std::thread::hardware_concurrency() : nthreads;
        auto chunk_size = get_chunk_size(first, last, extra_split_factor * t);

        // TODO
        (void)policy;

        while (first < last) {
            auto iter_chunk_size = get_iter_chunk_size(first, last, chunk_size);
            RandIt loop_end = advanced(first, iter_chunk_size);

            threads.emplace_back(std::thread(chunk, first, loop_end, chunk_args...));

            first = loop_end;
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }


    ///Chunk a single range.
    template <class ExecPolicy,
              class RandIt,
              class Chunk,
              class ChunkRet,
              typename... A>
    std::vector<std::future<ChunkRet>>
    parallel_chunk_for_1(ExecPolicy &&policy,
                         RandIt first,
                         RandIt last,
                         Chunk chunk, ChunkRet*,
                         const int extra_split_factor,
                         const uint32_t nthreads,
                         A&&... chunk_args) noexcept {
        std::vector<std::future<ChunkRet>> futures;
        auto& task_pool = *policy.pool();
        const uint32_t t = nthreads == 0 ? task_pool.get_num_threads() : nthreads;
        auto chunk_size = get_chunk_size(first, last, extra_split_factor * t);

        while (first < last) {
            auto iter_chunk_size = get_iter_chunk_size(first, last, chunk_size);
            RandIt loop_end = advanced(first, iter_chunk_size);

            futures.emplace_back(task_pool.enqueue(
                chunk, first, loop_end, chunk_args...));

            first = loop_end;
        }

        return futures;
    }

    ///Element-wise chunk two ranges.
    template <class ExecPolicy, 
              class RandIt1,
              class RandIt2, 
              class Chunk, 
              class ChunkRet, 
              typename... A>
    std::vector<std::future<ChunkRet>>
    parallel_chunk_for_2(ExecPolicy &&policy, 
                         RandIt1 first1, 
                         RandIt1 last1, 
                         RandIt2 first2,
                         Chunk chunk, 
                         ChunkRet*, 
                         A&&... chunk_args,
                         const uint32_t nthreads=0) noexcept {
        std::vector<std::future<ChunkRet>> futures;
        auto& task_pool = *policy.pool();
        const uint32_t t = nthreads == 0 ? task_pool.get_num_threads() : nthreads;
        auto chunk_size = get_chunk_size(first1, last1, t);

        while (first1 < last1) {
            auto iter_chunk_size = get_iter_chunk_size(first1, last1, chunk_size);
            RandIt1 loop_end = advanced(first1, iter_chunk_size);
            futures.emplace_back(task_pool.enqueue(chunk,
                                                  first1, 
                                                  loop_end, 
                                                  first2, 
                                                  chunk_args...));

            first1 = loop_end;
            std::advance(first2, iter_chunk_size);
        }

        return futures;
    }



}; // end namespace internal

}; // end namespace cryptanalysislib
#endif 
