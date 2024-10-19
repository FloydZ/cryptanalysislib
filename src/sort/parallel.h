#pragma once

#include "algorithm/for_each.h"
#include "thread/thread.h"

namespace cryptanalysislib {
    namespace internal {
        /**
         * Identify a pivot element for quicksort. Chooses the middle element of the range.
         */
        template <typename Iterator>
        typename std::iterator_traits<Iterator>::value_type
        quicksort_pivot(Iterator first,
            Iterator last) noexcept {
            return *(std::next(first, std::distance(first, last) / 2));
        }

        /**
        * Predicate for std::partition (for quicksort)
        */
        template <class Compare,
                  class T>
        struct pivot_predicate {
            /// @param comp
            /// @param pivot
            pivot_predicate(Compare comp,
                            const T& pivot) noexcept :
                comp(comp), pivot(pivot) {}

            ///
            bool operator()(const T& em) noexcept {
                return comp(em, pivot);
            }

            Compare comp;
            const T pivot;
        };

        /**
         * Partition range according to predicate. Unstable.
         *
         * This implementation only parallelizes with p=2; will spawn and wait for only one task.
         */
        template <class RandIt,
                  class Predicate>
        #if __cplusplus > 201709L
        	requires std::random_access_iterator<RandIt>
        #endif
        RandIt partition_p2(pool_type &task_pool,
                            RandIt first,
                            RandIt last,
                            Predicate pred) noexcept {
            auto range_size = std::distance(first, last);
            if (range_size < 4) {
                return std::partition(first, last, pred);
            }

            // approach should be generalizable to arbitrary p
            RandIt mid = std::next(first + range_size / 2);

            // partition left and right halves in parallel
            auto left_future = task_pool.submit(std::partition<RandIt, Predicate>, first, mid, pred);
            RandIt right_mid = std::partition(mid, last, pred);
            RandIt left_mid = left_future.get();

            // merge the two partitioned halves
            auto left_highs_size = std::distance(left_mid, mid);
            auto right_lows_size = std::distance(mid, right_mid);
            if (left_highs_size <= right_lows_size) {
                std::swap_ranges(left_mid, mid, right_mid - left_highs_size);
                return right_mid - left_highs_size;
            } else {
                std::swap_ranges(mid, right_mid, left_mid);
                return left_mid + right_lows_size;
            }
        }

    }; // end namespace internal


    /**
     * Sort a range in parallel.
     *
     * @param sort_func Sequential sort method, like std::sort or std::stable_sort
     * @param merge_func Sequential merge method, like std::inplace_merge
     */
    template <class ExecPolicy,
              class RandIt,
              class Compare,
              class SortFunc,
              class MergeFunc>
    #if __cplusplus > 201709L
    	requires std::random_access_iterator<RandIt>
    #endif
    void parallel_mergesort(ExecPolicy &&policy,
                            RandIt first,
                            RandIt last,
                            Compare comp,
                            SortFunc sort_func,
                            MergeFunc merge_func) noexcept {
        if (first == last) {
            return;
        }

        // Sort chunks in parallel
        auto futures = parallel_chunk_for_gen(std::forward<ExecPolicy>(policy), first, last,
                         [&comp, sort_func] (RandIt chunk_first, RandIt chunk_last) {
                             sort_func(chunk_first, chunk_last, comp);
                             return std::make_pair(chunk_first, chunk_last);
                         });

        // Merge the sorted ranges
        using SortedRange = std::pair<RandIt, RandIt>;
        auto& task_pool = *policy.pool();
        std::vector<SortedRange> subranges;
        do {
            for (auto& future : futures) {
                subranges.emplace_back(future.get());
            }
            futures.clear();

            for (std::size_t i = 0; i < subranges.size(); ++i) {
                if (i + 1 < subranges.size()) {
                    // pair up and merge
                    auto& lhs = subranges[i];
                    auto& rhs = subranges[i + 1];
                    futures.emplace_back(task_pool.submit([&comp, merge_func] (RandIt chunk_first,
                                                                               RandIt chunk_middle,
                                                                               RandIt chunk_last) {
                        merge_func(chunk_first, chunk_middle, chunk_last, comp);
                        return std::make_pair(chunk_first, chunk_last);
                    }, lhs.first, lhs.second, rhs.second));
                    ++i;
                } else {
                    // forward the final extra range
                    std::promise<SortedRange> p;
                    futures.emplace_back(p.get_future());
                    p.set_value(subranges[i]);
                }
            }

            subranges.clear();
        } while (futures.size() > 1);
        futures.front().get();
    }

    /**
     * Quicksort worker function.
     */
    template <class RandIt,
              class Compare,
              class SortFunc,
              class PartFunc,
              class PivotFunc>
    #if __cplusplus > 201709L
    	requires std::random_access_iterator<RandIt>
    #endif
    void quicksort_impl(pool_type *task_pool,
                        const RandIt first,
                        const RandIt last,
                        Compare comp,
                        SortFunc sort_func,
                        PartFunc part_func,
                        PivotFunc pivot_func,
                        std::ptrdiff_t target_leaf_size,
                        std::vector<std::future<void>>* futures,
                        std::mutex* mutex,
                        std::condition_variable* cv,
                        int* inflight_spawns) noexcept {
        using T = typename std::iterator_traits<RandIt>::value_type;

        auto partition_size = std::distance(first, last);

        if (partition_size > target_leaf_size) {
            // partition the range
            auto mid = part_func(first, last, pivot_predicate<Compare, T>(comp, pivot_func(first, last)));

            if (mid != first && mid != last) {
                // was able to partition the range, so recurse
                std::lock_guard<std::mutex> guard(*mutex);
                ++(*inflight_spawns);

                futures->emplace_back(task_pool->submit(
                    quicksort_impl<RandIt, Compare, SortFunc, PartFunc, PivotFunc>,
                    task_pool, first, mid, comp, sort_func, part_func, pivot_func, target_leaf_size,
                    futures, mutex, cv, inflight_spawns));

                futures->emplace_back(task_pool->submit(
                    quicksort_impl<RandIt, Compare, SortFunc, PartFunc, PivotFunc>,
                    task_pool, mid, last, comp, sort_func, part_func, pivot_func, target_leaf_size,
                    futures, mutex, cv, inflight_spawns));
                return;
            }
        }

        // Range does not need to be subdivided (or was unable to subdivide). Run the sequential sort.
        {
            // notify main thread that partitioning may be finished
            std::lock_guard<std::mutex> guard(*mutex);
            --(*inflight_spawns);
        }
        cv->notify_one();

        sort_func(first, last, comp);
    }

    /**
     * Sort a range in parallel using quicksort.
     *
     * @param sort_func Sequential sort method, like std::sort or std::stable_sort
     * @param part_func Method that partitions a range, like std::partition or std::stable_partition
     * @param pivot_func Method that identifies the pivot
     */
    template <class ExecPolicy,
              class RandIt,
              class Compare,
              class SortFunc,
              class PartFunc,
              class PivotFunc>
    #if __cplusplus > 201709L
    	requires std::random_access_iterator<RandIt>
    #endif
    void parallel_quicksort(ExecPolicy &&policy,
                            RandIt first,
                            RandIt last,
                            Compare comp,
                            SortFunc sort_func,
                            PartFunc part_func,
                            PivotFunc pivot_func) noexcept {
        if (first == last) {
            return;
        }

        auto& task_pool = *policy.pool();

        // Target partition size. Range will be recursively partitioned into partitions no bigger than this
        // size. Target approximately twice as many partitions as threads to reduce impact of uneven pivot
        // selection.
        auto num_threads = task_pool.get_num_threads();
        std::ptrdiff_t target_leaf_size = std::max((std::ptrdiff_t)(std::distance(first, last) / (num_threads * 2)),
                                                   (std::ptrdiff_t)5);

        if (num_threads == 1) {
            target_leaf_size = std::distance(first, last);
        }

        // task_thread_pool does not support creating task DAGs, so organize the code such that
        // all parallel tasks are independent. The parallel tasks can spawn additional parallel tasks, and they
        // record their "child" task's std::future into a common vector to be waited on by the main thread.
        std::mutex mutex;

        // Futures of parallel tasks. Access protected by mutex.
        std::vector<std::future<void>> futures;

        // For signaling that all partitioning has been completed and futures vector is complete. Uses mutex.
        std::condition_variable cv;

        // Number of `quicksort_impl` calls that haven't finished yet. Nonzero value means futures vector may
        // still be modified. Access protected by mutex.
        int inflight_spawns = 1;

        // Root task.
        quicksort_impl(&task_pool, first, last, comp, sort_func, part_func, pivot_func, target_leaf_size,
                       &futures, &mutex, &cv, &inflight_spawns);

        // Wait for all partitioning to finish.
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return inflight_spawns == 0; });
        }

        // Wait on all the parallel tasks.
        internal::get_futures(futures);
    }



    /**
     * NOTE: Iterators are expected to be random access.
     *
     * Like `std::sort`, but allows specifying the sequential sort method, which must have the
     * same signature as the comparator version of `std::sort`.
     *
     * Implemented as a high-level quicksort that delegates to `sort_func`, in parallel, once the range has been
     * sufficiently partitioned.
     */
    template <class ExecPolicy, 
              class RandIt,
              class Compare>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
    void pluggable_sort(ExecPolicy &&policy,
                        RandIt first,
                        RandIt last,
                        Compare comp,
                        void (sort_func)(RandIt, RandIt, Compare) = std::sort) noexcept {
        if (is_seq<ExecPolicy>(policy)) {
            sort_func(first, last, comp);
            return;
        }

        // Parallel partition.
        // The partition_p2 method spawns and waits for its own child task. A deadlock is possible if all worker
        // threads are waiting for tasks that in turn have to workers to execute them. This is only an issue because
        // our thread pool does not have the concept of dependencies.
        // So ensure
        auto& task_pool = *policy.pool();
        std::atomic<int> allowed_parallel_partitions{(int)task_pool.get_num_threads() / 2};

        auto part_func = [&task_pool, &allowed_parallel_partitions]
                         (RandIt chunk_first,
                          RandIt chunk_last,
                          internal::pivot_predicate<Compare,
                                   typename std::iterator_traits<RandIt>::value_type> pred) {
            if (allowed_parallel_partitions.fetch_sub(1) > 0) {
                return internal::partition_p2(task_pool, chunk_first, chunk_last, pred);
            } else {
                return std::partition(chunk_first, chunk_last, pred);
            }
        };

        parallel_quicksort(std::forward<ExecPolicy>(policy),
                           first,
                           last,
                           comp,
                           sort_func,
                           part_func,
                           internal::quicksort_pivot<RandIt>);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     *
     * Like `std::sort`, but allows specifying the sequential sort method, which must have the
     * same signature as the comparator version of `std::sort`.
     *
     * Implemented as a parallel high-level quicksort that delegates to `sort_func` once the range has been
     * sufficiently partitioned.
     */
    template <class ExecPolicy,
              class RandIt>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
    void pluggable_sort(ExecPolicy &&policy,
                        RandIt first,
                        RandIt last,
                        void (sort_func)(RandIt, RandIt,
                                    std::less<typename std::iterator_traits<RandIt>::value_type>) = std::sort) noexcept {
        using T = typename std::iterator_traits<RandIt>::value_type;
        pluggable_sort(std::forward<ExecPolicy>(policy), first, last, std::less<T>(), sort_func);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     *
     * Parallel merge sort.
     *
     * @param comp Comparator.
     * @param sort_func Sequential sort method. Must have the same signature as the comparator version of `std::sort`.
     * @param merge_func Sequential merge method. Must have the same signature as `std::inplace_merge`.
     */
    template <class ExecPolicy, 
              class RandIt,
              class Compare>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
    void pluggable_mergesort(ExecPolicy &&policy,
                             RandIt first,
                             RandIt last,
                             Compare comp,
                             void (sort_func)(RandIt, RandIt, Compare) = std::sort,
                             void (merge_func)(RandIt, RandIt, RandIt, Compare) = std::inplace_merge) noexcept {
        if (is_seq<ExecPolicy>(policy)) {
            sort_func(first, last, comp);
            return;
        }

        parallel_mergesort(std::forward<ExecPolicy>(policy),
                           first,
                           last,
                           comp,
                           sort_func,
                           merge_func);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     *
     * Parallel merge sort.
     *
     * Uses `std::less` comparator.
     *
     * @param sort_func Sequential sort method. Must have the same signature as the comparator version of `std::sort`.
     * @param merge_func Sequential merge method. Must have the same signature as `std::inplace_merge`.
     */
    template <class ExecPolicy,
              class RandIt>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
    void pluggable_mergesort(ExecPolicy &&policy,
                             RandIt first,
                             RandIt last,
                             void (sort_func)(RandIt, RandIt,
                                    std::less<typename std::iterator_traits<RandIt>::value_type>) = std::sort,
                             void (merge_func)(RandIt, RandIt, RandIt,
                                    std::less<typename std::iterator_traits<RandIt>::value_type>) = std::inplace_merge) noexcept {
        using T = typename std::iterator_traits<RandIt>::value_type;
        pluggable_mergesort(std::forward<ExecPolicy>(policy),
                            first,
                            last,
                            std::less<T>(),
                            sort_func, merge_func);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     *
     * Parallel quicksort that allows specifying the sequential sort and partition methods.
     *
     * @param comp Comparator.
     * @param sort_func Sequential sort method to use once range is sufficiently partitioned. Must have the same
     *                  signature as the comparator version of `std::sort`.
     * @param part_func Sequential partition method. Must have the same signature as `std::partition`.
     * @param pivot_func Method that identifies the pivot element
     */
    template <class ExecPolicy,
              class RandIt,
              class Compare>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
    void pluggable_quicksort(ExecPolicy &&policy,
                             RandIt first,
                             RandIt last,
                             Compare comp,
                             void (sort_func)(RandIt, RandIt, Compare) = std::sort,
                             RandIt (part_func)(RandIt, RandIt, internal::pivot_predicate<Compare,
                            typename std::iterator_traits<RandIt>::value_type>) = std::partition,
                            typename std::iterator_traits<RandIt>::value_type (pivot_func)(RandIt, RandIt) =
                            internal::quicksort_pivot) noexcept{
        if (is_seq<ExecPolicy>(policy)) {
            sort_func(first, last, comp);
            return;
        }

        parallel_quicksort(std::forward<ExecPolicy>(policy),
                           first,
                           last,
                           comp,
                           sort_func,
                           part_func,
                           pivot_func);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     *
     * Parallel quicksort that allows specifying the sequential sort and partition methods.
     *
     * Uses `std::less` comparator.
     *
     * @param sort_func Sequential sort method to use once range is sufficiently partitioned. Must have the same
     *                  signature as the comparator version of `std::sort`.
     * @param part_func Sequential partition method. Must have the same signature as `std::partition`.
     * @param pivot_func Method that identifies the pivot element
     */
    template <class ExecPolicy, 
              class RandIt>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
    void pluggable_quicksort(ExecPolicy &&policy,
                             RandIt first,
                             RandIt last,
                             void (sort_func)(RandIt, RandIt,
                                    std::less<typename std::iterator_traits<RandIt>::value_type>) = std::sort,
                             RandIt (part_func)(RandIt, RandIt, internal::pivot_predicate<
                                std::less<typename std::iterator_traits<RandIt>::value_type>,
                            typename std::iterator_traits<RandIt>::value_type>) = std::partition,
                            typename std::iterator_traits<RandIt>::value_type (pivot_func)(RandIt, RandIt) =
                            internal::quicksort_pivot) noexcept {
        using T = typename std::iterator_traits<RandIt>::value_type;
        pluggable_quicksort(std::forward<ExecPolicy>(policy),
                            first,
                            last,
                            std::less<T>(),
                            sort_func,
                            part_func,
                            pivot_func);
    }

} // end namespace cryptanalysislib
