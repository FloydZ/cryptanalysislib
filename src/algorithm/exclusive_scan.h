#ifndef CRYPTANALYSISLIB_ALGORITHM_EXCLUSIVE_SCAN_H
#define CRYPTANALYSISLIB_ALGORITHM_EXCLUSIVE_SCAN_H
    /**
     * NOTE: Iterators are expected to be random access.
     * See std::exclusive_scan https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
     */
    template <class ExecPolicy, class RandIt1, class RandIt2, class T, class BinaryOp>
    poolstl::internal::enable_if_poolstl_policy<ExecPolicy, RandIt2>
    exclusive_scan(ExecPolicy &&policy, RandIt1 first, RandIt1 last, RandIt2 dest, T init, BinaryOp binop) {
        if (first == last) {
            return dest;
        }

        if (poolstl::internal::is_seq<ExecPolicy>(policy)) {
            return std::exclusive_scan(first, last, dest, init, binop);
        }

        // Pass 1: Chunk the input and find the sum of each chunk
        auto futures = poolstl::internal::parallel_chunk_for_gen(std::forward<ExecPolicy>(policy), first, last,
                             [binop](RandIt1 chunk_first, RandIt1 chunk_last) {
                                 auto sum = std::accumulate(chunk_first, chunk_last, T{}, binop);
                                 return std::make_tuple(std::make_pair(chunk_first, chunk_last), sum);
                             });

        std::vector<std::pair<RandIt1, RandIt1>> ranges;
        std::vector<T> sums;

        for (auto& future : futures) {
            auto res = future.get();
            ranges.push_back(std::get<0>(res));
            sums.push_back(std::get<1>(res));
        }

        // find initial values for each range
        std::exclusive_scan(sums.begin(), sums.end(), sums.begin(), init, binop);

        // Pass 2: perform exclusive scan of each chunk, using the sum of previous chunks as init
        std::vector<std::tuple<RandIt1, RandIt1, RandIt2, T>> args;
        for (std::size_t i = 0; i < sums.size(); ++i) {
            auto chunk_first = std::get<0>(ranges[i]);
            args.emplace_back(std::make_tuple(
                chunk_first, std::get<1>(ranges[i]),
                dest + (chunk_first - first),
                sums[i]));
        }

        auto futures2 = poolstl::internal::parallel_apply(std::forward<ExecPolicy>(policy),
            [binop](RandIt1 chunk_first, RandIt1 chunk_last, RandIt2 chunk_dest, T chunk_init){
                std::exclusive_scan(chunk_first, chunk_last, chunk_dest, chunk_init, binop);
            }, args);

        poolstl::internal::get_futures(futures2);
        return dest + (last - first);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     * See std::exclusive_scan https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
     */
    template <class ExecPolicy, class RandIt1, class RandIt2, class T>
    poolstl::internal::enable_if_poolstl_policy<ExecPolicy, RandIt2>
    exclusive_scan(ExecPolicy &&policy, RandIt1 first, RandIt1 last, RandIt2 dest, T init) {
        return std::exclusive_scan(std::forward<ExecPolicy>(policy), first, last, dest, init, std::plus<T>());
    }
#endif
