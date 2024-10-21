#ifndef CRYPTANALYSISLIB_ALGORITHM_EXCLUSIVE_SCAN_H
#define CRYPTANALYSISLIB_ALGORITHM_EXCLUSIVE_SCAN_H

#include "algorithm/algorithm.h"
#include "algorithm/accumulate.h"
#include "algorithm/prefixsum.h"
#include "algorithm/apply.h"

namespace cryptanalysislib {
    struct AlgorithmExclusiveScanConfig : public AlgorithmConfig {
    	constexpr static size_t min_size_per_thread = 131072;
    };
    constexpr static AlgorithmExclusiveScanConfig algorithmExclusiveScanConfig;

	/// \tparam InputIt
    /// \tparam OutputIt
    /// \tparam BinaryOp
    /// \tparam config
    /// \param first
    /// \param last
    /// \param d_first
    /// \param init
    /// \param op
    /// \return
    template<class InputIt,
             class OutputIt,
             class BinaryOp,
             const AlgorithmExclusiveScanConfig &config=algorithmExclusiveScanConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
		         std::forward_iterator<OutputIt> &&
    		     std::regular_invocable<BinaryOp,
									const typename InputIt::value_type&,
									const typename InputIt::value_type&>
#endif
    OutputIt exclusive_scan(InputIt first,
                            InputIt last,
                            OutputIt d_first,
                            const typename InputIt::value_type init,
                            BinaryOp op) {
    	using T = InputIt::value_type;
		if (first == last) {
			return d_first;
		}

    	T acc = init;
    	*(d_first++) = acc;
    	last -= 1;

    	while (first != last) {
    		acc = op(std::move(acc), *first);
    		*(d_first++) = acc;
    		first += 1;
    	}

    	return d_first;
    }

	/// \tparam InputIt
    /// \tparam OutputIt
    /// \tparam config
    /// \param first
    /// \param last
    /// \param d_first
    /// \param init
    /// \return
    template<class InputIt,
             class OutputIt,
             const AlgorithmExclusiveScanConfig &config=algorithmExclusiveScanConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt> &&
		         std::forward_iterator<OutputIt>
#endif
    OutputIt exclusive_scan(InputIt first,
                            InputIt last,
                            OutputIt d_first,
                            const typename InputIt::value_type init) noexcept {
    	using T = InputIt::value_type;
		return cryptanalysislib::exclusive_scan
            <InputIt, OutputIt, decltype(std::plus<T>()), config>
            (first, last, d_first, init, std::plus<T>());
    }


    template<class InputIt,
             class OutputIt,
             const AlgorithmExclusiveScanConfig &config=algorithmExclusiveScanConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
		         std::forward_iterator<OutputIt>
#endif
    OutputIt exclusive_scan(InputIt first,
                            InputIt last,
                            OutputIt d_first) noexcept {
    	using T = InputIt::value_type;
		return cryptanalysislib::exclusive_scan(first, last, d_first, T{}, std::plus<T>());
    }


	/// \tparam ExecPolicy
    /// \tparam RandIt1
    /// \tparam RandIt2
    /// \tparam BinaryOp
    /// \param policy
    /// \param first
    /// \param last
    /// \param dest
    /// \param init
    /// \param binop
    /// \return
    template <class ExecPolicy,
              class RandIt1,
              class RandIt2,
              class BinaryOp,
              const AlgorithmExclusiveScanConfig &config=algorithmExclusiveScanConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
		         std::random_access_iterator<RandIt2> &&
    		     std::regular_invocable<BinaryOp,
									const typename RandIt1::value_type&,
									const typename RandIt1::value_type&>
#endif
    RandIt2 exclusive_scan(ExecPolicy &&policy,
                           RandIt1 first,
                           RandIt1 last,
                           RandIt2 dest,
                           const typename RandIt1::value_type init,
                           BinaryOp binop) noexcept {
        if (first == last) {
            return dest;
        }

    	using T = RandIt1::value_type;
		const size_t size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::exclusive_scan
				<RandIt1, RandIt2, decltype(binop), config>
				(first, last, dest, init, binop);
		}

        // Pass 1: Chunk the input and find the sum of each chunk
        auto futures = internal::parallel_chunk_for_gen(std::forward<ExecPolicy>(policy), first, last,
        [binop](RandIt1 chunk_first, RandIt1 chunk_last) __attribute__((always_inline)) {
            auto sum = cryptanalysislib::accumulate(chunk_first, chunk_last, T{}, binop);
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

        auto futures2 = parallel_apply(std::forward<ExecPolicy>(policy),
            [binop](RandIt1 chunk_first, RandIt1 chunk_last, RandIt2 chunk_dest, T chunk_init){
                cryptanalysislib::exclusive_scan(chunk_first, chunk_last, chunk_dest, chunk_init, binop);
            }, args);

        internal::get_futures(futures2);
        return dest + (last - first);
    }

	/// \tparam ExecPolicy
    /// \tparam RandIt1
    /// \tparam RandIt2
    /// \tparam config
    /// \param policy
    /// \param first
    /// \param last
    /// \param dest
    /// \param init
    /// \return
    template<class ExecPolicy,
             class RandIt1,
             class RandIt2,
             const AlgorithmExclusiveScanConfig &config=algorithmExclusiveScanConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
		         std::random_access_iterator<RandIt2>
#endif
    RandIt2 exclusive_scan(ExecPolicy &&policy,
                           RandIt1 first,
                           RandIt1 last,
                           RandIt2 dest,
                           const typename RandIt1::value_type init) noexcept {
        using T = RandIt1::value_type;
        return cryptanalysislib::exclusive_scan
            <ExecPolicy, RandIt1, RandIt2, std::plus<T>, config>
            (std::forward<ExecPolicy>(policy), first, last, dest, init, std::plus<T>());
    }

    template<class ExecPolicy,
             class RandIt1,
             class RandIt2,
             const AlgorithmExclusiveScanConfig &config=algorithmExclusiveScanConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
		         std::random_access_iterator<RandIt2>
#endif
    RandIt2 exclusive_scan(ExecPolicy &&policy,
                           RandIt1 first,
                           RandIt1 last,
                           RandIt2 dest) noexcept {
        using T = RandIt1::value_type;
        return cryptanalysislib::exclusive_scan
            <ExecPolicy, RandIt1, RandIt2, std::plus<T>, config>
            (std::forward<ExecPolicy>(policy), first, last, dest, T{}, std::plus<T>());
    }
 } // end namespace cryptanalysislib
#endif
