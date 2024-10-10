#ifndef TRANSFORM_H
#define TRANSFORM_H
    /**
     * NOTE: Iterators are expected to be random access.
     * See std::transform_reduce https://en.cppreference.com/w/cpp/algorithm/transform_reduce
     */
    template <class ExecPolicy, class RandIt1, class T, class BinaryReductionOp, class UnaryTransformOp>
    poolstl::internal::enable_if_poolstl_policy<ExecPolicy, T>
    transform_reduce(ExecPolicy&& policy, RandIt1 first1, RandIt1 last1, T init,
                     BinaryReductionOp reduce_op, UnaryTransformOp transform_op) {
        if (poolstl::internal::is_seq<ExecPolicy>(policy)) {
            return std::transform_reduce(first1, last1, init, reduce_op, transform_op);
        }

        auto futures = poolstl::internal::parallel_chunk_for_1(std::forward<ExecPolicy>(policy), first1, last1,
                                                               std::transform_reduce<RandIt1, T,
                                                                                   BinaryReductionOp, UnaryTransformOp>,
                                                               (T*)nullptr, 1, init, reduce_op, transform_op);

        return poolstl::internal::cpp17::reduce(
            poolstl::internal::get_wrap(futures.begin()),
            poolstl::internal::get_wrap(futures.end()), init, reduce_op);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     * See std::transform_reduce https://en.cppreference.com/w/cpp/algorithm/transform_reduce
     */
    template <class ExecPolicy, class RandIt1, class RandIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
    poolstl::internal::enable_if_poolstl_policy<ExecPolicy, T>
    transform_reduce(ExecPolicy&& policy, RandIt1 first1, RandIt1 last1, RandIt2 first2, T init,
                     BinaryReductionOp reduce_op, BinaryTransformOp transform_op) {
        if (poolstl::internal::is_seq<ExecPolicy>(policy)) {
            return std::transform_reduce(first1, last1, first2, init, reduce_op, transform_op);
        }

        auto futures = poolstl::internal::parallel_chunk_for_2(std::forward<ExecPolicy>(policy), first1, last1, first2,
                                                               std::transform_reduce<RandIt1, RandIt2, T,
                                                                                  BinaryReductionOp, BinaryTransformOp>,
                                                               (T*)nullptr, init, reduce_op, transform_op);

        return poolstl::internal::cpp17::reduce(
            poolstl::internal::get_wrap(futures.begin()),
            poolstl::internal::get_wrap(futures.end()), init, reduce_op);
    }

    /**
     * NOTE: Iterators are expected to be random access.
     * See std::transform_reduce https://en.cppreference.com/w/cpp/algorithm/transform_reduce
     */
    template< class ExecPolicy, class RandIt1, class RandIt2, class T >
    poolstl::internal::enable_if_poolstl_policy<ExecPolicy, T>
    transform_reduce(ExecPolicy&& policy, RandIt1 first1, RandIt1 last1, RandIt2 first2, T init ) {
        return transform_reduce(std::forward<ExecPolicy>(policy),
            first1, last1, first2, init, std::plus<>(), std::multiplies<>());
    }
#endif //TRANSFORM_H
