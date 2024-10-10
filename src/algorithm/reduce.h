//
// Created by duda on 10.10.24.
//

#ifndef REDUCE_H
#define REDUCE_H
/**
 * NOTE: Iterators are expected to be random access.
 * See std::reduce https://en.cppreference.com/w/cpp/algorithm/reduce
 */
template <class ExecPolicy, class RandIt, class T, class BinaryOp>
poolstl::internal::enable_if_poolstl_policy<ExecPolicy, T>
reduce(ExecPolicy &&policy, RandIt first, RandIt last, T init, BinaryOp binop) {
	if (poolstl::internal::is_seq<ExecPolicy>(policy)) {
		return poolstl::internal::cpp17::reduce(first, last, init, binop);
	}

	auto futures = poolstl::internal::parallel_chunk_for_1(std::forward<ExecPolicy>(policy), first, last,
														   poolstl::internal::cpp17::reduce<RandIt, T, BinaryOp>,
														   (T*)nullptr, 1, init, binop);

	return poolstl::internal::cpp17::reduce(
		poolstl::internal::get_wrap(futures.begin()),
		poolstl::internal::get_wrap(futures.end()), init, binop);
}

/**
 * NOTE: Iterators are expected to be random access.
 * See std::reduce https://en.cppreference.com/w/cpp/algorithm/reduce
 */
template <class ExecPolicy, class RandIt, class T>
poolstl::internal::enable_if_poolstl_policy<ExecPolicy, T>
reduce(ExecPolicy &&policy, RandIt first, RandIt last, T init) {
	return std::reduce(std::forward<ExecPolicy>(policy), first, last, init, std::plus<T>());
}

/**
 * NOTE: Iterators are expected to be random access.
 * See std::reduce https://en.cppreference.com/w/cpp/algorithm/reduce
 */
template <class ExecPolicy, class RandIt>
poolstl::internal::enable_if_poolstl_policy<
	ExecPolicy, typename std::iterator_traits<RandIt>::value_type>
reduce(ExecPolicy &&policy, RandIt first, RandIt last) {
	return std::reduce(std::forward<ExecPolicy>(policy), first, last,
					   typename std::iterator_traits<RandIt>::value_type{});
}
#endif //REDUCE_H
