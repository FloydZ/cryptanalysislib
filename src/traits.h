#ifndef CRYPTANALYSISLIB_TRAITS_H
#define CRYPTANALYSISLIB_TRAITS_H

namespace cryptanalysislib {
	namespace internal {
		// NOTE: this value can be changed, if you need it.
		constexpr static size_t __max_nr_lambda_args = 50;

		struct any_argument {
			template <typename T>
			operator T&&() const;
		};

		template <typename Lambda,
				  typename Is,
				  typename = void>
		struct can_accept_impl
		    : std::false_type
		{};

		template <typename Lambda,
				  std::size_t ...Is>
		struct can_accept_impl<Lambda, std::index_sequence<Is...>,
		                       decltype(std::declval<Lambda>()(((void)Is, any_argument{})...), void())>
		    : std::true_type
		{};

		template <typename Lambda,
				  std::size_t N>
		struct can_accept
		    : can_accept_impl<Lambda, std::make_index_sequence<N>>
		{};

		template <typename Lambda,
				  std::size_t Max,
				  std::size_t N,
				  typename = void>
		struct lambda_details_impl
		    : lambda_details_impl<Lambda, Max, N - 1>
		{};

		template <typename Lambda,
				  std::size_t Max,
				  std::size_t N>
		struct lambda_details_impl<Lambda, Max, N, std::enable_if_t<can_accept<Lambda, N>::value>> {
			static constexpr bool is_variadic = (N == Max);
			static constexpr std::size_t argument_count = N;
		};

		/// NOTE: currently the maximal numbers of arguments the is limited,
		/// in this case `__max_nr_lambda_args`. But you can set it to what
		/// ever you want.
		/// usage:
		///	  auto fn = [](){
		///		// some lambda function
		///	  }
		///   if constexpr (cryptanalysislib::internal::lambda_details<decltype(fn)>::argument_count > 0) {
		///       // some stuff
		///   }
		/// \tparam Lambda
		/// \tparam Max
		template <typename Lambda, std::size_t Max = __max_nr_lambda_args>
		struct lambda_details
		    : lambda_details_impl<Lambda, Max, Max>
		{};
	}


	///
	template<typename TCallable,
			 typename TSignature>
	constexpr bool is_callable_as_v = std::is_constructible<std::function<TSignature>, TCallable>::value;

	template<
			typename TCallable,
			typename TSignature,
			typename = std::enable_if_t<is_callable_as_v<TCallable, TSignature>>>
	using Callable = TCallable;

	template<typename TCallable>
	struct CallableMetadata
			: CallableMetadata<decltype(&TCallable::operator())>
	{};

	///	 auto kek = [](){
	///     // some lambda
	///	 }
	///  auto *ptr = CallableMetadata<decltype(kek)>::generatePointer(kek);
	/// \tparam TClass
	/// \tparam TReturn
	/// \tparam TArgs
	template<class TClass,
			typename TReturn,
			typename... TArgs>
	struct CallableMetadata<TReturn(TClass::*)(TArgs...)> {
		using class_t = TClass;
		using return_t = TReturn;
		using args_tuple_t = std::tuple<TArgs...>;
		using ptr_t = TReturn(*)(TArgs...);

		// Beware! this function makes a copy of the closure! and of the arguments when called!
		static ptr_t generatePointer(const TClass& closure) {
			static TClass staticClosureCopy = closure;

			return [](TArgs... args){
			  return staticClosureCopy(args...);
			};
		}
	};

	/// \tparam TClass
	/// \tparam TReturn
	/// \tparam TArgs
	template<class TClass,
			 typename TReturn,
			 typename... TArgs>
	struct CallableMetadata<TReturn(TClass::*)(TArgs...) const> {
		using class_t 		= TClass;
		using return_t 		= TReturn;
		using args_tuple_t 	= std::tuple<TArgs...>;
		using ptr_t 		= TReturn(*)(TArgs...) ;

		// Beware! this function makes a copy of the closure!
		// and of the arguments when called!
		static ptr_t generatePointer(const TClass& closure) {
			static TClass staticClosureCopy = closure;

			return [](TArgs... args) {
			  return staticClosureCopy(args...);
			};
		}
	};

}


template<typename T>
struct IsStdArray : std::false_type {};

template<typename T, std::size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type {};
#endif
