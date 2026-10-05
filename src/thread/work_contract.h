#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "container/signal_tree/tree.h"
#include "traits.h"

//=========================================================================
/// TODO move somewhere usefull
static inline constexpr auto minimum_power_of_two(std::unsigned_integral auto value){
	return (1ull << minimum_bit_count(value - 1));
}

///
enum class synchronization_mode : std::uint32_t {
	synchronous = 0,
	blocking = synchronous,
	sync = synchronous,
	asynchronous = 1,
	non_blocking = asynchronous,
	async = asynchronous
};
using work_contract_id = std::uint64_t;

///////////////////////////////////////////////////////////////////////////////////

namespace cryptanalysislib::work_contract {
	/// \tparam T
	template<synchronization_mode T>
	class work_contract_group;

	//=========================================================================
	template<synchronization_mode T>
	class work_contract_token final : non_movable, non_copyable {
	public:
		///
		void schedule() noexcept {
			if (!released_) {
				owner_.schedule(contractId_);
			}
		}

		///
		void release() noexcept {
			if (!released_) {
				owner_.release(contractId_);
			}
			released_ = true;
		}

		/// \return
		work_contract_id get_contract_id() const noexcept {
			return contractId_;
		}


	private:
		friend work_contract_group<T>;
		work_contract_token() = delete;

		/// \param contractId
		/// \param owner
		work_contract_token(work_contract_id contractId,
				    work_contract_group<T> &owner) noexcept : contractId_(contractId),
									      owner_(owner) {}

		~work_contract_token() = default;

		work_contract_id contractId_{~0ull};
		work_contract_group<T> &owner_;
		bool released_{false};
	};// class work_contract_group<>::work_contract_token
}

using asynchronous_work_contract_token = cryptanalysislib::work_contract::work_contract_token<synchronization_mode::asynchronous>;
using synchronous_work_contract_token = cryptanalysislib::work_contract::work_contract_token<synchronization_mode::synchronous>;


template<typename T>
concept work_contract_token_callable = ((std::is_invocable_v<std::decay_t<T>, asynchronous_work_contract_token &>) ||
                                        (std::is_invocable_v<std::decay_t<T>, synchronous_work_contract_token &>) );

template<typename T>
concept work_contract_no_token_callable = (std::is_invocable_v<std::decay_t<T>>);


template<typename T>
concept work_contract_callable = (work_contract_token_callable<T> || work_contract_no_token_callable<T>);


/////////////////////////////////////////////////////////////////////////////////
namespace cryptanalysislib::work_contract {

	/// \tparam T
	template<synchronization_mode T>
	class alignas(64) work_contract : non_copyable {
	public:
		using id_type = std::uint64_t;

		enum class initial_state {
			unscheduled = 0,
			scheduled = 1
		};

		work_contract() = default;

		~work_contract() noexcept {
			release();
		}

		/// \param other
		work_contract(work_contract &&other) noexcept 
        : owner_(other.owner_), releaseToken_(other.releaseToken_), id_(other.id_) {
			other.owner_ = {};
			other.id_ = {};
			other.releaseToken_ = {};
		}

		/// \param other
		/// \return
		work_contract &operator=(work_contract &&other) noexcept {
			if (this != &other) {
				release();

				owner_ = other.owner_;
				id_ = other.id_;
				releaseToken_ = other.releaseToken_;

				other.owner_ = {};
				other.id_ = {};
				other.releaseToken_ = {};
			}

			return *this;
		}

		///
		void schedule() noexcept {
			owner_->schedule(id_);
		}

		/// \return
		bool release() noexcept {
			if (auto releaseToken = std::exchange(releaseToken_, nullptr); releaseToken) {
				releaseToken->schedule(*this);
				owner_ = {};
				return true;
			}

			return false;
		}

        // TODO
		bool deschedule();

		/// \return
		[[nodiscard]] constexpr inline bool is_valid() const noexcept {
			return ((releaseToken_) && (releaseToken_->is_valid()));
		}

		/// \return
		constexpr explicit operator bool() const noexcept {
			return is_valid();
		}

	private:
		friend class work_contract_group<T>;
		using work_contract_group_type = work_contract_group<T>;

		work_contract(work_contract_group_type *owner,
		              std::shared_ptr<typename work_contract_group_type::release_token> releaseToken,
		              id_type id,
		              initial_state initialState) : owner_(owner),
		                                            releaseToken_(releaseToken),
		                                            id_(id) {
			if (initialState == initial_state::scheduled)
				schedule();
		}

		/// \return
		[[nodiscard]] constexpr inline id_type get_id() const noexcept {
			return id_;
		}

		work_contract_group_type *owner_{};

		std::shared_ptr<typename work_contract_group_type::release_token> releaseToken_;

		id_type id_{};

	};// class work_contract
}// namespace cryptanalysislib::work_contract

using work_contract = cryptanalysislib::work_contract::work_contract<synchronization_mode::non_blocking>;
using blocking_work_contract = cryptanalysislib::work_contract::work_contract<synchronization_mode::blocking>;

//////////////////////////////////////////////////////////////////////////////////////////

namespace cryptanalysislib::work_contract {
	template<synchronization_mode T>
	class work_contract_group final : non_copyable,
					  non_movable {
	public:
		//=============================================================================
		template<std::uint64_t total_counters, 
                 std::uint64_t bits_per_counter,
                 std::uint64_t bias_bit = 1ull << 63u>
		struct largest_child_selector {
			inline auto operator()(
				std::uint64_t,
				std::uint64_t counters) const noexcept -> signal_index {
				if constexpr (bits_per_counter == 1) {
					return (counters > 0) ? std::countl_zero(counters) : ~0ull;
				} else {
					// this routine is only called to select new contract ids.  but we could improve speed here
					// with a expression fold as total counters is never more than 8 and often 4 or 2.
					auto selected = ~0ull;
					auto max = 0ull;
					/*static*/ auto /*constexpr*/ counter_mask = ((1ull << bits_per_counter) - 1);
					for (auto i = 0ull; i < total_counters; ++i) {
						if ((counters & counter_mask) > max) {
							max = (counters & counter_mask);
							selected = i;
						}
						counters >>= bits_per_counter;
					}
					return (total_counters - selected - 1);
				}
			}
		};

		static auto constexpr mode = T;
		using work_contract_type = cryptanalysislib::work_contract::work_contract<mode>;

		static auto constexpr default_capacity = 512;

		class release_token;

		///
		work_contract_group() : work_contract_group(default_capacity) {}

		/// \param capacity
		work_contract_group(std::uint64_t capacity) :
			subTreeCount_(minimum_power_of_two((capacity + (signal_tree_type::capacity - 1)) / signal_tree_type::capacity)),
		    subTreeMask_(subTreeCount_ - 1),
		    subTreeShift_(minimum_bit_count(signal_tree_type::capacity - 1)),
		    signalTree_(subTreeCount_),
		    available_(subTreeCount_),
		    contracts_(subTreeCount_ * signal_tree_type::capacity),
		    release_(contracts_.size()),
		    exception_(contracts_.size()),
		    releaseToken_(subTreeCount_ * signal_tree_type::capacity) {
			for (auto & subtree : available_) {
				for (auto i = 0ull; i < signal_tree_type::capacity; ++i) {
					subtree.set(i);
				}
			}
		}


		~work_contract_group() {
			stop();
		}

		/// \param workFunction
		/// \param initialState
		/// \return
		work_contract_type create_contract(
			work_contract_callable auto &&workFunction,
			work_contract_type::initial_state initialState = work_contract_type::initial_state::unscheduled){
			return create_contract(std::forward<std::decay_t<decltype(workFunction)>>(workFunction), [](){}, initialState);
		}

		/// \param workFunction
		/// \param releaseFunction
		/// \param initialState
		/// \return
		work_contract_type create_contract(
			work_contract_callable auto &&workFunction,
			std::invocable auto &&releaseFunction,
			work_contract_type::initial_state initialState = work_contract_type::initial_state::unscheduled){
			return create_contract(std::forward<std::decay_t<decltype(workFunction)>>(workFunction),
								  std::forward<decltype(releaseFunction)>(releaseFunction), [](auto &, auto){},
								  initialState);
		}

		/// \param workFunction
		/// \param releaseFunction
		/// \param exceptionFunction
		/// \param initialState
		/// \return
		work_contract_type create_contract(
			work_contract_callable auto &&workFunction,
			std::invocable auto &&releaseFunction,
			std::invocable<work_contract_token<T> &,std::exception_ptr> auto &&exceptionFunction,
			work_contract_type::initial_state initialState = work_contract_type::initial_state::unscheduled){
			if (auto workContractId = get_available_contract(); workContractId != ~0ull) {
				auto & contract = contracts_[workContractId];
				contract.flags_ = 0;
				if constexpr (work_contract_token_callable<std::decay_t<decltype(workFunction)>>)
					contract.work_ = std::forward<std::decay_t<decltype(workFunction)>>(workFunction);
				if constexpr (work_contract_no_token_callable<std::decay_t<decltype(workFunction)>>)
					contract.work_ = [work = std::forward<std::decay_t<decltype(workFunction)>>(workFunction)](auto &) mutable{work();};

				release_[workContractId] = std::forward<decltype(releaseFunction)>(releaseFunction);
				exception_[workContractId] = std::forward<decltype(exceptionFunction)>(exceptionFunction);
				return {this, releaseToken_[workContractId] = std::make_shared<release_token>(this), workContractId, initialState};
			}

			return {};
		}

		/// select a signal (a set signal) from the array of signal trees and, if found,
		/// (which clears the signal) then process the pending action on that contract
		/// based on the flags associated with that contract.
		/// \return
		std::uint64_t execute_next_contract() noexcept {
			return execute_next_contract(tls_biasFlags_);
		}

		// select a signal (a set signal) from the array of signal trees and, if found,
		// (which clears the signal) then process the pending action on that contract
		// based on the flags associated with that contract.
		std::uint64_t execute_next_contract(std::uint64_t &biasFlags) {
			if constexpr (mode == synchronization_mode::blocking) {
				if (!waitableState_.wait(this)) {
					// this should be done more graceful but for now ...
					return ~0ull;
				}
			}

			auto subTreeIndex = (biasFlags / signal_tree_type::capacity);
			for (auto i = 0ull; i < signalTree_.size(); ++i) {
				subTreeIndex &= subTreeMask_;
				if (auto [signalIndex, treeIsEmpty] = signalTree_[subTreeIndex].select(biasFlags); signalIndex != invalid_signal_index) {
					if constexpr (mode == synchronization_mode::blocking) {
						if (treeIsEmpty) {
							decrement_non_zero_counter();
						}
					}

					work_contract_id workContractId(subTreeIndex * signal_tree_capacity);
					workContractId |= signalIndex;
					std::uint64_t b = (1ull << std::countr_zero(select_bias_hint ^ biasFlags)) & (signal_tree_type::capacity - 1);
					if (b == 0) {
						biasFlags = ((subTreeIndex + 1) * signal_tree_type::capacity);
					} else {
						biasFlags |= b;
						biasFlags &= ~(b - 1);
					}
					process_contract(workContractId);
					return signalIndex;
				}
				biasFlags = (++subTreeIndex * signal_tree_type::capacity);
			}
			return ~0ull;

		}

		/// select a signal (a set signal) from the array of signal trees and, if found,
		/// (which clears the signal) then process the pending action on that contract
		/// based on the flags associated with that contract.
		/// \tparam rep
		/// \tparam period
		/// \param duration
		/// \return
		template<typename rep, typename period>
		std::uint64_t execute_next_contract(std::chrono::duration<rep, period> duration )
			requires(mode == synchronization_mode::blocking) {
			return execute_next_contract(duration, tls_biasFlags_);
		}


		/// select a signal (a set signal) from the array of signal trees and, if found,
		/// (which clears the signal) then process the pending action on that contract
		/// based on the flags associated with that contract.
		/// \tparam rep
		/// \tparam period
		/// \param duration
		/// \param biasFlags
		/// \return
		template<typename rep, typename period>
		std::uint64_t execute_next_contract(
			std::chrono::duration<rep, period> duration,
			std::uint64_t &biasFlags)
		    requires(mode == synchronization_mode::blocking) {
			if (waitableState_.wait_for(this, duration)) {
				return this->execute_next_contract(biasFlags);
			}
			return ~0ull;
		    }

		///
		void stop() noexcept {
			if (bool wasRunning = !stopped_.exchange(true); wasRunning) {
				for (auto & releaseToken : releaseToken_) {
					if ((bool)releaseToken) {
						releaseToken->orphan();
					}
				}
			}
		}


	private:
		class auto_erase_contract;
		class auto_clear_execute_flag;

		friend class cryptanalysislib::work_contract::work_contract<mode>;
		friend class release_token;
		friend class work_contract_token<T>;
		friend class auto_erase_contract;
		friend class auto_clear_execute_flag;

		using state_flags = std::int8_t;

		struct alignas(64) contract {
			static auto constexpr release_flag = 0x00000004;
			static auto constexpr execute_flag = 0x00000002;
			static auto constexpr schedule_flag = 0x00000001;

			std::atomic<state_flags> flags_;
			std::function<void(work_contract_token<T> &)> work_;
		};

		/// set the schedule flag.  if not previously set, and not currently executing
		/// then also set the signal associated with the contract.
		/// \param contractId
		void schedule(work_contract_id contractId) noexcept {
			static auto constexpr flags_to_set = contract::schedule_flag;
			auto previousFlags = contracts_[contractId].flags_.fetch_or(flags_to_set);
			auto notScheduledNorExecuting = ((previousFlags & (contract::schedule_flag | contract::execute_flag)) == 0);
			if (notScheduledNorExecuting) {
				set_contract_signal(contractId);
			}
		}

		/// \param contractId
		void release(work_contract_id contractId) noexcept {
			static auto constexpr flags_to_set = (contract::release_flag | contract::schedule_flag);
			auto previousFlags = contracts_[contractId].flags_.fetch_or(flags_to_set);
			auto notScheduledNorExecuting = ((previousFlags & (contract::schedule_flag | contract::execute_flag)) == 0);
			if (notScheduledNorExecuting) {
				set_contract_signal(contractId);
			}
		}

		/// set the signal that is associated with the specified contract
		/// @param contractId
		void set_contract_signal(work_contract_id contractId) {
			if constexpr (mode == synchronization_mode::non_blocking) {
				auto [treeIndex, signalIndex] = get_tree_and_signal_index(contractId);
				signalTree_[treeIndex].set(signalIndex);
			} else {
				auto [treeIndex, signalIndex] = get_tree_and_signal_index(contractId);
				if (auto [treeWasEmpty, success] = signalTree_[treeIndex].set(signalIndex); treeWasEmpty) {
					increment_non_zero_counter();
				}
			}
		}

		/// \param contractId
		void process_contract(work_contract_id contractId) {
			auto & contract = contracts_[contractId];
			auto flags = ++contract.flags_;

			if (auto isReleased = ((flags & contract::release_flag) == contract::release_flag); isReleased) {
				// release should be far less common path so ensure not inlined
				process_release(contractId);
				return;
			}

			/// TODO remove
			// the expected case. invoke the work function
			auto_clear_execute_flag autoClearExecuteFlag(contractId, *this);
			try {
				work_contract_token workContractToken(contractId, *this);
				contract.work_(workContractToken);
			} catch (...) {
				process_exception(contractId, std::current_exception());
			}
		}

		/// invoke the contract's release function.  use auto class to ensure
		/// erasure of contract in the event of exceptions in the release function.
		/// \param contractId
		void process_release(work_contract_id contractId){
			auto_erase_contract autoEraseContract(contractId, *this);
			try {
				release_[contractId]();
			} catch (std::exception const & exception) {
				process_exception(contractId, std::current_exception());}
		}

		/// \param contractId
		/// \param exception
		void process_exception(work_contract_id contractId,
							   std::exception_ptr exception) {
			if (exception_[contractId]) {
				work_contract_token workContractToken(contractId, *this);
				exception_[contractId](workContractToken, exception);
			} else {
				std::rethrow_exception(exception);
			}
		}


		/// \param contractId
		void clear_execute_flag(work_contract_id contractId) {
			if (((contracts_[contractId].flags_ -= contract::execute_flag) & contract::schedule_flag) == contract::schedule_flag)
				set_contract_signal(contractId);

		}

		void clear_invocation_count(work_contract_id,
									bool);

		/// \param contractId
		void erase_contract(work_contract_id contractId){
			auto & contract = contracts_[contractId];
			contract.work_ = nullptr;
			release_[contractId] = nullptr;
			exception_[contractId] = nullptr;
			if (auto releaseToken = std::exchange(releaseToken_[contractId], nullptr); releaseToken)
				releaseToken->orphan(); // mark as invalid

			auto [treeIndex, signalIndex] = get_tree_and_signal_index(contractId);
			available_[treeIndex].set(signalIndex);
		}


		/// \return
		work_contract_id get_available_contract() {
			for (auto i = 0ull; i < available_.size(); ++i) {
				auto subTreeIndex (nextAvailableTreeIndex_++ & subTreeMask_);
				if (!available_[subTreeIndex].empty()) {
					if (auto [signalIndex, _] = available_[subTreeIndex].select<largest_child_selector>(0); signalIndex != ~0ull) {
						work_contract_id workContractId(subTreeIndex * signal_tree_capacity);
						workContractId += signalIndex;
						return workContractId;
					}
				}
			}
			return ~0ull;
		}


		/// \param workContractId
		/// \return
		std::tuple<std::uint64_t, std::uint64_t> get_tree_and_signal_index(work_contract_id workContractId) const {
			return {workContractId / signal_tree_capacity, workContractId % signal_tree_capacity};
		}


		using signal_tree_type = signal_tree<64>;
		static auto constexpr signal_tree_capacity = signal_tree_type::capacity;

		std::uint64_t subTreeCount_;
		std::uint64_t subTreeMask_;
		std::uint64_t subTreeShift_;
		std::vector<signal_tree_type> signalTree_;
		std::vector<signal_tree_type> available_;
		std::vector<contract> contracts_;
		std::vector<std::function<void()>> release_;
		std::vector<std::function<void(work_contract_token<T> &, std::exception_ptr)>> exception_;
		std::vector<std::shared_ptr<release_token>> releaseToken_;

		std::mutex mutex_;
		std::atomic<bool> stopped_{false};
		std::atomic<std::uint64_t> nextAvailableTreeIndex_{0};
		static thread_local std::uint64_t tls_biasFlags_;
		std::atomic<std::int64_t> nonZeroCounter_{0};

		void decrement_non_zero_counter() noexcept {
			--nonZeroCounter_;
		}
		void increment_non_zero_counter() noexcept {
			if (nonZeroCounter_++ == 0) {
				waitableState_.notify_all();
			}
		}


		struct {
			std::mutex mutable mutex_;
			std::condition_variable mutable conditionVariable_;

			void notify_all() {
				std::lock_guard lockGuard(mutex_);
				conditionVariable_.notify_all();
			}

			bool wait(work_contract_group const *owner) const {
				if (owner->nonZeroCounter_ == 0) {
					std::unique_lock uniqueLock(mutex_);
					conditionVariable_.wait(uniqueLock, [owner]() { return ((owner->nonZeroCounter_ != 0) || (owner->stopped_)); });
					return (!owner->stopped_);
				}
				return true;
			}

			bool wait_for(
				work_contract_group const *owner,
				std::chrono::nanoseconds duration) const {
				if (owner->nonZeroCounter_ == 0) {
					std::unique_lock uniqueLock(mutex_);
					auto waitSuccess = conditionVariable_.wait_for(uniqueLock, duration, [owner]() mutable { return ((owner->nonZeroCounter_ != 0) || (owner->stopped_)); });
					return ((!owner->stopped_) && (waitSuccess));
				}
				return true;
			}

		} waitableState_;



	};// class work_contract_group

	//=========================================================================
	template <synchronization_mode T>
	class work_contract_group<T>::release_token final :
	    non_copyable,
	    non_movable {
	public:
		release_token() = delete;
		release_token(work_contract_group *workContractGroup)
			    : workContractGroup_(workContractGroup) {}

		/// \param workContract
		/// \return
		bool schedule(work_contract_type const & workContract) {
			std::lock_guard lockGuard(mutex_);
			if (auto workContractGroup = std::exchange(workContractGroup_, nullptr); workContractGroup != nullptr) {
				workContractGroup->release(workContract.get_id());
				return true;
			}
			return false;
		}

		///
		void orphan(){
			std::lock_guard lockGuard(mutex_);
			workContractGroup_ = nullptr;
		}

		bool is_valid() const {
			std::lock_guard lockGuard(mutex_);
			return ((bool)workContractGroup_);
		}

		std::mutex mutable      mutex_;
		work_contract_group *    workContractGroup_{};
	}; // class work_contract_group<>::release_token


	//=============================================================================
	template <synchronization_mode T>
	class work_contract_group<T>::auto_clear_execute_flag {
	public:
		auto_clear_execute_flag(std::uint64_t contractId,
								work_contract_group<T> & owner) :
			contractId_(contractId), owner_(owner) {}
		~auto_clear_execute_flag(){owner_.clear_execute_flag(contractId_);}
	private:
		std::uint64_t contractId_;
		work_contract_group<T> &owner_;
	};

	//=============================================================================
	template <synchronization_mode T>
	class work_contract_group<T>::auto_erase_contract {
	public:
		auto_erase_contract(std::uint64_t contractId,
							work_contract_group<T> & owner)
			:contractId_(contractId),owner_(owner) {}
		~auto_erase_contract() {
			owner_.erase_contract(contractId_);
		}

	private:
		std::uint64_t                   contractId_;
		work_contract_group<T> &         owner_;
	};


	template <synchronization_mode T>
	std::uint64_t thread_local work_contract_group<T>::tls_biasFlags_ = 0;
}

using blocking_work_contract_group = cryptanalysislib::work_contract::work_contract_group<synchronization_mode::blocking>;
using work_contract_group = cryptanalysislib::work_contract::work_contract_group<synchronization_mode::non_blocking>;
