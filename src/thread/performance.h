#pragma once 

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

// NOTE only available on unix
#include <assert.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif

// `RUSAGE_THREAD` is Linux-only; elsewhere fall back to process-wide usage
#ifndef RUSAGE_THREAD
#define RUSAGE_THREAD RUSAGE_SELF
#endif

#include <version>

namespace cryptanalysislib {
	namespace details {

#ifdef __cpp_lib_move_only_function
		using default_function_type = std::move_only_function<void()>;
#else
		using default_function_type = std::function<void()>;
#endif

		/// NOTE: `std::jthread` only if the standard library has it, e.g.
		/// 	older libc++ (Apple clang) does not.
#ifdef __cpp_lib_jthread
		using default_thread_type = std::jthread;
#else
		using default_thread_type = std::thread;
#endif
	};// namespace details



/// NOTE: non-static members, so a custom config can be passed, e.g.
/// 	`constexpr static SchedulerConfig c{.enable_remote_view = true};`.
/// 	Before, both were `constexpr static` and fixed to `false`.
class SchedulerConfig {
public:
	const bool enable_try_block = false;
	const bool enable_remote_view = false;
};
inline constexpr SchedulerConfig schedulerConfig;
}; // end namespace cryptanalysislib


namespace cryptanalysislib {
	// NOTE: this is linux only
	// NOTE: plain copy of the internal `rusage`, which uses unions.
	struct rusage__ {
		struct timeval ru_utime;
		struct timeval ru_stime;
		long int ru_maxrss;
		long int ru_ixrss;
		long int ru_idrss;
		long int ru_isrss;
		long int ru_minflt;
		long int ru_majflt;
		long int ru_nswap;
		long int ru_inblock;
		long int ru_oublock;
		long int ru_msgsnd;
		long int ru_msgrcv;
		long int ru_nsignals;
		long int ru_nvcsw;
		long int ru_nivcsw;
	};


	class SchedulerThreadLoad {
	public:
		rusage__ data;

		/// kernel thread id of the thread, which called `gather` last
		uint64_t tid = 0;

		/// get the needed values
		/// NOTE: `getrusage` only fails for an invalid `who` or pointer. In
		/// 	that case all values are zeroed.
		/// \return false if the values could not be gathered
		bool gather() noexcept {
			tid = current_tid();
			if (getrusage(RUSAGE_THREAD, (rusage *) &data) != 0) {
				data = rusage__{};
				return false;
			}
			return true;
		}

		/// \return the kernel thread id of the calling thread
		[[nodiscard]] static inline uint64_t current_tid() noexcept {
#if defined(__linux__)
			return uint64_t(syscall(SYS_gettid));
#elif defined(__APPLE__)
			uint64_t t = 0;
			pthread_threadid_np(nullptr, &t);
			return t;
#else
			return 0;
#endif
		}

		/// \return maximum resident set size
		[[nodiscard]] constexpr inline uint64_t maxrss() const noexcept {
			return data.ru_maxrss;
		}

		/// \return integral shared memory size
		[[nodiscard]] constexpr inline uint64_t ixrss() const noexcept {
			return data.ru_ixrss;
		}

		/// \return integral unshared data size
		[[nodiscard]] constexpr inline uint64_t idrss() const noexcept {
			return data.ru_idrss;
		}

		/// \return integral unshared stack size
		[[nodiscard]] constexpr inline uint64_t isrss() const noexcept {
			return data.ru_isrss;
		}

		/// \return page reclaims (soft page faults)
		[[nodiscard]] constexpr inline uint64_t minflt() const noexcept {
			return data.ru_minflt;
		}

		/// \return page faults (hard page faults )
		[[nodiscard]] constexpr inline uint64_t majflt() const noexcept {
			return data.ru_majflt;
		}

		/// \return number of swaps
		[[nodiscard]] constexpr inline uint64_t nswap() const noexcept {
			return data.ru_nswap;
		}

		/// \return block input operations
		[[nodiscard]] constexpr inline uint64_t inblock() const noexcept {
			return data.ru_inblock;
		}

		/// \return block output operations
		[[nodiscard]] constexpr inline uint64_t outblock() const noexcept {
			return data.ru_oublock;
		}

		/// \return IPC messages sent
		[[nodiscard]] constexpr inline uint64_t msgsnd() const noexcept {
			return data.ru_msgsnd;
		}

		/// \return IPC messages received
		[[nodiscard]] constexpr inline uint64_t msgrcv() const noexcept {
			return data.ru_msgrcv;
		}

		/// \return signal received
		[[nodiscard]] constexpr inline uint64_t nsignals() const noexcept {
			return data.ru_nsignals;
		}

		/// \return voluntary context switches
		[[nodiscard]] constexpr inline uint64_t nvcsw() const noexcept {
			return data.ru_nvcsw;
		}

		/// \return involuntary context switches
		[[nodiscard]] constexpr inline uint64_t nivcsw() const noexcept {
			return data.ru_nivcsw;
		}

		/// \return user time in microseconds
		[[nodiscard]] constexpr inline uint64_t usertime() const noexcept {
			return data.ru_utime.tv_sec * 1000000 + data.ru_utime.tv_usec;
		}

		/// \return system time
		[[nodiscard]] constexpr inline uint64_t systime() const noexcept {
			return data.ru_stime.tv_sec * 1000000 + data.ru_stime.tv_usec;
		}
	};


	// NOTE: the per thread data is send as raw bytes over the socket
	static_assert(std::is_trivially_copyable_v<SchedulerThreadLoad>);

	/// NOTE: just the data container. To access it use the Manager
	/// class.
	struct SchedulerPerformance {
		// per thread information
		std::vector<SchedulerThreadLoad> schedulerThreadLoad;

		// global information
		int number_active_threads = 0;
		int number_enqueud_tasks = 0;

		// NOTE: the thread ids are in `SchedulerThreadLoad::tid`
	};

	/// Manager class
	struct SchedulerPerformanceManager {
	private:
		SchedulerPerformance schedulerPerformance;

		/// wire format of a message: this header, followed by
		/// `nr_threads` raw `SchedulerThreadLoad` entries.
		struct MessageHeader {
			uint64_t nr_threads;
			int number_active_threads;
			int number_enqueud_tasks;
		};

		const bool server;

		// socket communication
		constexpr static char *socket_path = (char *) "/tmp/cryptanalysislib_scheduler.socket";
		constexpr static size_t buffer_size = 8096;
		int sockfd;
		std::thread server_thread;

	public:
		/// either create a server or client instance. The client is sending data.
		/// The server is receiving data.
		/// \param server
		explicit SchedulerPerformanceManager(const bool server) noexcept : server(server) {
			if ((sockfd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
				std::cout << "ERROR: Creating socket" << std::endl;
				exit(1);
			}

			struct sockaddr_un serv_addr;
			bzero((char *) &serv_addr, sizeof(serv_addr));
			serv_addr.sun_family = AF_UNIX;
			strcpy(serv_addr.sun_path, socket_path);
			// NOTE: `sun_path` does not start at `sizeof(sun_family)` on BSD/macOS
			// 	(`sun_len` precedes the 1 byte `sun_family`). Before, the address
			// 	was one byte too short there, i.e. the last character of the
			// 	path was cut off, and the server's `unlink` missed the stale file.
			const socklen_t servlen = offsetof(struct sockaddr_un, sun_path) + strlen(serv_addr.sun_path);

		    const int one = 1;
		    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(int)) < 0) {
		    	std::cerr << "setsockopt(SO_REUSEADDR) failed" << std::endl;
		    	return;
		    }

			if (server) {
				// NOTE: only the server removes a stale socket file. Before, this
				// 	was done by the client too, which deleted the server's socket
				// 	right before `connect` (and `< -1` never failed).
				if ((unlink(socket_path) < 0) && (errno != ENOENT)) {
					std::cerr << "unlink failed" << std::endl;
					return;
				}

				if (bind(sockfd, (struct sockaddr *) &serv_addr, servlen) < 0) {
					std::cout << "ERROR: binding socket" << std::endl;
					exit(1);
				}

				std::cout << "starting server" << std::endl;
				auto server_worker = [&]() -> void {
					if (listen(sockfd, 5) != 0) {
						std::cout << "ERROR: listen: " << strerror(errno) << std::endl;
						exit(1);
					}

					struct sockaddr_un cli_addr;
					char buf[buffer_size];
					socklen_t clilen = sizeof(cli_addr);

					int newsockfd = accept((int) sockfd, (struct sockaddr *) &cli_addr, &clilen);
					if (newsockfd < 0) {
						std::cout << "ERROR: accepting: " << strerror(errno) << std::endl;
						exit(1);
					}

					// NOTE: exactly one client (the scheduler) is served, `serve()`
					// 	returns once it disconnects. Several clients would need an
					// 	`accept` loop with a stop condition (or `fork`).
					// NOTE: a stream socket has no message boundaries: several
					// 	messages can arrive in one `read`, or one message in
					// 	several. Hence a message is only parsed, once all
					// 	bytes announced by its header are received (see `send`).
					std::string pending;
					while (true) {
						const ssize_t n = read(newsockfd, buf, buffer_size);
						if (n <= 0) { break; }
						pending.append(buf, n);

						while (pending.size() >= sizeof(MessageHeader)) {
							MessageHeader h;
							std::memcpy(&h, pending.data(), sizeof(MessageHeader));
							const size_t data_size = h.nr_threads * sizeof(SchedulerThreadLoad);
							if (pending.size() < sizeof(MessageHeader) + data_size) { break; }

							auto &loads = schedulerPerformance.schedulerThreadLoad;
							loads.resize(h.nr_threads);
							std::memcpy((void *)loads.data(), pending.data() + sizeof(MessageHeader), data_size);
							schedulerPerformance.number_active_threads = h.number_active_threads;
							schedulerPerformance.number_enqueud_tasks = h.number_enqueud_tasks;
							pending.erase(0, sizeof(MessageHeader) + data_size);
							print();
						}
					}

					std::cout << "server closing" << std::endl;
					close(newsockfd);
				};
				server_thread = std::thread(server_worker);
			} else {
				if (connect(sockfd, (struct sockaddr *) &serv_addr, servlen) < 0) {
					std::cout << "ERROR Connecting: " << strerror(errno) << std::endl;
					exit(1);
				}
			}
		}

		~SchedulerPerformanceManager() {
			std::cout << "closing socket" << std::endl;
			close(sockfd);
		}

		void print() noexcept {
			std::cout << "#Active Threads: " << schedulerPerformance.number_active_threads << std::endl;
			std::cout << "#Enqueud Tasks: " << schedulerPerformance.number_enqueud_tasks << std::endl;
			for (const auto &s: schedulerPerformance.schedulerThreadLoad) {
				std::cout << "Thread Id: " << s.tid << std::endl;
				std::cout << "UserTime: " << s.usertime() << std::endl;
				std::cout << "SystemTime: " << s.systime() << std::endl;
				std::cout << "Max Resident Set Size: " << s.maxrss() << std::endl;
				std::cout << "Integral Shared Memory Size: " << s.ixrss() << std::endl;
				std::cout << "Integral Unshared Memory Size: " << s.idrss() << std::endl;
				std::cout << "Integral Unshared Stack Size: " << s.isrss() << std::endl;
				std::cout << "Soft Page Faults: " << s.minflt() << std::endl;
				std::cout << "Hard Page Faults: " << s.majflt() << std::endl;
				std::cout << "#Swaps: " << s.nswap() << std::endl;
				std::cout << "#Block Input Operations: " << s.inblock() << std::endl;
				std::cout << "#Block Output Operations: " << s.outblock() << std::endl;
				std::cout << "#Message Sent: " << s.msgsnd() << std::endl;
				std::cout << "#Message Received: " << s.msgrcv() << std::endl;
				std::cout << "#Signals Received: " << s.nsignals() << std::endl;
				std::cout << "Voluntary Context Switches: " << s.nvcsw() << std::endl;
				std::cout << "Involuntary Context Switches:: " << s.nivcsw() << std::endl;
				std::cout << std::endl;
			}
		}

		/// start running the scheduler
		void serve() noexcept {
			assert(server);
			server_thread.join();
		}

		/// \param nr_threads[i]: set the number of threads available to the
		///     scheduler.
		constexpr void resize(const uint32_t nr_threads) noexcept {
			assert(nr_threads);
			schedulerPerformance.schedulerThreadLoad.resize(nr_threads);
		}

		/// write the gathered benchmark information.
		void send() noexcept {
			const auto &loads = schedulerPerformance.schedulerThreadLoad;
			const MessageHeader h{loads.size(),
			                      schedulerPerformance.number_active_threads,
			                      schedulerPerformance.number_enqueud_tasks};
			const size_t data_size = loads.size() * sizeof(SchedulerThreadLoad);

			std::string msg(sizeof(MessageHeader) + data_size, '\0');
			std::memcpy(msg.data(), &h, sizeof(MessageHeader));
			std::memcpy(msg.data() + sizeof(MessageHeader), (const void *)loads.data(), data_size);

			size_t written = 0;
			while (written < msg.size()) {
				const ssize_t k = write(sockfd, msg.data() + written, msg.size() - written);
				if (k < 0) {
					std::cout << "Error writing" << std::endl;
					return;
				}
				written += k;
			}
		}

		/// gather performance metrics for thread `tid`
		/// \param tid thread id
		constexpr inline void gather(const uint32_t tid) noexcept {
			assert(tid < schedulerPerformance.schedulerThreadLoad.size());
			schedulerPerformance.schedulerThreadLoad[tid].gather();
		}

		/// set performance metric
		/// this funcitons should only be called by thread 0
		/// \param number_active_threads
		/// \param number_enqueud_tasks
		constexpr inline void gather(const uint32_t number_active_threads,
		                             const uint32_t number_enqueud_tasks) noexcept {
			schedulerPerformance.schedulerThreadLoad[0].gather();
			schedulerPerformance.number_active_threads = number_active_threads;
			schedulerPerformance.number_enqueud_tasks = number_enqueud_tasks;
			send();
		}

		/// \param tid thread id
		/// \return
		constexpr inline SchedulerThreadLoad &operator[](const uint32_t tid) noexcept {
			assert(tid < schedulerPerformance.schedulerThreadLoad.size());
			return schedulerPerformance.schedulerThreadLoad[tid];
		}
	};
}
