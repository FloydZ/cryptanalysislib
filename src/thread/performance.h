#pragma once 

// apple doesnt provide jthread!
#ifndef __APPLE__
#include <thread>

// NOTE only available on unix
#include <assert.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <rfl.hpp>
#include <rfl/json.hpp>

namespace cryptanalysislib {
	namespace details {

#ifdef __cpp_lib_move_only_function
		using default_function_type = std::move_only_function<void()>;
#else
		using default_function_type = std::function<void()>;
#endif

		/// TODO: apple does not suport jthread
#ifdef __APPLE__
		using default_thread_type = std::thread;
#else
		using default_thread_type = std::jthread;
#endif
	};// namespace details



///
class SchedulerConfig {
public:
	constexpr static bool enable_try_block = false;
	constexpr static bool enable_remote_view = false;
};
constexpr static SchedulerConfig schedulerConfig;
}; // end namespace cryptanalysislib


namespace cryptanalysislib {
	// NOTE: this is linux only
	// NOTE: this is needed, as the real internal `rusage`
	// uses strange unions, which are not supported by
	// `reflect-cpp`
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

		///// get the needed values
		void gather() noexcept {
			if (getrusage(RUSAGE_THREAD, (rusage *) &data) != 0) {
				// TODO: what happen in this case
				assert(false);
			}
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


	/// NOTE: just the data container. To access it use the Manager
	/// class. This is needed for the reflect-cpp framework
	struct SchedulerPerformance {
		// per thread information
		std::vector<SchedulerThreadLoad> schedulerThreadLoad;

		// global information
		int number_active_threads = 0;
		int number_enqueud_tasks = 0;

		// TODO more information: like: tid
	};

	/// Manager class
	struct SchedulerPerformanceManager {
	private:
		SchedulerPerformance schedulerPerformance;

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
			const int servlen = strlen(serv_addr.sun_path) + sizeof(serv_addr.sun_family);

		    const int one = 1;
		    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(int)) < 0) {
		    	std::cerr << "setsockopt(SO_REUSEADDR) failed" << std::endl;
		    	return;
		    }

		    if (unlinkat(sockfd, socket_path, 0) < -1) {
		    	std::cerr << "unlinkat failed" << std::endl;
		    	return;
		    }

			if (server) {

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

					// TODO with fork etc we can handle multiple streams, but for now
					// its fine
					while (true) {
						memset(buf, (char) 0, buffer_size);
						const uint32_t n = read(newsockfd, buf, buffer_size);
						if (n == 0) { break; }
						assert(n < buffer_size);
						std::cout << "recv: " << std::endl;
						std::cout << buf << std::endl;

						schedulerPerformance = rfl::json::read<SchedulerPerformance>(buf).value();
						// TODO nice printing
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
			const auto data = rfl::json::write(schedulerPerformance);
			std::cout << "sending data:" << std::endl;
			std::cout << data << std::endl;

			const int k = write(sockfd, data.data(), data.size());
			if (k < 0) {
				std::cout << "Error writing" << std::endl;
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

#endif
