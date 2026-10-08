#include <unistd.h>

#include "thread/thread.h"

using namespace cryptanalysislib;

constexpr static SchedulerConfig remote_config{.enable_try_block = false,
                                               .enable_remote_view = true};

/// starts a server, runs a scheduler with remote view, and waits until the
/// scheduler closed its socket.
template<typename Scheduler>
static int run() {
	auto *server = new SchedulerPerformanceManager(true);
	// wait until the server listens
	usleep(200000);
	int ret;
	{
		Scheduler s(1);
		auto f = s.submit([]() { return 42; });
		ret = f.get() == 42 ? 0 : 1;
	}

	// returns once the client closed its socket
	server->serve();
	delete server;
	return ret;
}

int main() {
	int ret = run<StealingScheduler<details::default_thread_type,
	                                details::default_function_type,
	                                remote_config>>();
	ret |= run<SimpleScheduler<details::default_thread_type,
	                           details::default_function_type,
	                           remote_config>>();
	return ret;
}
