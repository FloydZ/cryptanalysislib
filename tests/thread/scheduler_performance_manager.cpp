#include "thread/thread.h"

using namespace cryptanalysislib;

int main() {
	auto s = SchedulerPerformanceManager(true);
	s.serve();
}
