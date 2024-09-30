#include "thread/scheduler.h"

using namespace cryptanalysislib;

int main() {
	auto s = SchedulerPerformanceManager(true);
	s.serve();
}
