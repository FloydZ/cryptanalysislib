#ifndef CRYPTANALYSISLIB_THREAD_H
#define CRYPTANALYSISLIB_THREAD_H

#if __cplusplus > 201709L
#include <cstddef>
template<class Scheduler>
concept SchedulerAble = requires(Scheduler a) {
	requires requires(const size_t i) {
        // returning a std::future
        //a.submit();
        // returning void
        //a.submit_detach();

        a.wait_for_tasks();

        a.pause();
        a.unpause();
        a.is_paused();
        
        a.clear_tasks();
        a.get_num_queued_tasks();
        a.get_num_running_tasks();
        a.get_num_tasks();
        a.get_num_thread();
	};
};
#endif 

#include "mythread.h"
#include "performance.h"
#include "steal.h"
#include "simple.h"
#include "execution.h"
#include "work_contract.h"
#endif
