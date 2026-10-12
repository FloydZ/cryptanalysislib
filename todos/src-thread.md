# TODOs: src-thread

34 entries. Format: `- [ ] file:line [MARKER] text`.

## src/thread/execution.h

- [ ] `src/thread/execution.h:52` **TODO** verstehen warum warnung [[nodiscard]] constexpr virtual pool_type pool() const noexcept;
- [ ] `src/thread/execution.h:53` **TODO** verstehen warum warnung [[nodiscard]] constexpr virtual bool par_allowed() const noexcept;
- [ ] `src/thread/execution.h:193` **TODO** ExecPolicy concept
- [ ] `src/thread/execution.h:223` **TODO** doc — \tparam Iterator
- [ ] `src/thread/execution.h:239` **TODO** doc — \tparam Iterator
- [ ] `src/thread/execution.h:317` **TODO** doc — \tparam ExecPolicy
- [ ] `src/thread/execution.h:352` **TODO** `(void)policy;`
- [ ] `src/thread/execution.h:454` **TODO** doc — \tparam ExecPolicy

## src/thread/mythread.h

- [ ] `src/thread/mythread.h:227` **TODO** remove the linked list and use the fs register to store information

## src/thread/old/heartbeat_scheduler.h

- [ ] `src/thread/old/heartbeat_scheduler.h:29` **TODO** apple doesnt support jthread
- [ ] `src/thread/old/heartbeat_scheduler.h:32` **TODO** translate the spice binarytree example
- [ ] `src/thread/old/heartbeat_scheduler.h:35` **TODO** use this future thingy, probably want to switch to std::future
- [ ] `src/thread/old/heartbeat_scheduler.h:104` **TODO** implement — `void clear_task_queue() noexcept {`
- [ ] `src/thread/old/heartbeat_scheduler.h:108` **TODO** implement — `void inline pause() noexcept {`
- [ ] `src/thread/old/heartbeat_scheduler.h:238` **TODO** this = Job();                  // Reset the current job.
- [ ] `src/thread/old/heartbeat_scheduler.h:331` **TODO** correct reference
- [ ] `src/thread/old/heartbeat_scheduler.h:335` **TODO** return std::invoke(func, &t, arg...);
- [ ] `src/thread/old/heartbeat_scheduler.h:411` **TODO** ? — `std::unique_lock lock(mutex);`
- [ ] `src/thread/old/heartbeat_scheduler.h:484` **TODO** memory allocator
- [ ] `src/thread/old/heartbeat_scheduler.h:489` **TODO** this seams wrong
- [ ] `src/thread/old/heartbeat_scheduler.h:510` **TODO** better ways.
- [ ] `src/thread/old/heartbeat_scheduler.h:552` **TODO** best_worker->shared_job->reset();

## src/thread/old/heartbeat_scheduler_v2.h

- [ ] `src/thread/old/heartbeat_scheduler_v2.h:159` **TODO** weak probably not needed?
- [ ] `src/thread/old/heartbeat_scheduler_v2.h:281` **TODO** `using T = uint64_t;`

## src/thread/performance.h

- [ ] `src/thread/performance.h:32` **TODO** apple does not suport jthread
- [ ] `src/thread/performance.h:86` **TODO** what happen in this case
- [ ] `src/thread/performance.h:183` **TODO** more information: like: tid
- [ ] `src/thread/performance.h:256` **TODO** with fork etc we can handle multiple streams, but for now
- [ ] `src/thread/performance.h:283` **TODO** nice printing

## src/thread/simple.h

- [ ] `src/thread/simple.h:290` **TODO** replace with `ConstVectorQueue`

## src/thread/steal.h

- [ ] `src/thread/steal.h:43` **TODO** use the MOVE operator from SimpleScheduler

## src/thread/work_contract.h

- [ ] `src/thread/work_contract.h:15` **TODO** move somewhere usefull
- [ ] `src/thread/work_contract.h:159` **TODO** `bool deschedule();`
- [ ] `src/thread/work_contract.h:489` **TODO** remove — the expected case. invoke the work function
