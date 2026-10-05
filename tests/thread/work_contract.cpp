#include <gtest/gtest.h>
#include <thread>

#include "thread/work_contract.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

int blocking_execute_after_scheduled() {
    blocking_work_contract_group workContractGroup;
    std::jthread workerThread([&](std::stop_token stopToken) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        while (!stopToken.stop_requested())
            workContractGroup.execute_next_contract();
    });
    std::atomic<bool> executed{false};
    auto workContract = workContractGroup.create_contract(
        [&](auto & token){std::cout << "contract executed\n"; executed = true;}
    );
    workContract.schedule();
    while (!executed)
        ;
    return 0;
}

TEST(WorkContract, Simple) {
    blocking_execute_after_scheduled();

    // create work contract group
    blocking_work_contract_group workContractGroup;

    // create async worker thread to service scheduled contracts
    std::jthread workerThread([&](std::stop_token stopToken) {
        while (!stopToken.stop_requested()) {
            workContractGroup.execute_next_contract();
        }
    });

    std::atomic<bool> executed{false};
    auto workContract = workContractGroup.create_contract(
        [&](auto & token){std::cout << "executed\n"; executed = true;}
    );

    for (auto i = 0; i < 1000; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "scheduled i = " << i << "   ";
        workContract.schedule();
        while (!executed);
        executed = false;
    }
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
