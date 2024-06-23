#pragma once

// Convenience header file that includes all loop_fusion implementations.

#include "loop_fusion/compiletime/loop_fusion.hpp"
#include "loop_fusion/main_range/loop_fusion.hpp"
#include "loop_fusion/runtime/loop_fusion.hpp"

namespace loop_fusion::simple {
// for backwards-compatibility
using namespace main_range;
} // namespace loop_fusion::simple
