#pragma once

#include "loss.h"

namespace rl_tools::foundation_policy::diff_pre_training{
    // This header intentionally stays small for the first MVP. The training loop
    // lives in main.cpp while the differentiable physics loss is isolated in
    // loss.h so the missing Jacobian terms can be extended without touching the
    // executable wiring.
}
