#pragma once

#include <cstddef>
#include <optional>

#include <scip/scip_tree.h>
#include <xtensor/xtensor.hpp>

#include "ecole/default.hpp"
#include "ecole/dynamics/parts.hpp"
#include "ecole/export.hpp"

namespace ecole::dynamics {

class ECOLE_EXPORT NodeselDynamics : public DefaultSetDynamicsRandomState {
public:
	NodeselDynamics() : selnode{nullptr} {};

	// the node to focus on and the allowed nodes (node ids)
	using Action = Defaultable<std::size_t>;
	using ActionSet =
		std::optional<std::tuple<xt::xtensor<std::size_t, 1>, xt::xtensor<std::size_t, 1>, xt::xtensor<std::size_t, 1>>>;

	using DefaultSetDynamicsRandomState::set_dynamics_random_state;

	ECOLE_EXPORT auto reset_dynamics(scip::Model& model) -> std::tuple<bool, ActionSet>;

	ECOLE_EXPORT auto step_dynamics(scip::Model& model, Action maybe_node_idx) -> std::tuple<bool, ActionSet>;

private:
	SCIP_NODE** selnode;

	auto action_set(scip::Model const& model) -> NodeselDynamics::ActionSet;
};

}  // namespace ecole::dynamics
