#pragma once

#include <cstddef>
#include <optional>

#include <scip/scip_tree.h>
#include <xtensor/xtensor.hpp>

#include "ecole/default.hpp"
#include "ecole/dynamics/parts.hpp"
#include "ecole/export.hpp"
#include "ecole/scip/callback.hpp"

namespace ecole::dynamics {

class ECOLE_EXPORT NodeselDynamics : public DefaultSetDynamicsRandomState {
public:
	NodeselDynamics() : fcall{nullptr}, num_to_node{} {};

	// the node to focus on and the allowed nodes (node ids)
	using Action = Defaultable<std::size_t>;
	using ActionSet =
		std::optional<std::tuple<xt::xtensor<std::size_t, 1>, xt::xtensor<std::size_t, 1>, xt::xtensor<std::size_t, 1>>>;

	using DefaultSetDynamicsRandomState::set_dynamics_random_state;

	ECOLE_EXPORT auto reset_dynamics(scip::Model& model) -> std::tuple<bool, ActionSet>;

	ECOLE_EXPORT auto step_dynamics(scip::Model& model, Action maybe_node_idx) -> std::tuple<bool, ActionSet>;

private:
	auto action_set(scip::Model const& model) -> ActionSet;

	// stores the pointer to the selected node variable
	scip::callback::NodeselCall fcall;
	std::map<SCIP_Longint, SCIP_NODE*> num_to_node;
};

}  // namespace ecole::dynamics
