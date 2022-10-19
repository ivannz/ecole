#include <algorithm>
#include <stdexcept>

#include <fmt/format.h>
#include <xtensor/xtensor.hpp>

#include "ecole/dynamics/nodesel.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/utils.hpp"

namespace ecole::dynamics {

auto NodeselDynamics::action_set(scip::Model const& model) -> NodeselDynamics::ActionSet {
	if (model.stage() != SCIP_STAGE_SOLVING) {
		return {};
	}

	auto* const scip_ptr = const_cast<SCIP*>(model.get_scip_ptr());
	SCIP_NODE **leaves, **children, **siblings;
	int n_leaves, n_siblings, n_children;

	/* collect leaves, children and siblings data */
	scip::call(SCIPgetOpenNodesData, scip_ptr, &leaves, &children, &siblings, &n_leaves, &n_children, &n_siblings);

	nonstd::span<SCIP_NODE*> s_leaves = {leaves, static_cast<std::size_t>(n_leaves)};
	auto xt_leaves = xt::xtensor<std::size_t, 1>::from_shape({static_cast<unsigned long>(n_leaves)});
	std::transform(s_leaves.begin(), s_leaves.end(), xt_leaves.begin(), SCIPnodeGetNumber);

	nonstd::span<SCIP_NODE*> s_children = {children, static_cast<std::size_t>(n_children)};
	auto xt_children = xt::xtensor<std::size_t, 1>::from_shape({static_cast<unsigned long>(n_children)});
	std::transform(s_children.begin(), s_children.end(), xt_children.begin(), SCIPnodeGetNumber);

	nonstd::span<SCIP_NODE*> s_siblings = {siblings, static_cast<std::size_t>(n_siblings)};
	auto xt_siblings = xt::xtensor<std::size_t, 1>::from_shape({static_cast<unsigned long>(n_siblings)});
	std::transform(s_siblings.begin(), s_siblings.end(), xt_siblings.begin(), SCIPnodeGetNumber);

	return {{xt_leaves, xt_children, xt_siblings}};
}

auto NodeselDynamics::reset_dynamics(scip::Model& model) -> std::tuple<bool, ActionSet> {
	// fire up the scip_solve in a concurrent coro
	auto fcall = model.solve_iter(scip::callback::NodeselConstructor{});
	if (fcall.has_value()) {
		// we just got back from the the scip's coro thread
		this->selnode = std::get<scip::callback::NodeselCall>(fcall.value()).selnode;
		*(this->selnode) = nullptr;

		// return control to python
		if (SCIPgetNNodesLeft(model.get_scip_ptr()) > 0) return {false, action_set(model)};
	}

	// Solving is finished
	return {true, {}};
}

auto NodeselDynamics::step_dynamics(scip::Model& model, Defaultable<std::size_t> maybe_node_idx)
	-> std::tuple<bool, ActionSet> {
	auto scip_result = SCIP_DIDNOTRUN;
	if (std::holds_alternative<std::size_t>(maybe_node_idx)) {
		auto const node_idx = std::get<std::size_t>(maybe_node_idx);
		auto* const scip_ptr = const_cast<SCIP*>(model.get_scip_ptr());

		// build a map of node number to node
		SCIP_NODE **leaves, **children, **siblings;
		int n_leaves, n_siblings, n_children;
		scip::call(SCIPgetOpenNodesData, scip_ptr, &leaves, &children, &siblings, &n_leaves, &n_children, &n_siblings);

		std::map<SCIP_Longint, SCIP_NODE*> num_to_node = {};
		for (int n = 0; n < n_leaves; ++n) {
			num_to_node[SCIPnodeGetNumber(leaves[n])] = leaves[n];
		}
		for (int n = 0; n < n_siblings; ++n) {
			num_to_node[SCIPnodeGetNumber(siblings[n])] = siblings[n];
		}
		for (int n = 0; n < n_children; ++n) {
			num_to_node[SCIPnodeGetNumber(children[n])] = children[n];
		}

		// node selection
		*(this->selnode) = num_to_node[static_cast<SCIP_Longint>(node_idx)];
		scip_result = SCIP_SUCCESS;
	}

	auto fcall = model.solve_iter_continue(scip_result);

	// While solving is not finished.
	if (fcall.has_value()) {
		this->selnode = std::get<scip::callback::NodeselCall>(fcall.value()).selnode;

		*(this->selnode) = nullptr;
		if (SCIPgetNNodesLeft(model.get_scip_ptr()) > 0) return {false, action_set(model)};
	}

	// Solving is finished.
	return {true, {}};
}

}  // namespace ecole::dynamics
