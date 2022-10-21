#include <algorithm>
#include <stdexcept>

#include <fmt/format.h>
#include <xtensor/xtensor.hpp>

#include "ecole/dynamics/nodesel.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/utils.hpp"

namespace ecole::dynamics {
using ActionSet = NodeselDynamics::ActionSet;

auto NodeselDynamics::action_set(scip::Model const& model) -> ActionSet {
	*(fcall.selnode) = nullptr;
	num_to_node.clear();

	if (model.stage() != SCIP_STAGE_SOLVING) {
		return {};
	}

	auto* const scip_ptr = const_cast<SCIP*>(model.get_scip_ptr());

	SCIP_NODE **s_leaves, **s_children, **s_siblings;
	int n_leaves, n_siblings, n_children;

	/* collect leaves, children and siblings data */
	scip::call(SCIPgetOpenNodesData, scip_ptr, &s_leaves, &s_children, &s_siblings, &n_leaves, &n_children, &n_siblings);

	int j;
	SCIP_NODE* node;
	SCIP_Longint nnum;

	// build a map of node number to node
	auto xt_leaves = xt::xtensor<SCIP_Longint, 1>::from_shape({static_cast<unsigned long>(n_leaves)});
	for (j = 0; j < n_leaves; j++) {
		node = s_leaves[j];
		nnum = SCIPnodeGetNumber(node);

		num_to_node[nnum] = node;
		xt_leaves[static_cast<unsigned long>(j)] = nnum;
	}

	auto xt_children = xt::xtensor<SCIP_Longint, 1>::from_shape({static_cast<unsigned long>(n_children)});
	for (j = 0; j < n_children; j++) {
		node = s_children[j];
		nnum = SCIPnodeGetNumber(node);

		num_to_node[nnum] = node;
		xt_children[static_cast<unsigned long>(j)] = nnum;
	}

	auto xt_siblings = xt::xtensor<SCIP_Longint, 1>::from_shape({static_cast<unsigned long>(n_siblings)});
	for (j = 0; j < n_siblings; j++) {
		node = s_siblings[j];
		nnum = SCIPnodeGetNumber(node);

		num_to_node[nnum] = node;
		xt_siblings[static_cast<unsigned long>(j)] = nnum;
	}

	return {{xt_leaves, xt_children, xt_siblings}};
}

auto NodeselDynamics::reset_dynamics(scip::Model& model) -> std::tuple<bool, ActionSet> {
	// fire up the scip_solve in a concurrent coro
	auto maybe_fcall = model.solve_iter(scip::callback::NodeselConstructor{});

	// return respond(model, maybe_fcall);
	if (maybe_fcall.has_value()) {
		// we just got back from the the scip's coro thread
		fcall = std::get<scip::callback::NodeselCall>(maybe_fcall.value());

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
		auto iter = num_to_node.find(static_cast<SCIP_Longint>(node_idx));

		if (iter != num_to_node.end()) {
			*(fcall.selnode) = iter->second;
			// num_to_node.clear();
			scip_result = SCIP_SUCCESS;
		}
	}

	// resume scip's coro
	auto maybe_fcall = model.solve_iter_continue(scip_result);

	// return respond(model, maybe_fcall);
	if (maybe_fcall.has_value()) {
		// we just got back from the the scip's coro thread
		fcall = std::get<scip::callback::NodeselCall>(maybe_fcall.value());

		// return control to python
		if (SCIPgetNNodesLeft(model.get_scip_ptr()) > 0) return {false, action_set(model)};
	}

	// Solving is finished
	return {true, {}};
}

}  // namespace ecole::dynamics
