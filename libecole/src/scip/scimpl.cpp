#include <algorithm>
#include <cassert>
#include <mutex>
#include <scip/type_result.h>
#include <scip/type_retcode.h>
#include <tuple>
#include <type_traits>
#include <utility>

#include <objscip/objbranchrule.h>
#include <objscip/objheur.h>
#include <objscip/objnodesel.h>
#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/type_timing.h>

#include "ecole/scip/callback.hpp"
#include "ecole/scip/scimpl.hpp"
#include "ecole/scip/utils.hpp"
#include "ecole/utility/coroutine.hpp"

namespace ecole::scip {

/*************************************
 *  Definition of reverse Callbacks  *
 *************************************/

namespace {

using Controller = utility::Coroutine<callback::DynamicCall, SCIP_RESULT>;
using Executor = typename Controller::Executor;

/**
 * Function to add a callback to SCIP.
 *
 * Needs to be implemented by all reverse callbacks.
 */
template <callback::Type type>
auto include_reverse_callback(SCIP* scip, std::weak_ptr<Executor> executor, callback::Constructor<type> args) -> void;

/**
 * In a callback send Callback type and wait for result.
 *
 * This function is commonly used inside reverse callbacks to wait for user action (the result).
 * For user to make the proper action, they need to know on which callback SCIP stoped (the stop location).
 * This function will pass the current call function arguments to the coroutine and wait for the result.
 */
template <callback::Type type>
auto handle_executor(SCIP* scip, std::weak_ptr<Executor>& weak_executor, callback::Call<type> call) noexcept
	-> std::tuple<SCIP_RETCODE, SCIP_RESULT> {
	if (weak_executor.expired()) {
		return {SCIP_OKAY, SCIP_DIDNOTRUN};
	}
	try {
		return std::visit(
			[&](auto result_or_stop) -> std::tuple<SCIP_RETCODE, SCIP_RESULT> {
				using StopToken = Executor::StopToken;
				if constexpr (std::is_same_v<decltype(result_or_stop), StopToken>) {
					return {SCIPinterruptSolve(scip), SCIP_DIDNOTRUN};
				} else {
					return {SCIP_OKAY, result_or_stop};
				}
			},
			weak_executor.lock()->yield(call));
	} catch (...) {
		return {SCIP_ERROR, SCIP_DIDNOTRUN};
	}
}

class ReverseBranchrule : public ::scip::ObjBranchrule {
public:
	ReverseBranchrule(
		SCIP* scip,
		int priority,
		int maxdepth,
		SCIP_Real maxbounddist,
		std::weak_ptr<Executor> weak_executor) :
		ObjBranchrule{
			scip,
			name(callback::Type::Branchrule),
			"Branchrule that wait for another thread to make the branching.",
			priority,
			maxdepth,
			maxbounddist},
		m_weak_executor{std::move(weak_executor)} {}

	/** branching execution method for fractional LP solutions
	 *
	 *  input:
	 *  - scip            : SCIP main data structure
	 *  - branchrule      : the branching rule itself
	 *  - allowaddcons    : is the branching rule allowed to add constraints to the current node in order to cut off the
	 *                      current solution instead of creating a branching?
	 *  - result          : pointer to store the result of the branching call
	 *
	 *  possible return values for *result (if more than one applies, the first in the list should be used):
	 *  - SCIP_CUTOFF     : the current node was detected to be infeasible
	 *  - SCIP_CONSADDED  : an additional constraint (e.g. a conflict constraint) was generated; this result code must not
	 *                      be returned, if allowaddcons is FALSE
	 *  - SCIP_REDUCEDDOM : a domain was reduced that rendered the current LP solution infeasible
	 *  - SCIP_SEPARATED  : a cutting plane was generated
	 *  - SCIP_BRANCHED   : branching was applied
	 *  - SCIP_DIDNOTFIND : the branching rule searched, but did not find a branching
	 *  - SCIP_DIDNOTRUN  : the branching rule was skipped
	 */
	auto scip_execlp(SCIP* scip, SCIP_BRANCHRULE* /*branchrule*/, SCIP_Bool allow_add_constraints, SCIP_RESULT* result)
		-> SCIP_RETCODE override {
		using Where = callback::BranchruleCall::Where;
		return scip_exec_any(scip, result, {static_cast<bool>(allow_add_constraints), Where::LP});
	}

	/** branching execution method for external candidates
	 *
	 *  input:
	 *  - scip            : SCIP main data structure
	 *  - branchrule      : the branching rule itself
	 *  - allowaddcons    : is the branching rule allowed to add constraints to the current node in order to cut off the
	 *                      current solution instead of creating a branching?
	 *  - result          : pointer to store the result of the branching call
	 *
	 *  possible return values for *result (if more than one applies, the first in the list should be used):
	 *  - SCIP_CUTOFF     : the current node was detected to be infeasible
	 *  - SCIP_CONSADDED  : an additional constraint (e.g. a conflict constraint) was generated; this result code must not
	 *                      be returned, if allowaddcons is FALSE
	 *  - SCIP_REDUCEDDOM : a domain was reduced that rendered the current pseudo solution infeasible
	 *  - SCIP_BRANCHED   : branching was applied
	 *  - SCIP_DIDNOTFIND : the branching rule searched, but did not find a branching
	 *  - SCIP_DIDNOTRUN  : the branching rule was skipped
	 */
	auto scip_execext(SCIP* scip, SCIP_BRANCHRULE* /*branchrule*/, SCIP_Bool allow_add_constraints, SCIP_RESULT* result)
		-> SCIP_RETCODE override {
		using Where = callback::BranchruleCall::Where;
		return scip_exec_any(scip, result, {static_cast<bool>(allow_add_constraints), Where::External});
	}

	/** branching execution method for not completely fixed pseudo solutions
	 *
	 *  input:
	 *  - scip            : SCIP main data structure
	 *  - branchrule      : the branching rule itself
	 *  - allowaddcons    : is the branching rule allowed to add constraints to the current node in order to cut off the
	 *                      current solution instead of creating a branching?
	 *  - result          : pointer to store the result of the branching call
	 *
	 *  possible return values for *result (if more than one applies, the first in the list should be used):
	 *  - SCIP_CUTOFF     : the current node was detected to be infeasible
	 *  - SCIP_CONSADDED  : an additional constraint (e.g. a conflict constraint) was generated; this result code must not
	 *                      be returned, if allowaddcons is FALSE
	 *  - SCIP_REDUCEDDOM : a domain was reduced that rendered the current pseudo solution infeasible
	 *  - SCIP_BRANCHED   : branching was applied
	 *  - SCIP_DIDNOTFIND : the branching rule searched, but did not find a branching
	 *  - SCIP_DIDNOTRUN  : the branching rule was skipped
	 */
	auto scip_execps(SCIP* scip, SCIP_BRANCHRULE* /*branchrule*/, SCIP_Bool allow_add_constraints, SCIP_RESULT* result)
		-> SCIP_RETCODE override {
		using Where = callback::BranchruleCall::Where;
		return scip_exec_any(scip, result, {static_cast<bool>(allow_add_constraints), Where::Pseudo});
	}

private:
	std::weak_ptr<Executor> m_weak_executor;

	auto scip_exec_any(SCIP* scip, SCIP_RESULT* result, callback::BranchruleCall call) -> SCIP_RETCODE {
		auto retcode = SCIP_OKAY;
		std::tie(retcode, *result) = handle_executor(scip, m_weak_executor, call);
		return retcode;
	}
};

template <>
auto include_reverse_callback<callback::Type::Branchrule>(
	SCIP* scip,
	std::weak_ptr<Executor> executor,
	callback::Constructor<callback::Type::Branchrule> args) -> void {
	scip::call(
		SCIPincludeObjBranchrule,
		scip,
		new ReverseBranchrule(scip, args.priority, args.max_depth, args.max_bound_distance, std::move(executor)),
		true);
}  // NOLINT

class ReverseHeur : public ::scip::ObjHeur {
public:
	ReverseHeur(
		SCIP* scip,
		int priority,
		int freq,
		int freqofs,
		int maxdepth,
		SCIP_HEURTIMING timingmask,
		std::weak_ptr<Executor> weak_executor) :
		ObjHeur{
			scip,
			name(callback::Type::Heuristic),
			"Primal heuristic that waits for another thread to provide a primal solution.",
			'e',
			priority,
			freq,
			freqofs,
			maxdepth,
			timingmask,
			false},
		m_weak_executor{std::move(weak_executor)} {}

	auto scip_exec(
		SCIP* scip,
		SCIP_HEUR* /*heur*/,
		SCIP_HEURTIMING heuristic_timing,
		SCIP_Bool node_infeasible,
		SCIP_RESULT* result) -> SCIP_RETCODE override {
		auto retcode = SCIP_OKAY;
		std::tie(retcode, *result) = handle_executor(
			scip, m_weak_executor, callback::HeuristicCall{heuristic_timing, static_cast<bool>(node_infeasible)});
		return retcode;
	}

private:
	std::weak_ptr<Executor> m_weak_executor;
};

template <>
auto include_reverse_callback<callback::Type::Heuristic>(
	SCIP* scip,
	std::weak_ptr<Executor> executor,
	callback::Constructor<callback::Type::Heuristic> args) -> void {
	scip::call(
		SCIPincludeObjHeur,
		scip,
		new ReverseHeur(
			scip,
			args.priority,
			args.frequency,
			args.frequency_offset,
			args.max_depth,
			args.timing_mask,
			std::move(executor)),
		true);
}  // NOLINT

class ReverseNodesel : public ::scip::ObjNodesel {
public:
	ReverseNodesel(
		SCIP* scip,
		int stdpriority, /**< priority of the node selector in standard mode */
		int memsavepriority,
		std::weak_ptr<Executor> weak_executor) :
		ObjNodesel{
			scip,
			name(callback::Type::Nodesel),
			"Nodesel that waits for another thread to pick the next open node.",
			stdpriority,
			memsavepriority},
		m_weak_executor{std::move(weak_executor)} {}

	/** node selection method of node selector
	 *
	 *  This method is called to select the next leaf of the branch and bound tree to be processed.
	 *
	 *  input:
	 *  - scip            : SCIP main data structure
	 *  - nodesel         : the node selector itself
	 *  - selnode         : pointer to store the selected node
	 *
	 *  possible return values for *selnode:
	 *  - NULL    : problem is solved, because tree is empty
	 *  - non-NULL: node to be solved next
	 */
	auto scip_select(SCIP* scip, SCIP_NODESEL* /* nodesel */, SCIP_NODE** selnode) -> SCIP_RETCODE override {
		auto retcode = SCIP_OKAY;
		auto result = SCIP_DIDNOTRUN;
		std::tie(retcode, result) = handle_executor(scip, m_weak_executor, callback::NodeselCall{selnode});
		return retcode;
	}

	/** node comparison method of node selector
	 *  XXX We use the default lowerbound prioritization.
	 *
	 *  This method is called to compare two nodes regarding their order in the node priority queue.
	 *
	 *  input:
	 *  - scip            : SCIP main data structure
	 *  - nodesel         : the node selector itself
	 *  - node1           : first node to compare
	 *  - node2           : second node to compare
	 *
	 *  possible return values:
	 *  - value < 0: node1 comes before (is better than) node2
	 *  - value = 0: both nodes are equally good
	 *  - value > 0: node2 comes after (is worse than) node2
	 */
	// https://github.com/scipopt/PySCIPOpt/blob/master/tests/test_nodesel.py
	auto scip_comp(SCIP* scip, SCIP_NODESEL* /* nodesel */, SCIP_NODE* node1, SCIP_NODE* node2) -> int override {
		// SCIP_Real lowerbound1 = SCIPnodeGetLowerbound(node1);
		// SCIP_Real lowerbound2 = SCIPnodeGetLowerbound(node2);

		// if (SCIPisLT(scip, lowerbound1, lowerbound2)) {
		// 	return -1;
		// }
		// if (SCIPisGT(scip, lowerbound1, lowerbound2)) {
		// 	return 1;
		// }
		return 0;  // both nodes are equally good
	}

private:
	std::weak_ptr<Executor> m_weak_executor;
};

template <>
auto include_reverse_callback<callback::Type::Nodesel>(
	SCIP* scip,
	std::weak_ptr<Executor> executor,
	callback::Constructor<callback::Type::Nodesel> args) -> void {
	scip::call(
		SCIPincludeObjNodesel,
		scip,
		new ReverseNodesel(scip, args.stdpriority, args.memsavepriority, std::move(executor)),
		true);
}  // NOLINT

}  // namespace

/****************************
 *  Definition of Scimpl  *
 ****************************/

void ScipDeleter::operator()(SCIP* ptr) {
	scip::call(SCIPfree, &ptr);
}

namespace {

std::unique_ptr<SCIP, ScipDeleter> create_scip() {
	SCIP* scip_raw;
	scip::call(SCIPcreate, &scip_raw);
	std::unique_ptr<SCIP, ScipDeleter> scip_ptr = nullptr;
	scip_ptr.reset(scip_raw);
	return scip_ptr;
}

}  // namespace

Scimpl::Scimpl() : m_scip{create_scip()} {}

Scimpl::Scimpl(Scimpl&&) noexcept = default;

Scimpl::Scimpl(std::unique_ptr<SCIP, ScipDeleter>&& scip_ptr) noexcept : m_scip(std::move(scip_ptr)) {}

Scimpl::~Scimpl() = default;

auto Scimpl::get_scip_ptr() noexcept -> SCIP* {
	return m_scip.get();
}

auto Scimpl::copy() const -> Scimpl {
	if (m_scip == nullptr) {
		return {nullptr};
	}
	if (SCIPgetStage(m_scip.get()) == SCIP_STAGE_INIT) {
		return {create_scip()};
	}
	auto dest = create_scip();
	// Copy operation is not thread safe
	static auto m = std::mutex{};
	auto g = std::lock_guard{m};
	scip::call(SCIPcopy, m_scip.get(), dest.get(), nullptr, nullptr, "", true, false, false, false, nullptr);
	return {std::move(dest)};
}

auto Scimpl::copy_orig() const -> Scimpl {
	if (m_scip == nullptr) {
		return {nullptr};
	}
	if (SCIPgetStage(m_scip.get()) == SCIP_STAGE_INIT) {
		return {create_scip()};
	}
	auto dest = create_scip();
	// Copy operation is not thread safe
	static auto m = std::mutex{};
	auto g = std::lock_guard{m};
	scip::call(SCIPcopyOrig, m_scip.get(), dest.get(), nullptr, nullptr, "", false, false, false, nullptr);
	return {std::move(dest)};
}

auto Scimpl::solve_iter(nonstd::span<callback::DynamicConstructor const> arg_packs)
	-> std::optional<callback::DynamicCall> {
	auto* const scip_ptr = get_scip_ptr();
	m_controller = std::make_unique<Controller>([=](std::weak_ptr<Executor> const& executor) {
		for (auto const pack : arg_packs) {
			std::visit([&](auto args) { include_reverse_callback(scip_ptr, executor, args); }, pack);
		}
		scip::call(SCIPsolve, scip_ptr);
	});
	return m_controller->wait();
}

auto Scimpl::solve_iter_continue(SCIP_RESULT result) -> std::optional<callback::DynamicCall> {
	m_controller->resume(result);
	return m_controller->wait();
}

}  // namespace ecole::scip
