#pragma once

#include <optional>

#include "ecole/export.hpp"
#include "ecole/observation/abstract.hpp"

namespace ecole::observation {

struct ECOLE_EXPORT FocusNodeObs {
	long long number;
	int depth;
	double lowerbound;
	double estimate;
	int n_added_conss;
	int n_vars;
	int nlpcands;
	int npseudocands;
	long long parent_number;
	double parent_lowerbound;
};

class ECOLE_EXPORT FocusNode {
public:
	auto before_reset(scip::Model& /*model*/) -> void {}

	ECOLE_EXPORT auto extract(scip::Model& model, bool done) -> std::optional<FocusNodeObs>;
};

}  // namespace ecole::observation
