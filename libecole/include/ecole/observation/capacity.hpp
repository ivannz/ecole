#pragma once

#include <optional>
#include <xtensor/xtensor.hpp>

#include "ecole/export.hpp"
#include "ecole/observation/abstract.hpp"

namespace ecole::observation {

using CapacityObs = xt::xtensor<double, 1>;

class ECOLE_EXPORT Capacity {
public:
	auto before_reset(scip::Model& /*model*/) -> void {}

	ECOLE_EXPORT auto extract(scip::Model& model, bool done) -> std::optional<CapacityObs>;
};

}  // namespace ecole::observation
