#include <Cellerator/compiler/frontend/source/define_pragma_interaction_with_preprocessing_conditional_v1.hh>

namespace Cellerator::compiler::frontend::source {

static_assert(pragma_may_activate_v1({true, false, preprocessing_source_v1::textual, true}));
static_assert(!pragma_may_activate_v1({false, true, preprocessing_source_v1::module, true}));

} // namespace Cellerator::compiler::frontend::source
