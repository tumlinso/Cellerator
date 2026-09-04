#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
namespace cellerator::compiler::reflection::v1 {
handle_status_v1 validate_handle_v1(const ir_handle_v1&h,const handle_context_v1&c,std::uint64_t g)noexcept{if(!(h.identity_high||h.identity_low)||!h.arena_epoch||!h.object_generation)return handle_status_v1::invalid_identity;if(static_cast<unsigned>(c.phase)<static_cast<unsigned>(h.available_at))return handle_status_v1::unavailable;if(h.lifetime==handle_lifetime_v1::expression&&c.phase!=h.available_at)return handle_status_v1::expired;if(h.arena_epoch!=c.arena_epoch||h.object_generation!=g)return handle_status_v1::stale;return handle_status_v1::valid;}
ir_handle_v1 preserve_handle_for_safe_transform_v1(const ir_handle_v1&h)noexcept{return h;}
ir_handle_v1 invalidate_handle_for_edit_v1(const ir_handle_v1&h)noexcept{auto r=h;++r.object_generation;return r;}
}
