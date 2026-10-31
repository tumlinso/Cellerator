#include <Cellerator/nn/readouts.hh>

#include <stdexcept>

namespace cellerator::nn {

void validate_feature_readout_context(const at::Tensor& context,
                                      const at::Tensor& target_embedding,
                                      std::int64_t width) {
    if (!context.defined() || (context.dim() != 2 && context.dim() != 3)) {
        throw std::invalid_argument(
            "FeatureReadout context must have shape [batch,width] or [batch,features,width]");
    }
    if (width <= 0 || context.size(-1) != width) {
        throw std::invalid_argument("FeatureReadout context width does not match its declaration");
    }
    if (!context.is_floating_point()) {
        throw std::invalid_argument("FeatureReadout context must use a floating-point dtype");
    }
    if (!target_embedding.defined() || context.device() != target_embedding.device()) {
        throw std::invalid_argument(
            "FeatureReadout context and target embeddings must share a device");
    }
    if (context.scalar_type() != target_embedding.scalar_type()) {
        throw std::invalid_argument(
            "FeatureReadout context and target embeddings must share a dtype");
    }
}

} // namespace cellerator::nn
