#include <Cellerator/compute/tokenization/sequence_buffer.hh>

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/repeat_interleave.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace cellerator::tokenization {
#if defined(CELLERATOR_TOKENIZATION_HAS_CUDA)
void launch_sequence_append_cuda(
    const at::Tensor& selected_rows, const at::Tensor& update_offsets,
    const at::Tensor& update_ids, at::Tensor& lengths,
    const at::Tensor& starts, const at::Tensor& capacities,
    const at::Tensor& row_owner_scratch, at::Tensor& append_error_scratch,
    at::Tensor& succeeded, at::Tensor& token_storage,
    const std::unordered_map<std::string, at::Tensor>& update_attributes,
    std::unordered_map<std::string, at::Tensor>& attribute_storage);
#endif

namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

bool scalar_bool(const at::Tensor& value) { return value.item<bool>(); }

bool tensors_overlap(const at::Tensor& a, const at::Tensor& b) {
    if (a.storage().unsafeGetStorageImpl() != b.storage().unsafeGetStorageImpl()) return false;
    return at::get_overlap_status(a, b) != at::MemOverlapStatus::No;
}

void require_attribute_schema(
    const std::unordered_map<std::string, at::Tensor>& expected,
    const std::unordered_map<std::string, at::Tensor>& actual,
    const at::Device& device,
    std::int64_t occurrences) {
    require(expected.size() == actual.size(), "append attribute names must match buffer schema");
    for (const auto& [name, dst] : expected) {
        auto it = actual.find(name);
        require(it != actual.end(), "append attribute names must match buffer schema");
        const auto& src = it->second;
        require(src.defined() && dst.scalar_type() == src.scalar_type()
                && dst.device() == device && src.device() == device
                && src.dim() == dst.dim() && src.size(0) == occurrences,
                "append attribute dtype, rank, and device must match buffer schema");
        for (std::int64_t dim = 1; dim < dst.dim(); ++dim)
            require(src.size(dim) == dst.size(dim),
                    "append attribute trailing dimensions must match buffer schema");
    }
}

} // namespace

SequenceBuffer SequenceBuffer::create(
    const std::vector<std::int64_t>& capacities,
    const std::unordered_map<std::string, at::Tensor>& attribute_examples,
    at::Device device) {
    SequenceBuffer result;
    result.capacities_host_ = capacities;
    result.starts_host_.resize(capacities.size());
    std::int64_t total = 0;
    for (std::size_t i = 0; i < capacities.size(); ++i) {
        require(capacities[i] >= 0 && capacities[i] <= std::numeric_limits<std::int64_t>::max() - total,
                "sequence capacities must be nonnegative and fit int64 aggregate storage");
        result.starts_host_[i] = total;
        total += capacities[i];
    }
    auto host_long = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    result.starts_ = at::tensor(result.starts_host_, host_long).to(device);
    result.capacities_ = at::tensor(capacities, host_long).to(device);
    result.row_owner_scratch_ = at::full(
        {static_cast<std::int64_t>(capacities.size())}, -1, host_long).to(device);
    result.append_error_scratch_ = at::zeros({}, host_long).to(device);
    result.lengths_ = at::zeros({static_cast<std::int64_t>(capacities.size())}, host_long).to(device);
    result.token_storage_ = at::empty({total}, host_long.device(device));
    // `cuda` without an index is resolved by ATen to the current CUDA device.
    // Normalize the requested device to the allocated tensor's concrete device
    // before validating attribute examples or allocating attribute storage.
    device = result.token_storage_.device();
    for (const auto& [name, example] : attribute_examples) {
        require(!name.empty() && example.defined() && example.dim() >= 1
                && (example.size(0) == 0 || example.size(0) == 1),
                "attribute examples must have shape [0,...] or [1,...] and a nonempty name");
        require(!example.requires_grad(), "mutable sequence buffers cannot store gradient-tracked attributes");
        require(example.device() == device, "attribute examples must use the buffer device");
        std::vector<std::int64_t> shape{total};
        for (std::int64_t d = 1; d < example.dim(); ++d) shape.push_back(example.size(d));
        result.attribute_storage_.emplace(name, at::empty(shape, example.options()));
    }
    return result;
}

std::int64_t SequenceBuffer::sequence_count() const noexcept {
    return static_cast<std::int64_t>(capacities_host_.size());
}

at::Tensor SequenceBuffer::lengths() const { return lengths_; }
at::Tensor SequenceBuffer::token_storage() const { return token_storage_; }
const std::unordered_map<std::string, at::Tensor>&
SequenceBuffer::attribute_storage() const noexcept { return attribute_storage_; }

AppendResult SequenceBuffer::append_into(const at::Tensor& selected_rows,
                                         const TokenBatch& updates,
                                         std::optional<at::Tensor> status_buffer) {
    // Validate only metadata here: CUDA offsets and selected rows stay device
    // resident on the bounded append path.
    require(selected_rows.defined() && selected_rows.scalar_type() == at::kLong
            && selected_rows.dim() == 1 && selected_rows.device() == lengths_.device(),
            "selected rows must be device-aligned rank-1 int64");
    require(updates.token_ids.defined() && updates.token_ids.scalar_type() == at::kLong
            && updates.token_ids.dim() == 1 && updates.token_ids.device() == lengths_.device(),
            "update token IDs must be device-aligned rank-1 int64");
    require(updates.offsets.defined() && updates.offsets.scalar_type() == at::kLong
            && updates.offsets.dim() == 1 && updates.offsets.device() == lengths_.device()
            && updates.offsets.numel() == selected_rows.numel() + 1,
            "update offsets must define one row per selected sequence");
    require_attribute_schema(attribute_storage_, updates.attributes, lengths_.device(),
                             updates.token_ids.numel());
    for (const auto& [name, src] : updates.attributes) {
        (void)name;
        require(!src.requires_grad(), "mutable sequence append rejects gradient-tracked attributes");
    }
    at::Tensor succeeded;
    if (status_buffer) {
        succeeded = *status_buffer;
        require(succeeded.defined() && succeeded.scalar_type() == at::kBool
                && succeeded.dim() == 1 && succeeded.numel() == selected_rows.numel()
                && succeeded.device() == lengths_.device() && succeeded.is_contiguous()
                && !succeeded.requires_grad(),
                "status_buffer must be contiguous bool[submitted_rows] on the buffer device");
    } else {
        succeeded = at::empty({selected_rows.numel()},
            at::TensorOptions().dtype(at::kBool).device(lengths_.device()));
    }
    std::vector<at::Tensor> append_reads{selected_rows, updates.offsets, updates.token_ids};
    for (const auto& [name, value] : updates.attributes) {
        (void)name;
        append_reads.push_back(value);
    }
    std::vector<at::Tensor> mutable_storage{lengths_, token_storage_};
    for (const auto& [name, value] : attribute_storage_) {
        (void)name;
        mutable_storage.push_back(value);
    }
#if defined(CELLERATOR_TOKENIZATION_HAS_CUDA)
    if (lengths_.device().is_cuda()) {
        mutable_storage.push_back(row_owner_scratch_);
        mutable_storage.push_back(append_error_scratch_);
    }
#endif
    for (const auto& read : append_reads) {
        for (const auto& write : mutable_storage) {
            require(!tensors_overlap(read, write),
                    "append input tensors must not overlap mutable buffer storage or status");
        }
    }
    if (status_buffer) {
        for (const auto& read : append_reads)
            require(!tensors_overlap(succeeded, read),
                    "append status must not overlap input payloads");
        for (const auto& write : mutable_storage)
            require(!tensors_overlap(succeeded, write),
                    "append status must not overlap mutable buffer storage");
    }
    if (lengths_.device().is_cuda()) {
#if defined(CELLERATOR_TOKENIZATION_HAS_CUDA)
        row_owner_scratch_.fill_(-1);
        append_error_scratch_.zero_();
        succeeded.fill_(true);
        launch_sequence_append_cuda(selected_rows, updates.offsets, updates.token_ids,
            lengths_, starts_, capacities_, row_owner_scratch_, append_error_scratch_,
            succeeded, token_storage_,
            updates.attributes, attribute_storage_);
        return {succeeded, lengths_};
#else
        throw std::runtime_error("CUDA sequence append is unavailable in this build");
#endif
    }

    auto rows = selected_rows.contiguous();
    auto offsets = updates.offsets.contiguous();
    auto source_ids = updates.token_ids.contiguous();
    auto row_values = rows.accessor<std::int64_t, 1>();
    const auto* offset_values = offsets.data_ptr<std::int64_t>();
    const auto* source_id_values = source_ids.data_ptr<std::int64_t>();
    require(offset_values[0] == 0 && offset_values[rows.numel()] == source_ids.numel(),
            "append offsets must begin at zero and end at the token count");
    for (std::int64_t i = 0; i < rows.numel(); ++i)
        require(offset_values[i] <= offset_values[i + 1],
                "append offsets must be nondecreasing");
    for (std::int64_t i = 0; i < source_ids.numel(); ++i)
        require(source_id_values[i] >= 0, "token IDs appended to a sequence must be nonnegative");

    auto len_values = lengths_.accessor<std::int64_t, 1>();
    std::set<std::int64_t> seen;
    for (std::int64_t i = 0; i < rows.numel(); ++i) {
        const auto row = row_values[i];
        require(row >= 0 && row < sequence_count(), "selected sequence row is out of range");
        require(seen.insert(row).second, "selected sequence rows must be unique");
        require(len_values[row] >= 0 && len_values[row] <= capacities_host_[row],
                "sequence length must be within its capacity");
    }
    auto* ok_values = succeeded.data_ptr<bool>();
    std::fill_n(ok_values, succeeded.numel(), false);
    struct AttributeCopy {
        at::Tensor tensor;
        std::size_t bytes_per_token;
        const std::uint8_t* source_bytes;
        std::uint8_t* destination_bytes;
    };
    std::vector<AttributeCopy> attribute_copies;
    attribute_copies.reserve(updates.attributes.size());
    for (const auto& [name, value] : updates.attributes) {
        auto contiguous = value.contiguous();
        std::size_t bytes_per_token = contiguous.element_size();
        for (std::int64_t dim = 1; dim < contiguous.dim(); ++dim)
            bytes_per_token *= static_cast<std::size_t>(contiguous.size(dim));
        auto* destination_bytes = reinterpret_cast<std::uint8_t*>(
            attribute_storage_.at(name).data_ptr());
        auto* source_bytes = reinterpret_cast<const std::uint8_t*>(contiguous.data_ptr());
        attribute_copies.push_back(AttributeCopy{
            std::move(contiguous), bytes_per_token, source_bytes, destination_bytes});
    }
    auto* destination_ids = token_storage_.data_ptr<std::int64_t>();
    for (std::int64_t i = 0; i < rows.numel(); ++i) {
        const auto row = row_values[i];
        const auto begin = offset_values[i];
        const auto count = offset_values[i + 1] - begin;
        if (count > capacities_host_[row] - len_values[row]) continue;
        const auto destination = starts_host_[row] + len_values[row];
        if (count > 0) {
            std::memcpy(destination_ids + destination, source_id_values + begin,
                        static_cast<std::size_t>(count) * sizeof(std::int64_t));
            for (const auto& attribute : attribute_copies) {
                if (attribute.bytes_per_token == 0) continue;
                std::memcpy(attribute.destination_bytes
                                + static_cast<std::size_t>(destination) * attribute.bytes_per_token,
                            attribute.source_bytes
                                + static_cast<std::size_t>(begin) * attribute.bytes_per_token,
                            static_cast<std::size_t>(count) * attribute.bytes_per_token);
            }
            len_values[row] += count;
        }
        ok_values[i] = true;
    }
    return {succeeded, lengths_};
}

void SequenceBuffer::append_checked(const at::Tensor& selected_rows,
                                    const TokenBatch& updates) {
    auto result = append_into(selected_rows, updates);
    require(result.succeeded.numel() == 0 || result.succeeded.all().item<bool>(),
            "sequence append exceeded capacity for one or more rows");
}

SequenceBuffer SequenceBuffer::reserve(
    const std::vector<std::int64_t>& new_capacities) const {
    require(new_capacities.size() == capacities_host_.size(),
            "reserve must provide one capacity per existing sequence");
    std::unordered_map<std::string, at::Tensor> examples;
    for (const auto& [name, storage] : attribute_storage_) {
        std::vector<std::int64_t> shape{1};
        for (std::int64_t d = 1; d < storage.dim(); ++d) shape.push_back(storage.size(d));
        examples.emplace(name, at::empty(shape, storage.options()));
    }
    auto result = SequenceBuffer::create(new_capacities, examples, lengths_.device());
    auto old = compact();
    auto selected = at::arange(sequence_count(),
        at::TensorOptions().dtype(at::kLong).device(lengths_.device()));
    auto append_result = result.append_into(selected, old);
    require(append_result.succeeded.numel() == 0 || append_result.succeeded.all().item<bool>(),
            "new capacities must not be smaller than current sequence lengths");
    return result;
}

TokenBatch SequenceBuffer::compact() const {
    require(scalar_bool(((lengths_ >= 0) & (lengths_ <= capacities_)).all()),
            "sequence lengths must be within their capacities before compaction");
    auto total = lengths_.sum().item<std::int64_t>();
    auto rows = at::repeat_interleave(at::arange(sequence_count(), lengths_.options()), lengths_,
                                      0, total);
    auto offsets = at::cat({at::zeros({1}, lengths_.options()), at::cumsum(lengths_, 0)});
    auto prior = offsets.slice(0, 0, sequence_count());
    auto within = at::arange(total, lengths_.options()) - prior.index_select(0, rows);
    auto positions = starts_.index_select(0, rows) + within;
    TokenBatch result{token_storage_.index_select(0, positions), offsets, {}};
    for (const auto& [name, storage] : attribute_storage_)
        result.attributes.emplace(name, storage.index_select(0, positions));
    return result;
}

AppendResult event_stream_into(SequenceBuffer& output, const at::Tensor& selected_rows,
                               const at::Tensor& identity_ids, const at::Tensor& offsets,
                               const Vocabulary& vocabulary,
                               const std::unordered_map<std::string, at::Tensor>& attributes,
                               std::optional<at::Tensor> status_buffer) {
    return output.append_into(selected_rows,
        event_stream(identity_ids, offsets, vocabulary, attributes), status_buffer);
}

AppendResult ranked_feature_into(SequenceBuffer& output, const at::Tensor& selected_rows,
                                 const at::Tensor& values, const at::Tensor& feature_ids,
                                 const Vocabulary& vocabulary,
                                 const RankedFeatureOptions& options,
                                 const std::unordered_map<std::string, at::Tensor>& attributes,
                                 std::optional<at::Tensor> status_buffer) {
    return output.append_into(selected_rows,
        ranked_feature(values, feature_ids, vocabulary, options, attributes), status_buffer);
}

} // namespace cellerator::tokenization
