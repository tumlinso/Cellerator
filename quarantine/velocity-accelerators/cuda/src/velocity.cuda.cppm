module;
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

export module velocity.cuda;
export import velocity;
export import velocity.precision;
export import anndata.cuda;

export namespace velocity::cuda {
using anndata::cuda::DenseView;
using anndata::cuda::SparseView;
using anndata::cuda::Stream;

struct MatrixView {
    bool sparse = false;
    DenseView dense{};
    SparseView compressed{};
    MatrixView() = default;
    MatrixView(DenseView x) : dense(x) {}
    MatrixView(SparseView x) : sparse(true), compressed(x) {}
};

// A block source fills/reuses caller-owned device buffers in selected-gene
// order. It enqueues transfers on stream and keeps its buffers alive through
// their consumers on that stream. The returned matrices are dense N by count;
// no full input allocation or hidden scheduler is introduced.
struct SourceBlock {
    DenseView spliced{}, unspliced{};
    // Optional borrowed host bytes for requested S/U pairs, valid until the
    // enclosing stage method returns. A zero reports an unavailable original;
    // nullptr reports every requested pair available.
    const std::uint8_t* available = nullptr;
};
struct BlockSource {
    void* context = nullptr;
    // Host source_genes is valid during this callback only. Copy it into owned
    // staging before any asynchronous use. Returned device buffers remain alive
    // through their consumers on stream.
    void (*acquire)(void*, const std::uint64_t* source_genes, std::size_t count,
                    Stream, SourceBlock&) = nullptr;
    // Total caller-owned numerical materialization/packing storage in host
    // and device memory. Borrowed resident inputs require zero staging. This
    // declaration is charged together with internal scratch to workspace_bytes.
    std::size_t staging_bytes = 0;
};
struct Input {
    MatrixView weights, spliced, unspliced;
    anndata::Axis genes = anndata::Axis::all();
    std::optional<DenseView> fit_mask, gene_mask;
    // For a block source, source_genes declares the logical full gene extent;
    // spliced/unspliced resident views are unused. Cell extent comes from W.
    BlockSource source;
    std::size_t source_genes = 0;
    // Required dtype declaration for a block provider's automatic precision.
    anndata::cuda::ScalarType source_spliced_dtype = anndata::cuda::ScalarType::float32;
    anndata::cuda::ScalarType source_unspliced_dtype = anndata::cuda::ScalarType::float32;
};

namespace detail { struct PreparedState; struct ExpressionState; struct WorkspaceState; struct ResultState; struct StageState; struct Access; }
class PreparedInput {
public:
    // Acquisition completes preparation on the supplied stream; subsequent
    // numerical calls may use another explicitly ordered stream on the device.
    PreparedInput(Input, Stream, anndata::Validation = anndata::Validation::full);
    ~PreparedInput();
    PreparedInput(PreparedInput&&) noexcept;
    PreparedInput& operator=(PreparedInput&&) noexcept;
    PreparedInput(const PreparedInput&) = delete;
    PreparedInput& operator=(const PreparedInput&) = delete;
    std::size_t n_cells() const noexcept;
    std::size_t n_genes() const noexcept;
    std::size_t source_genes() const noexcept;
    std::size_t gene_index(std::size_t) const;
    std::size_t storage_bytes() const noexcept;
    int device() const noexcept;
    // Explicit acquisition-boundary value check, including stream completion.
    // Numerical values may change between runs; structural mutation requires
    // re-preparation. All input owners remain caller-owned.
    void validate_values(Stream) const;
private:
    std::unique_ptr<detail::PreparedState> state_;
    friend struct detail::Access;
};

class Workspace {
public:
    Workspace(const PreparedInput&, RunOptions = {});
    Workspace(const PreparedInput&, StochasticRunOptions);
    ~Workspace();
    Workspace(Workspace&&) noexcept;
    Workspace& operator=(Workspace&&) noexcept;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    std::size_t block_genes() const noexcept;
    std::size_t scratch_bytes() const noexcept;
    std::size_t staging_bytes() const noexcept;
    std::size_t budget_bytes() const noexcept;
    ComputePrecision compute_precision() const noexcept;
private:
    std::unique_ptr<detail::WorkspaceState> state_;
    friend struct detail::Access;
};

// Alternate S/U preparation reuses the base graph and axes. A missing source
// is deliberate: requested promotions then retain the base FP16 values. Keep
// this object, its owners/provider, and staging alive and unmoved through both
// stochastic passes and stream completion. Original values stay fixed per run.
class ExpressionSource {
public:
    ExpressionSource(const PreparedInput&, MatrixView spliced, MatrixView unspliced,
                     Stream, anndata::Validation = anndata::Validation::full);
    ExpressionSource(const PreparedInput&, BlockSource = {});
    ~ExpressionSource();
    ExpressionSource(ExpressionSource&&) noexcept;
    ExpressionSource& operator=(ExpressionSource&&) noexcept;
    ExpressionSource(const ExpressionSource&) = delete;
    ExpressionSource& operator=(const ExpressionSource&) = delete;
    std::size_t storage_bytes() const noexcept;
private:
    std::unique_ptr<detail::ExpressionState> state_;
    friend struct detail::Access;
};

// All numerical diagnostics have the chosen compute dtype. Selection/status
// fields and global flags are uint8 device arrays. Shape is genes by 1, flags
// are 1 by 1. FitResult owns storage; ResultView also permits foreign owners.
struct ResultView {
    DenseView initial_gamma{}, initial_offset{}, r2{}, selected{}, initial_status{};
    DenseView gamma{}, offset{}, offset2{}, first_scale{}, second_scale{}, objective{}, status{};
    DenseView low_signal{}, refit_all{};
};
class FitResult {
public:
    FitResult(const PreparedInput&, ComputePrecision, bool stochastic = false);
    ~FitResult();
    FitResult(FitResult&&) noexcept;
    FitResult& operator=(FitResult&&) noexcept;
    FitResult(const FitResult&) = delete;
    FitResult& operator=(const FitResult&) = delete;
    ResultView view() const noexcept;
    std::size_t storage_bytes() const noexcept;
private:
    std::unique_ptr<detail::ResultState> state_;
};

// Moment fields borrow workspace and use column-major N by count views.
// Empty output views mean unrequested fields. Caller-owned outputs may use
// any validated nonoverlapping signed strides and FP16/FP32/FP64 storage.
struct MomentBlock {
    std::size_t first = 0, count = 0;
    DenseView spliced{}, unspliced{}, spliced_second{}, cross{}, unspliced_second{};
};
struct FitBlock { std::size_t first = 0, count = 0; ResultView result; };
struct Outputs {
    DenseView velocity{}, second_residual{}, spliced{}, unspliced{};
    DenseView spliced_second{}, cross{}, unspliced_second{};
    // Reusable N by workspace.block_genes() output buffers. emit writes local
    // columns, then calls the optional sink before submitting the next block.
    bool block_local = false;
};
struct OutputBlock {std::size_t first=0,count=0;Outputs outputs;};
struct BlockSink {
    void* context=nullptr;
    // Enqueue consumption on stream or establish explicit event ordering.
    // Borrowed output buffers may be overwritten by the next block on stream.
    void (*write)(void*,const OutputBlock&,Stream)=nullptr;
};
struct SensitivityView {
    DenseView state{}, low{}, high{}, delta{}, prediction_change{}, r2_change{};
    DenseView expanded_status{}, contracted_status{}, status_change{}, selection_change{};
};

class ReferenceStages {
public:
    ReferenceStages(const PreparedInput&, Workspace&, ResultView, FitOptions, Outputs, Stream, BlockSink = {});
    ReferenceStages(const PreparedInput&, Workspace&, ResultView, StochasticOptions, Outputs, Stream, BlockSink = {});
    ~ReferenceStages();
    ReferenceStages(ReferenceStages&&) noexcept;
    ReferenceStages& operator=(ReferenceStages&&) noexcept;
    ReferenceStages(const ReferenceStages&) = delete;
    ReferenceStages& operator=(const ReferenceStages&) = delete;
    MomentBlock initial_moments(std::size_t first, std::size_t count);
    // Mask is borrowed host uint8 over the original source-gene axis. nullptr
    // requests no promotion. Optional host block statuses: 0 not requested,
    // 1 original applied, 2 unavailable (including originals still in FP16).
    // Stochastic second pass reuses the initialization choices exactly.
    MomentBlock initial_moments_from(const ExpressionSource&, std::size_t first, std::size_t count,
                                    const std::uint8_t* source_gene_mask = nullptr,
                                    std::uint8_t* block_promotion_status = nullptr);
    FitBlock initial_fit(MomentBlock);
    void accept_initialization(MomentBlock, FitBlock);
    void resolve_selection();
    MomentBlock moments(std::size_t first, std::size_t count);
    MomentBlock moments_from(const ExpressionSource&, std::size_t first, std::size_t count,
                            const std::uint8_t* source_gene_mask = nullptr,
                            std::uint8_t* block_promotion_status = nullptr);
    void sensitivity(MomentBlock, FitBlock, SensitivityView,
                     velocity::precision::SensitivityOptions = {});
    FitBlock scales(MomentBlock);
    FitBlock fit(MomentBlock);
    FitBlock fit(MomentBlock, FitBlock scales);
    void emit(MomentBlock, FitBlock);
    ResultView finish();
private:
    std::unique_ptr<detail::StageState> state_;
};

namespace reference {
// Completion is explicit: these calls enqueue into stream. Inputs, output
// owners, result, and workspace remain alive until completion. They never
// synchronize to run fitting on the CPU or widen compute precision.
FitResult steady_state(const PreparedInput&, Workspace&, FitOptions, Outputs, Stream, BlockSink = {});
FitResult stochastic(const PreparedInput&, Workspace&, StochasticOptions, Outputs, Stream, BlockSink = {});
void steady_state_into(const PreparedInput&, Workspace&, ResultView, FitOptions, Outputs, Stream, BlockSink = {});
void stochastic_into(const PreparedInput&, Workspace&, ResultView, StochasticOptions, Outputs, Stream, BlockSink = {});
// Explicit scVelo neighborhood helper. Produces dense FP32/FP64 support/self/
// row-normalized weights; duplicate sums use that requested precision. The
// normal moments path applies W literally and performs no normalization. This
// acquisition operation completes stream before releasing its temporary index.
void prepare_neighborhood(MatrixView connectivities, DenseView output, Stream);
}
}
