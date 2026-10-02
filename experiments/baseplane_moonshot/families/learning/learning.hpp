#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ce_moon::learning {
// CPU double precision; caller owns rows, labels and d+1 model coefficients.
// The final coefficient is bias. Fitting initializes a fresh model at zero.
struct FitOptions {
  std::size_t epochs = 600;
  double learning_rate = 0.25;
  double l2 = 0.001;
  double budget_weight = 0.0;
  double positive_budget = 0.5;
};
struct FitReport {
  double initial_loss, final_loss, positive_rate;
  std::size_t epochs;
};
inline void finite(double x) {
  if (!std::isfinite(x)) throw std::invalid_argument("nonfinite numerical input/result");
}
inline double sigmoid(double z) {
  finite(z);
  return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
}
inline double predict_logistic(const double* row, std::size_t d, const double* weights) {
  if (!row || !weights || !d) throw std::invalid_argument("empty logistic model");
  double z = weights[d]; finite(z);
  for (std::size_t j = 0; j < d; ++j) {
    finite(row[j]); finite(weights[j]); z += row[j] * weights[j];
  }
  return sigmoid(z);
}
inline void validate_options(const FitOptions& o) {
  finite(o.learning_rate); finite(o.l2); finite(o.budget_weight); finite(o.positive_budget);
  if (!o.epochs || o.learning_rate <= 0 || o.l2 < 0 || o.budget_weight < 0 ||
      o.positive_budget < 0 || o.positive_budget > 1)
    throw std::invalid_argument("invalid fit options");
}
// Objective and full gradient: mean BCE + .5*l2*||w||^2 (bias excluded)
// + budget_weight*max(0, mean(sigmoid(z))-positive_budget)^2.
// Internal scratch is O(n+d); no stream/device allocation or sequence interpretation.
inline double logistic_objective(const double* rows, const double* targets,
                                 std::size_t n, std::size_t d, const double* weights,
                                 FitOptions o = {}, double* gradient = nullptr,
                                 double* positive_rate = nullptr) {
  validate_options(o);
  if (!rows || !targets || !weights || !n || !d ||
      d == std::numeric_limits<std::size_t>::max() ||
      n > std::numeric_limits<std::size_t>::max() / d)
    throw std::invalid_argument("invalid logistic shape");
  std::vector<double> probs(n);
  double loss = 0, mean = 0;
  if (gradient) std::fill(gradient, gradient + d + 1, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    finite(targets[i]);
    if (targets[i] < 0 || targets[i] > 1) throw std::invalid_argument("target outside [0,1]");
    double z = weights[d]; finite(z);
    for (std::size_t j = 0; j < d; ++j) {
      finite(rows[i*d+j]); finite(weights[j]); z += rows[i*d+j] * weights[j];
    }
    probs[i] = sigmoid(z); mean += probs[i] / double(n);
    loss += (std::max(z, 0.0) - targets[i]*z + std::log1p(std::exp(-std::abs(z)))) / double(n);
  }
  const double excess = std::max(0.0, mean - o.positive_budget);
  loss += o.budget_weight * excess * excess;
  for (std::size_t j = 0; j < d; ++j) {
    loss += 0.5 * o.l2 * weights[j] * weights[j];
    if (gradient) gradient[j] = o.l2 * weights[j];
  }
  if (gradient) {
    for (std::size_t i = 0; i < n; ++i) {
      const double residual = (probs[i] - targets[i] + 2*o.budget_weight*excess*probs[i]*(1-probs[i])) / double(n);
      for (std::size_t j = 0; j < d; ++j) gradient[j] += residual * rows[i*d+j];
      gradient[d] += residual;
    }
    for (std::size_t j = 0; j <= d; ++j) finite(gradient[j]);
  }
  finite(loss);
  if (positive_rate) *positive_rate = mean;
  return loss;
}
inline FitReport fit_logistic(const double* rows, const double* targets,
                             std::size_t n, std::size_t d, double* weights,
                             FitOptions o = {}) {
  if (!weights || !d || d == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("invalid output model");
  std::vector<double> model(d+1, 0), gradient(d+1);
  FitReport report{logistic_objective(rows, targets, n, d, model.data(), o), 0, 0, o.epochs};
  for (std::size_t epoch = 0; epoch < o.epochs; ++epoch) {
    logistic_objective(rows, targets, n, d, model.data(), o, gradient.data());
    for (std::size_t j = 0; j <= d; ++j) { model[j] -= o.learning_rate * gradient[j]; finite(model[j]); }
  }
  report.final_loss = logistic_objective(rows, targets, n, d, model.data(), o, nullptr, &report.positive_rate);
  std::copy(model.begin(), model.end(), weights);
  return report;
}

// E29: independent relaxed Bernoulli gates over LUT3 rows. Bit index = 4*a+2*b+c.
struct TruthTeacher {
  std::array<double,8> logits{};
  std::uint64_t weight_version = 0;
  double predict(unsigned row) const {
    if (row >= 8) throw std::invalid_argument("LUT row");
    return sigmoid(logits[row]);
  }
};
inline TruthTeacher fit_truth_table(const std::array<double,8>& targets,
                                   std::size_t epochs = 200, double rate = 0.5,
                                   std::uint64_t version = 1) {
  finite(rate);
  if (!epochs || rate <= 0) throw std::invalid_argument("invalid teacher fit");
  TruthTeacher teacher; teacher.weight_version = version;
  for (double y : targets) { finite(y); if (y < 0 || y > 1) throw std::invalid_argument("teacher target"); }
  for (std::size_t e = 0; e < epochs; ++e)
    for (unsigned i = 0; i < 8; ++i) { teacher.logits[i] -= rate*(teacher.predict(i)-targets[i]); finite(teacher.logits[i]); }
  return teacher;
}
struct HardLut {
  std::uint8_t immediate = 0;
  std::uint64_t teacher_version = 0, circuit_version = 0;
  bool predict(unsigned row) const {
    if (row >= 8) throw std::invalid_argument("LUT row");
    return (immediate >> row) & 1u;
  }
};
inline HardLut harden(const TruthTeacher& teacher, std::uint64_t circuit_version = 1) {
  HardLut lut{0, teacher.weight_version, circuit_version};
  for (unsigned i = 0; i < 8; ++i) if (teacher.predict(i) >= 0.5) lut.immediate |= std::uint8_t(1u<<i);
  return lut;
}
inline std::string emit_lop3(const HardLut& lut) {
  // Native instruction source, deliberately no runtime compilation or GPU launch.
  return "lop3.b32 %0, %1, %2, %3, " + std::to_string(unsigned(lut.immediate)) + ";";
}

// E30: one learned linear score per caller-defined predicate bank/state row.
// Scores/selection are hypotheses; the caller owns each bank's predicate meaning.
struct BankSelection { std::size_t index; double score; std::uint64_t state_version, weight_version; };
inline BankSelection select_bank(const double* state, std::size_t d, const double* bank_weights,
                                 std::size_t banks, std::uint64_t state_version,
                                 std::uint64_t weight_version) {
  if (!state || !bank_weights || !d || !banks || d == std::numeric_limits<std::size_t>::max() ||
      banks > std::numeric_limits<std::size_t>::max()/(d+1)) throw std::invalid_argument("bank shape");
  BankSelection best{0,-1,state_version,weight_version};
  for (std::size_t b=0; b<banks; ++b) {
    double score=predict_logistic(state,d,bank_weights+b*(d+1));
    if (score>best.score) { best.index=b; best.score=score; }
  }
  return best;
}

// E31: bounded equality saturation of exact 3-input Boolean expressions. A domain
// tag carries the caller's coordinate/type identity and must match at every join.
enum class MaskOp { input, constant, negate, conjunction, disjunction, exclusive_or };
struct MaskExpr;
using Mask = std::shared_ptr<const MaskExpr>;
struct MaskExpr { MaskOp op; std::uint64_t domain; unsigned value; Mask left, right; };
inline Mask input(unsigned i, std::uint64_t domain) {
  if (i>=3) throw std::invalid_argument("three-input Boolean space");
  return std::make_shared<MaskExpr>(MaskExpr{MaskOp::input,domain,i,{},{}});
}
inline Mask constant(bool b, std::uint64_t domain) {
  return std::make_shared<MaskExpr>(MaskExpr{MaskOp::constant,domain,unsigned(b),{},{}});
}
inline Mask unary(Mask a) {
  if (!a) throw std::invalid_argument("null mask");
  return std::make_shared<MaskExpr>(MaskExpr{MaskOp::negate,a->domain,0,a,{}});
}
inline Mask binary(MaskOp op, Mask a, Mask b) {
  if (!a || !b || a->domain!=b->domain ||
      (op!=MaskOp::conjunction && op!=MaskOp::disjunction && op!=MaskOp::exclusive_or))
    throw std::invalid_argument("Boolean type/domain mismatch");
  return std::make_shared<MaskExpr>(MaskExpr{op,a->domain,0,a,b});
}
inline bool evaluate(const Mask& e, unsigned row) {
  if (!e || row>=8) throw std::invalid_argument("mask/row");
  switch(e->op) {
    case MaskOp::input: return (row>>(2-e->value))&1u;
    case MaskOp::constant: return e->value;
    case MaskOp::negate: return !evaluate(e->left,row);
    case MaskOp::conjunction: return evaluate(e->left,row) && evaluate(e->right,row);
    case MaskOp::disjunction: return evaluate(e->left,row) || evaluate(e->right,row);
    case MaskOp::exclusive_or: return evaluate(e->left,row) != evaluate(e->right,row);
  }
  throw std::invalid_argument("mask opcode");
}
inline unsigned truth_table(const Mask& e) {
  unsigned lut=0;
  for(unsigned i=0;i<8;++i) lut |= unsigned(evaluate(e,i))<<i;
  return lut;
}
inline std::size_t instruction_cost(const Mask& e) {
  if (!e) throw std::invalid_argument("null mask");
  if (e->op==MaskOp::input || e->op==MaskOp::constant) return 0;
  return 1+instruction_cost(e->left)+(e->right?instruction_cost(e->right):0);
}
struct RewriteResult { Mask expression; std::size_t before, after, rounds; };
inline RewriteResult minimize_boolean(const Mask& e, std::size_t rounds=3) {
  if (!e || rounds>4) throw std::invalid_argument("bounded rewrite rounds [0,4]");
  std::array<Mask,256> representatives{};
  auto insert=[&](Mask candidate) {
    unsigned lut=truth_table(candidate);
    if (!representatives[lut] || instruction_cost(candidate)<instruction_cost(representatives[lut]))
      representatives[lut]=candidate;
  };
  insert(e); insert(constant(false,e->domain)); insert(constant(true,e->domain));
  for(unsigned i=0;i<3;++i) { auto a=input(i,e->domain); insert(a); insert(unary(a)); }
  for(std::size_t r=0;r<rounds;++r) {
    auto previous=representatives;
    for(const auto& a:previous) if(a) {
      insert(unary(a));
      for(const auto& b:previous) if(b)
        for(auto op:{MaskOp::conjunction,MaskOp::disjunction,MaskOp::exclusive_or}) insert(binary(op,a,b));
    }
  }
  const auto out=representatives[truth_table(e)];
  return {out,instruction_cost(e),instruction_cost(out),rounds};
}
enum class NumericalMode { exact_boolean, strict_float, approximate_float };
inline bool reassociation_allowed(NumericalMode mode) { return mode==NumericalMode::approximate_float; }

// E32: immutable prepared circuit plus explicit versions; all guard failures use
// current unspecialized teacher, including a changed query/state/domain tag.
struct Guard {
  std::uint64_t weight_version=0, query_version=0, state_version=0, domain_version=0;
  bool operator==(const Guard& o) const {
    return weight_version==o.weight_version && query_version==o.query_version &&
           state_version==o.state_version && domain_version==o.domain_version;
  }
};
struct Specialization { HardLut lut; Guard guard; };
struct GuardedDecision { bool decision; bool specialized; };
inline Specialization specialize(const TruthTeacher& teacher, Guard guard,
                                 std::uint64_t circuit_version=1) {
  if(guard.weight_version!=teacher.weight_version) throw std::invalid_argument("teacher guard mismatch");
  return {harden(teacher,circuit_version),guard};
}
inline GuardedDecision guarded_predict(const Specialization& cached, const TruthTeacher& current,
                                       Guard context, unsigned row) {
  if(context.weight_version!=current.weight_version) throw std::invalid_argument("current teacher/context mismatch");
  if(cached.guard==context && cached.lut.teacher_version==current.weight_version)
    return {cached.lut.predict(row),true};
  return {current.predict(row)>=0.5,false};
}
} // namespace ce_moon::learning
