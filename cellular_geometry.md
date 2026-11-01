# Cellular geometry

Cellerator aims to exploit restrictions on how a modeled cellular system can
depend on itself. These restrictions concern both which interactions are
possible and which limited parts of state determine their conditions.
Expressing that organization allows learning and execution to work within
admissible dependencies, avoiding unnecessary candidate construction, context
evaluation, computation, and communication.

Cellerator aims to make this shortcut as model-agnostic as possible. Core
structural knowledge about the system should be reusable across compatible
model architectures, learning objectives, and downstream questions. Asking a
foundation model a different question should not require it to relearn that
same core structure from observations. What remains unknown or specific to the
question still needs to be learned; established dependencies and their
conditioning requirements should be available as reusable foundations.

The opportunity begins before choosing active interactions or a sparse storage
format. Some dependencies can be ruled out throughout a declared biological
regime. Others can participate only under conditions determined by a restricted
part of the cell's state. Where those restrictions are available, Cellerator
should enable a model to express them and organize its operations around them.
It should not need to construct every possible feature pair, evaluate a dense
interaction, and only then discard what could have been excluded earlier.

This is a principle for biological machine learning: use knowledge of the
system to structure what remains to be learned and computed. It guides
representations, learned mechanisms, derivatives, layouts, and execution
together. The discussion below establishes that principle; it does not claim
that the current implementation realizes every consequence.

## Complete state and incomplete measurement

At the cleanest limit, a sufficiently complete cellular state contains
everything needed to determine subsequent evolution within the regime of
interest. It may include expression, chromatin, signaling, environmental
conditions, and other relevant information. Under deterministic dynamics,

$$
x_{t+\Delta}=\Phi_{\theta,\Delta}(x_t),
\qquad x_t\in M,
$$

the state is Markov: once the current complete state is known, the past adds
nothing to the prediction. Equal states under the same conditions and model
have equal futures. For stochastic dynamics, the corresponding statement is
equality of conditional future distributions, rather than identical realized
outcomes. External forcing must be included in the state or specified as part
of the prediction's conditions. Markov sufficiency does not itself imply
sparsity; biological and model constraints supply the restrictions to exploit.

An experiment usually measures an image of that state,

$$
y_t=h(x_t),
$$

possibly with additional measurement noise. A transcriptional profile is not
the complete state merely because it is the available observation. Chromatin,
signaling, or other regulators can differ between cells with the same measured
profile. If, illustratively, the measurement selects coordinates so that
$x=(y,z)$, then

$$
\dot y=f_{\theta,y}(y,z).
$$

The observed relation among features in $y$ can change because the unobserved
configuration $z$ changed. Observed features and history may provide proxies
for that configuration. An exact reduced description can require memory even
when the complete system is Markov. Projection methods such as the
[Mori–Zwanzig formulation](https://math.berkeley.edu/~chorin/CHK02.pdf)
make this appearance of unresolved effects and memory explicit.

Broad attention offers a flexible way to use many observations to infer
missing context. Its usefulness does not establish that every pair of
underlying cellular variables can directly interact, or that every interaction
requires the entire cell as context. Knowledge of the underlying restrictions
can reduce the inference and computation required. The model may still need
to infer hidden context, retain history, or represent uncertainty when the
experiment does not supply it. Unobserved regulators are not automatically
recoverable from the available measurements.

## Restrictions on the dependency field

For a smooth deterministic continuous-time model, write

$$
\dot x=f_\theta(x),
\qquad
J_{ij}(x)=\frac{\partial f_{\theta,i}}{\partial x_j}(x).
$$

In meaningful cellular coordinates, the Jacobian describes instantaneous local
dependencies of the evolution law. Its field over the admissible state domain
describes where those direct dependencies can occur. Cellerator need not
materialize a full Jacobian to exploit a model's knowledge of its support.

There are two complementary restrictions. A **global exclusion** states that
a direct dependency is absent throughout the declared domain:

$$
J_{ij}(x)=0\qquad\text{for every admissible }x.
$$

If this exclusion belongs to the model's structural contract, the candidate
pair can be omitted before evaluating the current context. During learning,
that contract must hold for the parameter family to which the preparation
applies, or structural changes must trigger appropriate reconsideration.

A **conditional restriction** can establish absence using a smaller context
set $C_{ij}$. For an illustrative eligibility function $q_{ij}$,

$$
q_{ij}(x_{C_{ij}})=0
\quad\Longrightarrow\quad
J_{ij}(x)=0.
$$

An eligible pair need not have a nonzero dependency at every state. Eligibility
can be a conservative envelope. The useful knowledge is which inputs suffice
to establish exclusion; it need not specify the complete interaction law.
Those inputs can themselves include inferred latent variables. A reliable
small context set is a biological or model assumption to establish, not a
consequence of Markovianity alone.

The condition's function, its parameters, and the interaction that remains
eligible may be learnable. Knowing what information they can depend on already
creates an opportunity to reduce their inputs and candidate work. **The model
defines and evaluates the conditions.** Cellerator supplies ways to express
their input dependencies, locate the relevant information, and specialize
execution accordingly. Evaluating a condition is itself computation with
dependencies; it must not introduce an undeclared need for the entire state.

A missing experimental observation or an unlisted relationship is not a global
exclusion. An established restriction, a model-imposed assumption, a suggested
prior, and a small learned numerical value have different meanings. Exact
omission must preserve the declared computation; an approximate omission must
be identified and assessed as an approximation. A certificate that one input
has no influence also does not imply that the entire output or mechanism is
zero.

Pairwise derivatives are notation for this explanation, not a mandatory model
representation. A joint mechanism may read several participants and a separate
conditioning set; participants may overlap across mechanisms and participate
in feedback. Pairwise Jacobian support does not identify a unique mechanism
decomposition or fully describe higher-order interactions. Cellerator should
enable those joint operations directly, with their own declared dependencies.

## A worked example

Consider the complete state $x=(a,b,c,z)$ and the deterministic system

$$
\dot a=-a,\qquad
\dot b=-b+g(z)a,\qquad
\dot c=-c+b,\qquad
\dot z=0.
$$

Here $g$ is a fixed smooth gate that is identically zero on a declared region
and positive on another. This is a mathematical illustration, not a validated
biological model. The hidden regulator is included in the complete state, and
its evolution is specified even though it remains constant in this example.

The direct dependency of $b$'s evolution on $a$ is

$$
J_{ba}=g(z).
$$

Only $z$ is needed to determine whether this coupling is eligible; neither
$c$ nor every possible pair of features must be inspected. The gate also
contributes its own possible dependency,

$$
J_{bz}=g'(z)a.
$$

Ignoring the regulator while exploiting the gate would therefore discard part
of the mechanism. If the experiment observes only $y=(a,b,c)$, equal observed
profiles with nonzero $a$ can have different $\dot b$ because their hidden
$z$ values give different $g(z)$. The full state remains Markov; the
measurement has concealed the condition.

There is no direct instantaneous $a$-to-$c$ dependency: $J_{ca}=0$.
Nevertheless, influence can propagate through $b$. Since $z=z_0$,

$$
c(t)=e^{-t}\left(c_0+b_0t+\frac{g(z_0)a_0t^2}{2}\right),
\qquad
\frac{\partial c(t)}{\partial a_0}
=e^{-t}\frac{g(z_0)t^2}{2}.
$$

This derivative is nonzero for $t>0$ when the gate is positive. A sampled
finite-time transition can consequently have denser dependencies than the
instantaneous evolution law. The shortcut exploits the composition of local
mechanisms while retaining their indirect effects.

## Exploit the geometry throughout computation

These are connected design opportunities, not required runtime stages or a
prescribed object hierarchy. The same semantic structure can admit different
representations and algorithms for different operations and hardware.

**Candidate construction and context.** Build indexes over admissible
participants and mechanisms instead of enumerating an unrestricted pair
space. Index the inputs required by each condition, including latent or shared
context where the model uses it. Changes to those inputs identify conditions
that may need reevaluation, subject to the model's temporal and invalidation
contract. Independent conditions need not repeatedly inspect the entire cell.
Inferred context still has a computation cost; indexing does not make an
unknown regulator known.

**Learned operations and derivatives.** Evaluate eligible joint mechanisms
using their required inputs. A learned function can remain expressive within
its admissible domain while avoiding inputs and interactions excluded by the
model. Preserve gradients through participant values, condition evaluation,
latent inference, and parameters whenever training requires them. A zero
realized value is insufficient to skip backward work: $uv=0$ at $u=0$
can still have derivative $\partial(uv)/\partial u=v$. By contrast, a fixed
smooth gate identically zero on an open region has zero local derivatives
there. Forward and derivative work must each follow the operation's contract.

**Layout and persistent preparation.** Organize values, parameters, indexes,
and gradient mappings around the declared participants, overlapping mechanisms,
and relevant context. Prepare address maps, reusable buffers, packing, and
appropriate kernel specializations for that structure. Many logical units can
occupy a few bulk allocations. Geometry does not require a heap object or
tensor allocation for each interaction, module, or cell. Logical identities
must survive physical reordering, packing, and gradient accumulation.

**Distribution and communication.** Partition computation along actual
dependencies. A suitable module may execute locally while exchanging the
information required by cross-module mechanisms. Conditioning, normalization,
readouts, and backward operations contribute dependencies too; their costs
must be represented. **One Cell may legitimately span several devices.** Its
logical boundary is the modeled instance, rather than a resident allocation.
Populations are another use case, not a prerequisite for exploiting geometry.

**Repeated learning and execution.** Cellerator builds models for learning the
same system repeatedly. Knowledge of that system can support reusable
preparation across compatible observations, model instances, and training
iterations. Values, parameters, and activity may change while an admissibility
envelope and the inputs controlling eligibility remain valid. Changing those
structural assumptions, the represented variables, or their biological regime
requires reconsidering the affected preparation. A temporary active set and a
reusable set of possible dependencies have different lifetimes.

This reuse extends beyond execution plans. Applicable knowledge of admissible
dependencies and their conditioning inputs should be expressible independently
of one network architecture or task-specific readout. Expression prediction,
perturbation questions, and auxiliary measurements can use compatible parts of
the same structural foundation. A new objective may require additional
information or mechanisms; it need not discard what is already established
about the system. Reuse preserves the meaning and applicability of supplied or
learned structure, without assuming that arbitrary models share parameter
formats, latent coordinates, or interchangeable weights.

The computational advantage must survive complete costs. For $n$ features,
an unrestricted pair domain has $n^2$ entries. A known admissible domain
with $m$ pairs permits indexing and candidate work around $m$, with the
additional cost of evaluating the required context and active mechanisms.
Higher-order mechanisms require their own participant counts. These savings
are feasible when avoided work exceeds preparation, context inference,
condition evaluation, maintenance, packing, movement, synchronization, and
execution overhead over the reuse interval. Irregular access or communication
can dominate. Dense computation can be appropriate within modules or wherever
the model requires it; the advantageous physical implementation must be
measured. This argument establishes an opportunity, not a speedup claim.

## Use the geometry to discover better implementations

Begin a design with the computation's dependencies and the information that
can establish them, before committing to a container, kernel, or scheduling
unit. The purpose of this geometry is to make better designs possible across
data structures, numerical algorithms, kernels, scheduling, and execution.
It should prompt alternatives to an inherited implementation, not merely
suggest how to tune that implementation.

Useful questions include:

- What can be ruled out before candidates, scores, intermediates, or messages
  are constructed? What is the least information needed to establish that
  exclusion, and what does obtaining it cost?
- Which facts can remain reusable as states, parameters, architectures, or
  questions change? Which changes genuinely require rediscovery, rebuilding,
  or moving data?
- What information must be read jointly, what context is reused, and how do
  overlapping mechanisms affect locality? What representation makes those
  access patterns inexpensive without losing logical identity?
- Can the operation exploit a restriction directly, without translating it
  through an unnecessarily general representation? Does the restriction
  suggest a different algorithm, rather than a smaller version of the same
  dense computation?
- What unit of work best fits the dependencies and hardware? Where is
  coordination actually necessary, and what can proceed independently across
  parts of a Cell or across Cells?
- Which alternative preserves the required results and derivatives while
  reducing complete cost over the intended workload and reuse interval?

These questions do not select an adjacency structure, tensor format, partition,
dispatcher, compiler, or launch strategy. A dependency can sometimes be
represented implicitly; a useful work unit need not correspond to a biological
entity or an allocation. The design should follow the structure and the
operation's semantics, with candidate choices tested against relevant
alternatives. Established model meaning and numerical correctness constrain
that freedom; a preferred implementation should not define the biology.

## Meaning, evidence, and the Cell interface

Here *cellular geometry* means the organization of admissible dependencies
over biologically meaningful state space, including the conditions that make
them possible and the measurement that exposes or conceals their regulators.
It includes more than a static graph, geometric distance, or physical
proximity. The useful restrictions belong to the modeled system; storage
layout and device placement are realizations designed to exploit them.

The representation and regime matter. Arbitrary coordinate changes can obscure
Jacobian sparsity. On a constrained state manifold, dependencies must be
interpreted through admissible perturbations rather than independently varying
coordinates that cannot vary independently. A Jacobian field describes local
dependencies, not the complete evolution law by itself. In stochastic models,
dependencies can also enter diffusion, noise correlations, and jump mechanisms;
the drift Jacobian alone is insufficient. These distinctions determine which
shortcuts preserve a particular model.

Structural knowledge and experimental inference have different authority.
Exclusions can come from applicable biological knowledge or explicit model
assumptions; experiments may constrain additional structure and its unknown
functions. Unknown relationships remain possible unless the model deliberately
restricts them. A learned sparse fit alone does not establish natural
impossibility, causal identification, or sufficiency of the experiment. A useful
model need not recover every hidden regulator to exploit restrictions it can
legitimately establish.

This rationale should guide the eventual Cell interface. A Cell must permit
model-native information to be represented and accessed, including relevant
latent context, uncertainty, history, or distributed state. It need not expose
one complete measured vector or prescribe a timestamp, trajectory container,
device, or agent runtime. The model evaluates its conditions and learned
operations; the Cell interface enables access to the information those
operations require. The concrete Cell class design remains a separate task.
Neither FrameShift nor Cytoplast defines the universal Cell contract.

A future consequence is reusable, precalculated preparation for a specified
experiment, including analysis of what can be learned to omit and community
contributions. Declared exclusions, suggested structure, and optimization hints
would retain their different meanings; absence from such a map would not alone
establish impossibility. Their formats and publication machinery are separate
work. The present design commitment is to make restrictions on dependencies
and their conditions expressible, and to exploit them across learning, layout,
and execution.
