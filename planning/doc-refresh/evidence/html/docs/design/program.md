# One scientific program, three different questions

A genome describes possibilities, a cell realizes a particular state, and an observation reveals only part of that state. These repositories address different computational obstacles between those levels.

**Baseplane** asks how genome-wide information can be represented hierarchically without charging every subsequent calculation for an equally expensive token at every base. Exact sequence is the ground truth beneath experimental learned representations. The useful hierarchy must be inferred during representation, not assumed from a complete annotation map.

**Cellerator** asks how a biological system's repeated structure can become executable structure. It discovers/prepares useful organization, generates or assembles suitable execution, and separates reusable topology from changing values. Its intended runtime can select and reorganize computation using prediscovered gates rather than treating a compiled plan as a permanently fixed path.

**GlassHelix** asks what state and mechanism the available observations actually support. A useful outcome can be a constrained family of explanations, a shared prediction or a discriminating perturbation—not necessarily a unique mechanism.

The collaboration is not a one-way pipeline. In the intended feedback loop, GlassHelix can support a new state variable or relation; Cellerator can represent and specialize it; and that state may inform where Baseplane needs richer sequence information. This loop is a design goal, not a claim that automatic cross-project mechanism promotion is complete. A supported chromatin variable still needs a measured value, estimate or update law. Specialization must preserve assumptions, scope and a way to reconsider them.

The common wager is that **biological organization can be computational structure, not just metadata attached to a tensor**. Whether an organization saves work is an empirical question. Better scientific models can sometimes need more computation, not less.

[Baseplane](https://github.com/tumlinso/Baseplane) owns sequence meaning and provenance. [Cellerator](https://github.com/tumlinso/Cellerator) owns general structured numerical execution and compilation. [GlassHelix](https://github.com/tumlinso/GlassHelix) owns scientific dynamics/inference. CelleraTorch is a Cellerator adapter; [CellShard](https://github.com/tumlinso/CellShard) supplies physical persistence/distribution without taking over those scientific meanings.

These boundaries allow implementation collaboration, not three copies of the same planner or state store. The precise installed interfaces and currently supported integrations are described in each repository's development pages.
