# Small regulatory reuse demonstration

This is a post-Semantic-Spine-v1 C++ acceptance example, not a preprint result. Keep `regulatory_reuse.cc`, `fixture.hh`, `CMakeLists.txt` and this file together in `examples/semantic_spine_v1/`.

The normal program requires the provisional native API implemented by the epic. It must execute actual CUDA forward and transpose application, reuse one prepared topology across two value generations, and reject an old generation before changing output. It is not expected to build against the pre-epic baseline, and there is no mock successful backend.

The optional compile-time `CELLERATOR_SPINE_REFERENCE_ONLY` mode checks the toy fixture without Cellerator/CUDA. It explicitly reports that no Cellerator or GPU execution was tested and cannot satisfy acceptance.

After implementation, build target `ceSemanticSpineDemo` through the opt-in `CELLERATOR_BUILD_SEMANTIC_SPINE_V1` root integration and run `./build-ss1/ceSemanticSpineDemo --require-sm70`. See `planning/semantic-spine-v1/04_VALIDATION_AND_DEMO.md` for the exact future build steps, lifetime/numerical rules, and expected outputs.

The relation is an illustrative four-regulator/five-gene signed network with nine edges. Names are labels, not biological claims. Transpose is reverse accumulation, not an inverse or full learning algorithm. The CPU oracle exists only to validate small device results.
