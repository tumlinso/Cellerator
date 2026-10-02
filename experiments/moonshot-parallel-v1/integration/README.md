# Native product2 integration

The native operation evaluates `y[i] = k[i] * x[a[i]] * x[b[i]]` in FP32. Direct CPU and CUDA calls and typed `prepared_program_v2` stages passed the retained tests. An installed C++ consumer links exported `Cellerator::product2` without compiling numerical source. Input and coefficient VJPs and a JVP with both input and coefficient directions are implemented. The tests exercise zeros, repeated inputs, empty support, generation checks, aliases, indices, typed axis admission and CUDA owner/context topology agreement.

The standalone Torch adapter passed 10 CPU tests. It uses the new native product library through `CELLERATOR_PRODUCT2_LIBRARY`; CUDA Torch tensors, mixed precision and higher-order differentiation remain unsupported. Existing indexed-mechanism files and libraries are separate dependencies.

Actual controller evidence: `4b0564f9-4c0b-4ab6-8d5d-a097ff5c095f`. The manifest binds source and transitive headers to retained clean-build command logs, output hashes, and actual execution evidence. Inputs and outputs were checked before and after execution. Actual dynamically resolved product libraries are hashed. Pure verification depends on the retained managed build workspace and binaries; a clean checkout needs fresh qualification. `capability.json` supplies callable paths, shape and precision policy, derivative support and hashed test evidence for the GH consumer. Consumer acceptance must separately bind the final delivered CE commit and live task completion.

No performance victory, biological fit, native optimizer ownership or sanitizer qualification is claimed.

Pure gate: `python experiments/moonshot-parallel-v1/integration/verify_product_evidence.py verify`.
