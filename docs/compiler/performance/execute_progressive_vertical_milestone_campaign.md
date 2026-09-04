# Progressive vertical milestone campaign v1

`progressive_milestones_v1.tsv` freezes one exact command and one output
artifact for each of eleven milestones. A campaign records command exit status,
source/toolchain/profile identities and SHA-256 of every available artifact.
Unavailable optional NVCC/custom-pass milestones remain explicit. Individual
milestones are reproducible independently and run under the benchmark mutex.
