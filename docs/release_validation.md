# Release validation notes

This release publishes source, tests, configuration, and method documentation. It intentionally excludes generated images, training logs, checkpoints, and the local model/data directories.

The source project owner reports a prior 83-test pass and a low-budget RTX 3080 10 GB run with two episodes, six transitions, four PPO optimizer steps, and nonzero Actor, Critic, Mapper, and Alpha updates. Those local output files are not included in this repository. The publication-copy fast test run is recorded after its pre-push verification; it is separate from the prior model-backed run.

The code is a paper-based engineering reimplementation, not a reproduction of the paper’s private model weights, original full-scale training, human study, or reported performance numbers. Known approximations and unresolved paper details are documented in `paper_to_code_matrix.md`, `equation_audit.md`, `equation_mapping_addendum.md`, `execution_principles.md`, `reproduction_scope.md`, and `open_questions.md`.

## Publication-copy tests

The weight-independent tests were run in this release clone with the D-drive Python 3.12 environment: **77 passed, 6 skipped**. The skipped checks require saved model-backed output artifacts, which are deliberately excluded. The first invocation hit the managed environment's default temporary-directory access restriction; the successful run used a repository-local pytest temp directory. No model was downloaded and no training was run for this release.
