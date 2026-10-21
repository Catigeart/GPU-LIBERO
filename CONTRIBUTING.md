# Contributing to Flash-LIBERO

## One Source Repository

Develop, review, and release from this repository. Use the current working tree, including intended uncommitted changes; historical implementation directories and published source copies are not separate development branches. Preserve later fixes when bringing in an older experimental patch.

Start a focused branch from current `main` and submit a pull request:

```bash
git switch main
git pull --ff-only
git switch -c feature/your-change
```

Keep changes small enough to review. Describe the affected behavior, source dependencies, checks, and any domain restrictions. Keep raw inputs, large evidence, compiler caches, SDKs, and binary outputs outside the tracked source tree.

## Development Installation

Python 3.10 or newer is required. Reuse an existing compatible environment when available:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

The environment must already provide the project's runtime dependencies and a compatible setuptools backend (`>=64,<81`). For a new environment, follow the installation section in README. The import namespaces remain `libero.libero` and `libero.gpu`; the distribution name is `Flash-LIBERO`.

## Checks

```bash
python tools/check_source.py
python tools/check_source.py --cpu-tests
```

The CPU check command runs selected installation, resource/toolchain, source-dependency, batch/memory/deferred-build, and fake vector-protocol tests. Graphics-owner and Observable tests remain available as documented in [tests/README.md](tests/README.md) when their dependencies are installed. CPU checks do not compile CUDA or run GPU physics.

For new GPU mathematics, assess the modified independent function using FP64 error accumulation, residuals, and sensitivity as appropriate. Preserve exact index/order/copy/capacity/owner/error contracts. Existing accepted numerical conclusions are not requalified merely because of workspace or packaging changes. A complete application numerical test is not a prerequisite for adopting an independently accepted function.

Measure local performance against the current implementation using the relevant real model sizes and world counts. Report cold construction, warm host enqueue, GPU intervals, launches, and work separately. Keep conditional benefits scoped; revert negative domains and preserve fallback choices. Do not combine different baselines or describe local results as whole-step speedups.

Serialize GPU execution, nvcc, and LTO on a shared device. Reuse build products only under their existing source/toolchain/policy/owner binding and invalidation rules.

Use the [runtime path index](libero/libero/gpu/ARCHITECTURE.md) to locate default,
fallback and explicitly configured implementations. Product GPU modules must not
import development scripts. Common model/build/runtime helpers belong in their
own package; experiments and generated evidence stay outside the source checkout.
The [environment lifecycle guide](libero/libero/gpu/ENVIRONMENT_LIFECYCLE.md)
separates runtime resource ownership, fixed offline products and retired storage
maintenance. Current lifetime tests and legacy collector tests are kept separate;
see [the test index](tests/README.md).
Use the [stable validation entry](tools/validation/README.md) instead of creating
another dated copy of actions, reference comparison or timing calculations.

## Releases

Build one wheel and one source archive from the current source:

```bash
python tools/release/build.py --output-dir dist
```

The build tool uses the installed setuptools PEP 517 backend. It constructs a temporary packaging context, checks source and license inclusion, and generates each artifact once. It does not create another maintained Git checkout or push source copies. Output JSON is build metadata and belongs with the artifacts, not in the Git repository.

Merge accepted changes into `main`. A version tag must match the version declared in `setup.py`, for example `v0.2.0` for version `0.2.0`. Pushing a new `v*` tag runs the release workflow and attaches the checked artifacts to a GitHub release. Existing release assets are not silently overwritten. PyPI publishing is not configured.

## Documentation and Licensing

Write public documentation in English. Keep original source notices and component license exceptions. Project-original contributions use the MIT terms in [LICENSE](LICENSE). See [Third-Party Notices](libero/libero/gpu/THIRD_PARTY_NOTICES.md) for inherited and external code.
