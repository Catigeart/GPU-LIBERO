# Environment and offline cache lifecycle

Environment lifetime follows CPU LIBERO: construction owns the model and runtime,
reset updates or replaces those owners, and close releases them. GPU execution
additionally needs fixed offline CUDA products. It does not need a persistent
filesystem directory for each environment or reset generation.

## Ownership

| Resource | Owner | Runtime behavior |
| --- | --- | --- |
| Task XML, initial states, prepared FP32 MJB | Offline project | Read and verify; never register, evict or rewrite |
| CMake libraries, native products, generated constraint sources | Offline build | Select completed products; check the actual ABI and supported architecture |
| Native model, decoded arrays, model metadata, replacement preflight | Environment | In memory; release with the runtime |
| Device arrays, workspace, Graph, renderer and stream | Actual runtime | Separate mutable owners; synchronize and release explicitly |
| Remaining native loader files | Environment temporary directory | Remove after successful runtime release |

The default independent FP32 backend reads fixed scenes and libraries without
creating runtime cache directories or taking writer leases. Only offline scene
preparation takes a per-scene writer lock. Normal batch reset preserves its shared
model; the public single-environment hard-reset and XML/parameter replacement
semantics remain separate.

Native compatibility intake transports immutable metadata and actual model
owners in memory. Model cloning uses MuJoCo's model copy rather than writing and
reopening MJB files. The native dynamics factory deep-copies model metadata once
and passes it directly to its components; it no longer encodes an authenticated
JSON copy for each downstream consumer. Component preparation does not reopen
model/decoder/source records. Model descriptors, actual dimensions, participating
array values, topology, capacity and owner replacement checks remain enforced.
An existing native model passed to Newton is retained without reserialization;
an explicit external MJB path still has its file identity checked. Initial model
snapshot generation and explicit diagnostic export can still serialize models.

The forward adapter validates its actual backend, stream and buffer owners.
It no longer authenticates Python method objects, bytecode or default arguments.
Ordinary and wide-angle dynamics factories register model-scoped runtime
capabilities without source inventories. Their IDs use the explicit implementation
family and full model identity. Missing source provenance is represented by
`source_sha256=None`; no placeholder digest is fabricated. Such capabilities
cannot inherit source-qualified numerical acceptance records. Structural checks,
implementation-domain restrictions and the explicit offline qualification API remain.
Legacy `source_records` / `metadata_record`
component keywords remain accepted while internal callers are being simplified;
they no longer license native model intake. Static fields are imported into one immutable model snapshot; components share
its arrays and retain layout/owner checks. Model-update semantics remain.

The handwritten collision path retains an in-memory model binding: immutable
static arrays, model context, options, device/capacity and installed plan owners.
It does not scan the source tree, acquire source-file leases or authenticate
compiler dependencies, Python bytecode or build manifests. Device binding still
checks the actual loaded exports, argument layouts, buffer owners and stream;
each binding validation checks the collision model once. Model replacement and
close invalidate the old owners as before. These checks never authorize device
uploads or numerical support for a new model family.

The five native environment factories share a descriptive decoder-record helper.
It reuses the model factory's recorded digest without reopening or authenticating
the decoder source. This record is diagnostic metadata, not fresh provenance.
Portable constraint intake also excludes decoder/source provenance from its model
descriptor and no longer retains or repeatedly verifies source-file pins. Its
fresh native metadata check still compares physical arrays, topology, options and
dimensions, rejects unsupported native callbacks, and enforces parameter policy
and actual model ownership. This uses `native_model_metadata` with source
provenance disabled; normal model preparation can still record offline provenance.
Constraint package loading uses the offline instance index, compares the actual
model manifest and requires the indexed files to exist. It does not rehash every
source file on a runtime hit or regenerate a missing instance. Explicit offline
generation/indexing still checks source freshness and product consistency.

Native model updates publish physical array/option permissions without a producer
source certificate. Preparing, validating and committing an update no longer reads
the update implementation to authenticate it. Historical schemas with a `producer`
field are accepted when their physical permissions match. Actual model identity,
parameter layouts/values, owner replacement, transaction state and old-binding
invalidation remain checked. Fresh model preparation may still record decoder
provenance; model snapshots and repeated physical model comparisons remain.
The unused foundation source certificates, compiler provenance sessions, Graph
audits and source-reason witnesses have been removed, together with their file
lease provider and Windows bridge. Foundation now contains the resident forward
adapter, CUDA array ownership helpers, dimension checks and device admission/reset
logic. It no longer allocates unused reason-witness buffers or status-routing
tables. Actual capacity, state epochs, buffer ownership and first-error propagation
remain checked. The unused job-source authority and private-payload source-lease
wrapper are also removed. Newton's generated Python instance loader checks the
requested dimensions and file presence at runtime; source freshness checks belong
to explicit offline preparation. Other FP64 component preparation paths can still
read source inventories.

Stateless `NativePlan` code loads from a fixed shared library. Each caller retains
its own parameters, device buffers and library wrapper. Some direct collision and
Newton loaders still materialize private artifacts and evidence. Those small
files use Python `TemporaryDirectory` under the system temporary location, not a
persistent cache generation. No model MJB is copied there. A failed GPU close
retains this owner for retry; a successful GPU close is not repeated if temporary
file cleanup fails. Independent generations cannot delete each other's files.
Normal exception handling and interpreter cleanup release temporary directories;
an uncatchable process kill can leave them for ordinary temporary-file maintenance.
No runtime directory scan or PID-age deletion heuristic is installed.

Private library wrappers balance `dlopen` through a GC finalizer after all borrowed
functions are released. Runtime close synchronizes device work first. Fixed code
may remain process resident. Private MathDx products disable GNU-unique host
symbols and hide vendor archive host symbols to avoid glibc `NODELETE` retention.
Per-instance Python launchers are not registered in `sys.modules`; releasing them
also removes their source from `linecache`.

## CPU compatibility boundaries

Resource ownership follows CPU LIBERO; complete environment behavior is not
identical. The default FP32 batch shares one static model and resets world state.
It rejects `hard_reset=True` and does not independently rebuild each world's XML
or fixture model. The native single-environment compatibility entry retains hard
reset and XML replacement, subject to matching offline products and supported
model families. CPU LIBERO can construct a model directly from new XML at runtime.

## Offline products

`LIBERO_GPU_CACHE_DIR` defaults to `~/.cache/libero-gpu`:

```text
project/                    task catalogues, generated sources and CMake products
project/mjwarp/models/      fixed FP32 scene inputs
products/native-v1/         ordinary native libraries/PTX and build metadata
products/signed-v1/         legacy products, supported for read-only runtime reuse
```

Ordinary public constructors, reset/replacement calls and direct facade operations
are load-only. A missing product raises `MissingCompiledProduct`; it does not grant
permission to compile. The frozen project context does not take the offline writer
lock. An explicit offline scope may compile but does not grant later public calls
permanent compiler permission.

The CMake substep and FP32 controller locators do not execute Ninja dry-runs.
Substep loading also skips compiler discovery, source-project discovery and
recipe writes. CMake owns dependency freshness during explicit offline builds;
after editing CUDA, rebuild before starting a new process. Runtime bindings
continue to check the loaded ABI, precision, supported dimensions and numerical
configuration. Model-input and component-specific loader checks remain.

The independent FP32 runtime, bridge, fused kernels and shared handwritten
physics loaders no longer compare libraries with CUDA/CMake source hashes.
They check the existing exported entry points, argument layouts and device
architecture where provided. No CUDA interface or numerical kernel changed in
this cleanup. Source fingerprints remain available to explicit diagnostic tools;
CMake handles source dependencies during offline builds. Interface changes must
update the ABI contract, including a version when layout size alone is insufficient.

Portable scene parameters no longer depend on exporter source fingerprints.
Their format, task/seed, scene/model association and body layouts remain checked.
Content integrity checks run during explicit offline preparation. Modern scene
selection uses engine versions, XML, options and portable parameter metadata;
asset/library hashes are offline validation records, not runtime lookup keys.
Legacy model discovery and conversion are offline-only (see below).

Native AOT readiness reads the current manifest, checks precision/world count/
scene seed, and selects the requested task and observation profile. It no longer
hashes the source tree or the complete artifact bank. Actual physical products
are checked by their loaders. The optional ready-session API is now a compatible
build-selection handle, not a source-tree certificate or cached qualification.
Offline build/resume keeps its dependency and artifact checks.

Prepare a missing native compatibility profile explicitly, for example:

```bash
python -m libero.gpu.build --compatibility --precision fp64 --suite libero_10 --task-ids 4
```

The product cache has no signing keys or HMAC envelopes. New offline builds
publish separate library/PTX files and ordinary build metadata. Runtime selection
uses generated-source identity, options, architecture and required symbols.
Dependency, toolchain and artifact-hash validation belongs to offline builds.
The generic native loader does not query the compiler on a cache hit; actual
symbols and ABI remain checked at loading. Old signed products can be read
without their old private key and are not rewritten on runtime access.

Source sidecars are retained for consumers that inspect generated declarations.
An old product without sidecars materializes those declarations in its owned
temporary directory. The direct collision loader checks the actual contact ABI
and CUDA function resources. It no longer parses ELF symbols, inspects process
maps, hashes SDK headers, or writes per-instance module identity files. Compiler
and artifact fingerprints are checked during offline preparation, not runtime
reuse. Runtime selection still uses project sources, architecture and compiler
options. Existing libraries retain their private lifetime and launch errors
still propagate. Modern cache hits materialize the library/PTX without an extra
`BUILD.json`; the legacy directory loader still reads its build recipe.

Generated Newton CUDA preparation is split between
`host/build/newton.py` (offline source expansion, compiler dependencies and linking)
and `host/runtime/newton/native_product.py` (loaded-library ABI/resource binding).
The runtime selects an ordinary native-cache product using precision, architecture,
dimensions, launch options and its in-memory parameter-layout/schedule signature.
It does not expand CUDA headers, hash SDK archives, locate nvcc, or write generated
CUDA/manifest files. Offline rebuilds still check actual dependencies and invalidate
changed products. This uses the existing product cache and its storage budget.

The library exports `newton_kernel_info`. One query obtains actual kernel and
device limits, checks the parameter ABI, architecture, thread count and combined
static/dynamic shared memory, and retains the result with that library owner.
The outer capacity installer reuses this result. Neither path invokes `cuobjdump`
or repeatedly hashes the loaded binary. The S07 reference bridge also defers
source reads and NVRTC lookup until its standalone implementation is actually used;
the native Newton path only needs its argument buffers.

Old generated Newton binaries lack the query export and the new launch-selection
record. Prepare the required FP64 compatibility profiles offline once after this
update; runtime does not compile or silently adopt those old binaries. No bank-wide
rebuild is performed during environment creation. The independent CMake FP32 path
is unaffected. Other FP64 components can still require the installed toolkit;
these changes do not make the entire FP64 environment toolkit-independent.

`CapabilityRegistry` checks only model scope, supported dimensions, capacity,
mutable fields and execution scope. Runtime admission is independent of author,
reviewer, reports and numerical verdicts, for both legacy mode labels. Those
records live in `qualification.py` (`QualifiedRegistry`); an explicit offline
`load_qualified_registry` still verifies sources/reports. `qualified_record()`
reports numerical acceptance separately and cannot widen runtime support.
Generation checks require both model domains and explicit supported update fields;
the model-update boundary still validates actual topology and changed data.

Offline products have a separate persistent budget:

| Variable | Default |
| --- | ---: |
| `LIBERO_GPU_PRODUCT_CACHE_MAX_BYTES` | 4 GiB |
| `LIBERO_GPU_PRODUCT_CACHE_MAX_ENTRIES` | 4096 |

Offline compatibility preparation requires persistence. Exceeding that budget
raises `ENOSPC`; interrupted publications still count toward it. The compatibility
manifest records `preparing`, `failed` or `complete`. A private unpersisted build
cannot mark the profile ready. Remove or rebuild fixed inputs only outside training.

Environment close releases its runtime resources and leaves offline products
intact. The retired storage manager and its maintenance CLI are no longer shipped.
Existing legacy cache directories are left untouched; the old disk, model, run
and scene budget variables do not govern current environments.

## Fixed scene loading and explicit FP64 construction

Normal FP32 scene construction uses a direct `modern-scene-v2` directory selected
by the engine version, imported XML, options and portable parameter record. It
checks the receipt format, model presence, task binding and actual model layout.
It does not hash assets, the MuJoCo library, initial-state files, portable NPZ
contents or MJB contents, and does not scan historical scene directories. Native
MuJoCo loading and body/joint mapping checks still reject incompatible inputs.
The key hashes in-memory selection data; they are not source authentication.

Full content checks belong to explicit offline scene preparation (`prepare=True`).
That path validates assets, the native library and model bytes, and can import a
matching old cached MJB and publish a v2 receipt/model without recompiling CUDA.
Existing old directories remain untouched for other users/processes. A runtime
miss asks for offline preparation; it never migrates or writes a scene. Editing
assets or portable parameter contents requires offline preparation before reuse.
Only the matching scene profiles need this one-time migration; CUDA products are
unaffected. Full model-content validation is deliberately no longer a runtime
promise.

FP64 assembly no longer accumulates source-tree SHA256 inventories. Offline
compiler dependency tracking remains separate. Dynamics, postprocessing, Newton
and collision constructors accept explicit implementation types/loaders; their
assembly factories no longer rewrite constructor source with Python ASTs. The
recovery cleanup route likewise passes physical factories, assemblers and callback
factories explicitly. Model-replacement admission, release/restore/publication
order and retryable cleanup owners are retained. Numerical launch-plan/kernel
composition still uses its existing AST machinery; this cleanup changes host
construction rather than numerical code generation.

## Shared static inputs and continuation transport

FP64 dynamics factories create one `PreparedModelArrays` snapshot in
`host/runtime/core/static_readset.py`. Each native field is copied into immutable
bytes-backed storage and checked against the model metadata once at this boundary.
Kinematics, frames, mass, passive/actuation, Jacobian and postprocessing borrow the
same fields; required ABI conversions still allocate their converted arrays.
Standalone component callers import their own snapshot. Shape/device/owner checks
remain active, while mass/kinematics/frame/camlight bindings no longer rehash
unchanged static payloads. Updating model parameters creates a new snapshot.

`host/runtime/step/continuation.py` downloads the old state once, checks the entire
transfer layout before writing, uploads to the new owners and synchronizes before
publication. Normal transport neither hashes every dynamic field nor downloads
it again to prove the copy. Explicit `verify=True` on both snapshot and restore
retains payload-hash and full readback checks for diagnostics. Result metadata
reports `transport_completed=True`; `data_bitwise_equal` is `None` unless that
comparison actually ran. Small controller-static transition checks and ready/
model-generation/error publication checks remain; these are not full-state
readback tests. Observer history and failure-cleanup ownership are unchanged.
