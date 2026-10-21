# Environment and offline cache lifecycle

Environment lifetime follows CPU LIBERO: construction owns the model and runtime,
reset updates or replaces those owners, and close releases them. GPU execution
additionally needs fixed offline CUDA products. It does not need a persistent
filesystem directory for each environment or reset generation.

## Ownership

| Resource | Owner | Runtime behavior |
| --- | --- | --- |
| Task XML, initial states, prepared FP32 MJB | Offline project | Read and verify; never register, evict or rewrite |
| CMake libraries, signed native products, generated constraint sources | Offline build | Load completed products with source, binary, architecture and ABI checks |
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
reopening MJB files. Byte identity, decoder/source pins, capacity, owner replacement
and error checks remain enforced. Model export writes files only when explicitly
requested by the diagnostic export API.

Stateless `NativePlan` code loads from a fixed signed file. Each caller retains
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
products/signed-v1/         completed, signed native compiler products
storage-v1/                 retired managed runtime storage; not used by environments
```

Ordinary public constructors, reset/replacement calls and direct facade operations
are load-only. A missing product raises `MissingCompiledProduct`; it does not grant
permission to compile. The frozen project context does not take the offline writer
lock. An explicit offline scope may compile but does not grant later public calls
permanent compiler permission.

Prepare a missing native compatibility profile explicitly, for example:

```bash
python -m libero.gpu.build --compatibility --precision fp64 --suite libero_10 --task-ids 4
```

The completed-product loader verifies the original signed receipt and current
source/dependency/toolchain component bytes, flags, architecture and symbols. A
change in unrelated caller shell variables does not require a new binary or new
signature. Offline publication retains its full compiler environment identity.
Prepared source sidecars allow stateless loads without compiler dependency queries;
older products without sidecars retain the checked cold-load path.

Signed offline products have a separate persistent budget:

| Variable | Default |
| --- | ---: |
| `LIBERO_GPU_PRODUCT_CACHE_MAX_BYTES` | 4 GiB |
| `LIBERO_GPU_PRODUCT_CACHE_MAX_ENTRIES` | 4096 |

Offline compatibility preparation requires persistence. Exceeding that budget
raises `ENOSPC`; interrupted publications still count toward it. The compatibility
manifest records `preparing`, `failed` or `complete`. A private unpersisted build
cannot mark the profile ready. Remove or rebuild fixed inputs only outside training.

## Retired storage maintenance

Environment operations no longer import the old storage manager. The old disk,
model, run and scene budget variables do not govern newly created environments.
Historical registered storage is retained for an explicit offline maintenance pass:

```bash
python -m libero.gpu.host.build.legacy_storage
python -m libero.gpu.host.build.legacy_storage --collect
```

The collector preserves live leases, foreign/damaged markers, symlinks and
unregistered data. Prepared scenes leave its catalog without file deletion.
It does not remove historical unregistered `builds/`, `intake/`, `model-updates/`,
compiler projects, assets, toolchains, external Warp/driver caches or experiment
results. Existing registered native model copies remain subject to the legacy
collector's configured budget. Environment close does not manage offline assets.
