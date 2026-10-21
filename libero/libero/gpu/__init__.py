"""GPU arithmetic policy: permit FMA while preserving IEEE error handling."""
if __name__ != "libero.gpu":
    raise ImportError("GPU modules use the canonical libero.gpu namespace")

# Warp-specific configuration belongs to the optional reference backend.
# Importing the arithmetic policy or the independent CUDA runtime must not
# initialize a device or require an optional compiler framework.
