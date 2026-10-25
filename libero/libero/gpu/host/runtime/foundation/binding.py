"""Resident CUDA array identity and allocation ownership helpers."""
import warp as wp

class BindingError(ValueError):pass

def dtype_name(dtype):
    return getattr(dtype,"__name__",str(dtype)).removeprefix("wp.").removeprefix("warp.types.")

def array_token(array):
    if not isinstance(array,wp.array) or not array.device.is_cuda:
        raise BindingError("actual CUDA Warp array required")
    return (id(array),int(array.ptr),tuple(array.shape),tuple(array.strides),
            dtype_name(array.dtype),str(array.device),int(array.capacity),
            bool(array.is_contiguous))

def allocation_origin(array):
    """Follow Warp's actual retained view parents to its owning allocation.

    A raw ptr wrapper has no intrinsic owner and is deliberately not inferred
    from address equality. It can only bind as an explicit canonical alias.
    """
    array_token(array);seen=set();chain=[]
    while isinstance(array,wp.array):
        if id(array) in seen:raise BindingError("cyclic view owner chain")
        seen.add(id(array));chain.append(array)
        parent=getattr(array,"_ref",None)
        if parent is None:break
        if not isinstance(parent,wp.array):raise BindingError("external allocation owner unsupported without explicit native allocation proof")
        array=parent;array_token(array)
    allocator=getattr(array,"_allocator",None);deleter=getattr(array,"deleter",None)
    if allocator is None or deleter is None or getattr(deleter,"__self__",None) is not allocator:
        raise BindingError("array has no authentic Warp allocator/deleter owner")
    if int(array.capacity)<=0 or int(array.ptr)<=0:raise BindingError("empty physical allocation; use one unread sentinel")
    return array,chain

def origin_token(allocation):
    root,chain=allocation_origin(allocation)
    if root is not allocation:raise BindingError("allocation argument is a view, not the actual owner")
    return (array_token(root),id(root._allocator),id(root.deleter),
            id(root.deleter.__self__),tuple(id(x) for x in chain))
