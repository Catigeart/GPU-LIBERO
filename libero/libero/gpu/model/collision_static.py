"""Cold collision model extraction from the live MuJoCo 2.3.7 model."""
import numpy as np


def static_from_model(m) -> dict:
    """Extract the chain's static model arrays from a MuJoCo MjModel."""
    import mujoco
    i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    ngeom = int(m.ngeom)
    nbody = int(m.nbody)
    npair = int(m.npair)
    # dispatch table from the live library (frozen 2.3.7 image)
    import ctypes as ct
    lib = ct.CDLL(str(mujoco.__path__[0]) + '/libmujoco.so.2.3.7')
    table = (ct.c_void_p * 64).in_dll(lib, 'mjCOLLISIONFUNC')
    present = np.array([[int(bool(table[8 * r + c])) for c in range(8)] for r in range(8)], dtype=np.int32)
    # mesh data
    mesh_vert = np.ascontiguousarray(m.mesh_vert, dtype=np.float32).ravel() if m.nmesh else np.zeros(0, np.float32)
    mesh_vertadr = i32(m.mesh_vertadr) if m.nmesh else np.zeros(0, np.int32)
    mesh_vertnum = i32(m.mesh_vertnum) if m.nmesh else np.zeros(0, np.int32)
    mesh_graphadr = i32(m.mesh_graphadr) if m.nmesh else np.zeros(0, np.int32)
    mesh_graph = i32(m.mesh_graph) if m.nmesh else np.zeros(0, np.int32)
    # Body trees occupy the first sum(body_bvhnum) BVH nodes (mesh trees follow).
    body_nodes = int(np.sum(m.body_bvhnum))
    return {
        'body_geomadr': i32(m.body_geomadr), 'body_geomnum': i32(m.body_geomnum),
        'body_weldid': i32(m.body_weldid), 'body_parentid': i32(m.body_parentid),
        'body_bvhadr': i32(m.body_bvhadr), 'body_bvhnum': i32(m.body_bvhnum),
        'geom_bodyid': i32(m.geom_bodyid), 'geom_type': i32(m.geom_type),
        'geom_contype': i32(m.geom_contype), 'geom_conaffinity': i32(m.geom_conaffinity),
        'geom_size': f64(m.geom_size.reshape(ngeom, 3)),
        'geom_dataid': i32(m.geom_dataid),
        'geom_rbound': f64(m.geom_rbound), 'geom_margin': f64(m.geom_margin),
        'geom_gap': f64(m.geom_gap), 'geom_priority': i32(m.geom_priority),
        'geom_condim': i32(m.geom_condim),
        'geom_friction': f64(m.geom_friction.reshape(ngeom, 3)),
        'geom_solmix': f64(m.geom_solmix),
        'geom_solref': f64(m.geom_solref.reshape(ngeom, 2)),
        'geom_solimp': f64(m.geom_solimp.reshape(ngeom, 5)),
        'geom_aabb': f64(m.geom_aabb.reshape(ngeom, 6)),
        'bvh_aabb': f64(m.bvh_aabb.reshape(-1, 6)[:body_nodes]),
        'bvh_child': i32(m.bvh_child.reshape(-1, 2)[:body_nodes]),
        'bvh_geomid': i32(m.bvh_geomid[:body_nodes]),
        'pair_geom1': i32(m.pair_geom1) if npair else np.zeros(0, np.int32),
        'pair_geom2': i32(m.pair_geom2) if npair else np.zeros(0, np.int32),
        'pair_dim': i32(m.pair_dim) if npair else np.zeros(0, np.int32),
        'pair_margin': f64(m.pair_margin) if npair else np.zeros(0, np.float64),
        'pair_gap': f64(m.pair_gap) if npair else np.zeros(0, np.float64),
        'pair_friction': f64(m.pair_friction.reshape(npair, 5)) if npair else np.zeros((0, 5), np.float64),
        'pair_solref': f64(m.pair_solref.reshape(npair, 2)) if npair else np.zeros((0, 2), np.float64),
        'pair_solreffriction': f64(m.pair_solreffriction.reshape(npair, 2)) if npair else np.zeros((0, 2), np.float64),
        'pair_solimp': f64(m.pair_solimp.reshape(npair, 5)) if npair else np.zeros((0, 5), np.float64),
        'pair_signature': i32(m.pair_signature) if npair else np.zeros(0, np.int32),
        'exclude_signature': i32(m.exclude_signature) if m.nexclude else np.zeros(0, np.int32),
        'override': i32([int(bool(m.opt.enableflags & 1))]),
        'o_margin': f64([m.opt.o_margin]),
        'o_solref': f64(m.opt.o_solref),
        'o_solimp': f64(m.opt.o_solimp),
        'dispatch_present': present,
        'mesh_vert': mesh_vert, 'mesh_vertadr': mesh_vertadr,
        'mesh_vertnum': mesh_vertnum, 'mesh_graphadr': mesh_graphadr,
        'mesh_graph': mesh_graph,
        'filterparent_disabled': int(bool(m.opt.disableflags & 512)),
        'midphase_disabled': int(bool(m.opt.disableflags & 8192)),
        'contact_disabled': int(bool(m.opt.disableflags & 16)),
        'constraint_disabled': int(bool(m.opt.disableflags & 1)),
        'nconmax': int(m.nconmax) if hasattr(m, 'nconmax') else 5000,
        'collision_mode': int(m.opt.collision),
        'mpr_iterations': int(m.opt.mpr_iterations),
        'mpr_tolerance': float(m.opt.mpr_tolerance),
        'multiccd': int(bool(m.opt.enableflags & 16)),
    }
