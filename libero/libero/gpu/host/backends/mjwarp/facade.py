"""Local compatibility for robosuite 1.4 address queries on MuJoCo 3.16.

The original methods test a NumPy scalar's membership in a tuple of pybind
enums. New MuJoCo enums compare asymmetrically with NumPy integers, causing a
valid hinge to fail that test. Use the already compiled joint spans instead.
No installed package or global facade class is modified.
"""
from types import MethodType


def _address(model, name, array, spans):
    index = model.joint_name2id(name)
    begin, end = model._metadata[spans][name]
    address = getattr(model, array)[index]
    return address if end - begin == 1 else (address, address + end - begin)


def _qpos(model, name):
    return _address(model, name, 'jnt_qposadr', 'joint_qpos_slices')


def _qvel(model, name):
    return _address(model, name, 'jnt_dofadr', 'joint_qvel_slices')


def bind_model_compatibility(model):
    # Explicit instance methods belong to this backend's metadata view only.
    object.__setattr__(model, 'get_joint_qpos_addr', MethodType(_qpos, model))
    object.__setattr__(model, 'get_joint_qvel_addr', MethodType(_qvel, model))
