"""Exact binding-layout matching for an explicitly frozen prefix comparison.

This diagnostic never supplies a library to ordinary environment loading. It
matches the complete generated binding wrapper after undoing only the current
subtree constant and block-size selection. Mathematical headers were checked
when the reference catalogue was frozen.
"""
import ctypes as ct
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

class PrefixVersions:
    def __init__(self, runtime, folder):
        folder=Path(folder)
        rows=json.loads((folder/'manifest.json').read_text())['rows']
        seen=set();plans={}
        def walk(obj):
            if id(obj) in seen:return
            seen.add(id(obj))
            if isinstance(obj,dict):
                for value in obj.values():walk(value)
            elif isinstance(obj,(list,tuple)):
                for value in obj:walk(value)
            elif isinstance(obj,SimpleNamespace) or (type(obj).__module__.startswith('libero.gpu') and hasattr(obj,'__dict__')):
                plan=vars(obj).get('_cuda_position_plan')
                if plan is not None:plans[id(plan)]=plan
                for value in vars(obj).values():walk(value)
        walk(runtime)
        if not plans:raise ValueError('no native position plan found for frozen comparison')
        self.bindings=[];self.report=[]
        for plan in plans.values():
            source=Path(plan.manifest['source']).read_text()
            tail=source[source.rfind('namespace libero_position {'):]
            tail=re.sub(r'__device__ __constant__ int mass_subtree_sizes\[\d+\] = \{[0-9,]+\};\n','',tail)
            tail=tail.replace(', libero_position::mass_subtree_sizes','')
            original_threads=plan.manifest['original_threads_per_world']
            tail=tail.replace(f'<<<dim3(1,worlds,1),{plan.block_size},',
                              f'<<<dim3(1,worlds,1),{original_threads},')
            digest=hashlib.sha256(tail.encode()).hexdigest()
            matches=[row for row in rows if row['wrapper_sha256']==digest]
            if not matches:raise ValueError('frozen prefix binding wrapper not found: '+digest)
            # Identical wrappers may have multiple receipts. Their frozen math
            # and compiler inputs must select one binary, rather than guessing.
            if len({row['sha256'] for row in matches})!=1:
                raise ValueError('ambiguous frozen prefix binaries for one wrapper')
            row=matches[0];path=folder/row['library']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('frozen prefix binary was changed')
            if (row['parameter_abi']['parameter_bytes']!=ct.sizeof(plan.params) or
                    row['dynamic']!=plan.dynamic_shared_bytes):
                raise ValueError('frozen prefix ABI/shared arena differs')
            old=ct.CDLL(str(path.resolve()))
            old.launch.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p];old.launch.restype=ct.c_int
            old.configure_shared.argtypes=[ct.c_int];old.configure_shared.restype=ct.c_int
            if old.configure_shared(plan.dynamic_shared_bytes):raise RuntimeError('configure frozen prefix failed')
            self.bindings.append((plan,plan.library,old))
            candidate_path=Path(plan.manifest['library'])
            self.report.append(dict(baseline=str(path),baseline_sha256=row['sha256'],
                candidate=str(candidate_path),candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                baseline_threads=original_threads,candidate_threads=plan.block_size,
                binding_wrapper_sha256=digest))

    def activate(self, name):
        for plan,current,old in self.bindings:plan.library=old if name=='baseline' else current
