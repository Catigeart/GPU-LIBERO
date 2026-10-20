"""Byte-pinned Python source and loaded code witness, without science imports.

Preparation identity only. It does not turn source guards/AST into device or
mathematical evidence. A command must still bind actual fields and be executed.
"""
import libero.gpu.precision as scalar
from pathlib import Path
from functools import lru_cache
from ..core.cold_credentials import FileDigestLeasePool
import hashlib,json,types,ast,dis,struct,copy,os

def _code_constant(value):
    """Content types of this pinned interpreter, including exact float bits."""
    kind=type(value)
    if kind is types.CodeType:return ["code",code_content(value)]
    if value is None:return ["None"]
    if value is Ellipsis:return ["Ellipsis"]
    if kind is bool:return ["bool",value]
    if kind is int:return ["int",str(value)]
    if kind is float:return ["float64-bits",struct.pack("!d",value).hex()]
    if kind is complex:return ["complex128-bits",struct.pack("!dd",value.real,value.imag).hex()]
    if kind is str:return ["str",value]
    if kind is bytes:return ["bytes",value.hex()]
    if kind is tuple:return ["tuple",[_code_constant(item) for item in value]]
    if kind is frozenset:
        # Its member order is not Python set semantics and differs with -I
        # process hash randomization. Preserve every member's type and bits.
        return ["frozenset",sorted((_code_constant(item) for item in value),key=_canonical_json)]
    raise ValueError("unsupported original compiled constant type: "+str(kind))

def _canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)

def _constant_identity_topology(code):
    """Preserve observable internal const aliasing without process addresses."""
    groups={}
    def walk(value,path):
        groups.setdefault(id(value),[]).append(path)
        if type(value) is types.CodeType:
            for index,item in enumerate(value.co_consts):walk(item,path+"/codeconst/"+str(index))
        elif type(value) is tuple:
            for index,item in enumerate(value):walk(item,path+"/tuple/"+str(index))
        elif type(value) is frozenset:
            for index,item in enumerate(sorted(value,key=lambda item:_canonical_json(_code_constant(item)))):
                walk(item,path+"/setmember/"+str(index))
    for index,item in enumerate(code.co_consts):walk(item,"const/"+str(index))
    return sorted(sorted(paths) for paths in groups.values() if len(paths)>1)

def code_content(code):
    """All content-valued co_* fields of the actual pinned CodeType version."""
    if type(code) is not types.CodeType:raise ValueError("actual Python code required")
    fields={}
    for key in dir(code):
        if key.startswith("co_") and not callable(getattr(code,key)):
            fields[key]=["normalized-file","frozen-source"] if key=="co_filename" else _code_constant(getattr(code,key))
    return {"schema":"complete-typed-code-content-v1","fields":fields,
            "internal_constant_identity_equivalence_paths":_constant_identity_topology(code)}

def fingerprint(code):
    if type(code) is not types.CodeType:raise ValueError('actual Python code required')
    # CodeType equality omits some observable co_* fields. Identity plus the
    # strongly retained immutable code object prevents equality/id-reuse bugs.
    return _fingerprint_code_identity(id(code),code)

@lru_cache(maxsize=4096)
def _fingerprint_code_identity(identity,code):
    # Marshal includes ref-count/reference flags outside CodeType contents.
    # These are storage encodings, not Python code semantics. Keep opcodes,
    # argument flags, line/exception tables and all nested const values/aliases.
    # FrozenSources.code separately verifies the actual filename/source blob,
    # while live command/function witnesses retain in-process object identities.
    return hashlib.sha256(_canonical_json(code_content(code)).encode("ascii")).hexdigest()

def codes(code):
    yield code
    for value in code.co_consts:
        if isinstance(value,types.CodeType):yield from codes(value)

class FrozenSources:
    def __init__(self,records):
        if not isinstance(records,(list,tuple)) or not records:raise ValueError('explicit complete source set required')
        self.inventory={};self.files={};self.tokens={};self.syntax={};self.literal_globals={}
        self._source_leases=FileDigestLeasePool()
        self._source_facts={}
        for record in records:
            p=Path(record['path']).resolve();key=str(p)
            if key in self.inventory or type(record['bytes']) is not int or record['bytes']<0:raise ValueError('duplicate/invalid source record')
            raw=p.read_bytes()
            if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('source pin mismatch: '+key)
            self.inventory[key]={'path':key,'sha256':record['sha256'],'bytes':record['bytes']}
            self._source_facts[key]=(record['sha256'],len(raw))
            if self._source_leases.digest(key,record['sha256'])!=record['sha256']:
                raise ValueError('source pin mismatch: '+key)
            if p.suffix!='.py':continue
            tree=ast.parse(raw,filename=key);compiled=compile(raw,key,'exec',dont_inherit=True)
            self.files[key]={'path':key,'sha256':record['sha256'],'bytes':record['bytes']}
            self.tokens[key]={fingerprint(c) for c in codes(compiled)};self.syntax[key]=tree
            literals={}
            for node in tree.body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                    try:value=ast.literal_eval(node.value)
                    except (ValueError,TypeError):continue
                    if type(value) in (int,float,bool,str,bytes,tuple,type(None)):literals[node.targets[0].id]=value
            self.literal_globals[key]=literals
        self._source_facts=types.MappingProxyType(dict(self._source_facts))
        if not self.files:raise ValueError('Python source set empty')
    def validate(self):
        with self._source_leases.verification():
            for key,row in self.inventory.items():
                cold=self._source_facts.get(key)
                # Preserve public mutable inventory semantics. A changed SHA or
                # new path does not inherit the original cold length fact.
                with self._source_leases._lock:
                    lease=self._source_leases._leases.get(os.path.abspath(key))
                    if cold is not None and cold[0]==row['sha256'] and lease is not None:
                        actual=self._source_leases.digest(key,row['sha256'])
                        if os.path.abspath(key) in self._source_leases._leases:
                            if row['bytes']!=cold[1] or actual!=row['sha256']:
                                raise ValueError('retained source changed: '+key)
                            continue
                # Unsupported/broken seals execute the original one-read
                # bytecount + digest check; no cold length/metadata substitute.
                raw=Path(key).read_bytes()
                if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                    raise ValueError('retained source changed: '+key)
    def invalidate_source_leases(self,reason='explicit source update',*,strict=False):
        return self._source_leases.invalidate(reason,disable=strict)
    def code(self,code):
        if not isinstance(code,types.CodeType):raise ValueError('actual loaded Python code required')
        key=str(Path(code.co_filename).resolve())
        if any(str(Path(child.co_filename).resolve())!=key for child in codes(code)):
            raise ValueError('nested code filename differs from original source file')
        digest=fingerprint(code)
        if key not in self.tokens or digest not in self.tokens[key]:raise ValueError('loaded code differs from frozen source: '+key+':'+code.co_name)
        return dict(self.files[key],function=code.co_name,first_line=code.co_firstlineno,
                    loaded_code_sha256=digest,loaded_code_identity=id(code),
                    code_fingerprint_schema='complete-typed-code-content-v1',
                    code_identity_scope='current process retained code object; never cross-process address comparison')
    def function(self,function):
        value=getattr(function,'__func__',function)
        row=self.code(getattr(value,'__code__',None))
        used={i.argval for code in codes(value.__code__) for i in dis.get_instructions(code) if i.opname in ('LOAD_GLOBAL','LOAD_NAME')}
        def exact(a,b):
            if type(a) is not type(b):return False
            if isinstance(a,float):return struct.pack('!d',a)==struct.pack('!d',b)
            if isinstance(a,tuple):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
            if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
            return a==b
        matches=[n for n in ast.walk(self.syntax[row['path']]) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==value.__code__.co_name and min([n.lineno]+[d.lineno for d in n.decorator_list])==value.__code__.co_firstlineno]
        if len(matches)!=1:raise ValueError('actual function has no unique original defaults/closure declaration')
        original=matches[0]
        # Python function names/qualnames are mutable. Canonical writer/entry
        # identity is the unique original AST declaration and lexical parents.
        parents={child:parent for parent in ast.walk(self.syntax[row['path']]) for child in ast.iter_child_nodes(parent)}
        chain=[original.name];cursor=original
        while cursor in parents:
            cursor=parents[cursor]
            if isinstance(cursor,(ast.FunctionDef,ast.AsyncFunctionDef)):chain[:0]=[cursor.name,'<locals>']
            elif isinstance(cursor,ast.ClassDef):chain.insert(0,cursor.name)
        row['qualname']='.'.join(chain)
        row['original_name']=original.name
        row['original_declaration_line']=original.lineno
        row['original_first_line']=min([original.lineno]+[item.lineno for item in original.decorator_list])
        if value.__code__.co_freevars or value.__closure__ is not None:raise ValueError('captured closure source values require a separate original binding; unsupported by this witness')
        try:
            expected_defaults=tuple(ast.literal_eval(n) for n in original.args.defaults)
            expected_kwdefaults={arg.arg:ast.literal_eval(default) for arg,default in zip(original.args.kwonlyargs,original.args.kw_defaults) if default is not None}
        except (ValueError,TypeError) as e:raise ValueError('nonliteral original defaults require an independent typed source binding') from e
        if not exact(value.__defaults__ or (),expected_defaults) or not exact(value.__kwdefaults__ or {},expected_kwdefaults):raise ValueError('actual loaded defaults differ from original source')
        row['literal_defaults']=repr(expected_defaults);row['literal_kwdefaults']=repr(expected_kwdefaults);row['closure_binding']='none; captured closures rejected'
        row['literal_global_constants']={}
        for name,expected in self.literal_globals[row['path']].items():
            if name in used:
                if name not in value.__globals__ or not exact(value.__globals__[name],expected):raise ValueError('actual loaded original global constant differs: '+name)
                row['literal_global_constants'][name]=repr(expected)
        return row
    def stack(self,frame,entry_code):
        rows=[]
        while frame is not None:
            row=self.code(frame.f_code);row['actual_line']=frame.f_lineno;rows.append(row)
            if frame.f_code is entry_code:return rows
            frame=frame.f_back
        raise ValueError('actual operation did not originate in frozen entry')
    def access(self,function):
        row=self.function(function);code=getattr(function,'__func__',function).__code__;tree=self.syntax[row['path']]
        matches=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==code.co_name and (min([n.lineno]+[d.lineno for d in n.decorator_list])==code.co_firstlineno)]
        if len(matches)!=1:raise ValueError('loaded function has no unique frozen AST')
        fn=matches[0];parameters=[a.arg for a in fn.args.posonlyargs+fn.args.args];access=[];calls=[]
        parents={child:parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
        for n in ast.walk(fn):
            if isinstance(n,ast.Subscript):
                base=n.value
                while isinstance(base,ast.Subscript):base=base.value
                field=ast.unparse(base);root=field.split('.')[0]
                if root not in parameters:continue
                guard=[];child=n
                while child in parents:
                    parent=parents[child]
                    if isinstance(parent,ast.If) and child is not parent.test:guard.append({'condition':ast.unparse(parent.test),'branch':'else' if child in parent.orelse else 'then'})
                    elif isinstance(parent,(ast.For,ast.While)):guard.append({'loop':ast.unparse(parent) .split(':',1)[0]})
                    child=parent
                context=type(n.ctx).__name__
                if isinstance(parents.get(n),ast.AugAssign):context='ReadWrite'
                access.append({'argument':field,'index':ast.unparse(n.slice),'context':context,'line':n.lineno,'guards':list(reversed(guard))})
            elif isinstance(n,ast.Call):calls.append({'callee':ast.unparse(n.func),'arguments':[ast.unparse(a) for a in n.args],'line':n.lineno})
        return {'source':row,'parameters':parameters,'direct_access':access,'calls':calls,'transitive_helper_write_proof':'not inferred; helper/source binding required before complete write-set acceptance'}

    def syntax_function(self,function):
        value=getattr(function,'__func__',function)
        value=getattr(value,'func',value)
        row=self.function(value);code=value.__code__;tree=self.syntax[row['path']]
        matches=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==code.co_name and min([n.lineno]+[d.lineno for d in n.decorator_list])==code.co_firstlineno]
        if len(matches)!=1:raise ValueError('loaded function has no unique source AST')
        return value,row,matches[0]

    def transitive_access(self,function,*,require_warp_lowering=False):
        """Conservative field census, with authentic loaded helper/formal edges.

        Includes every original branch, loop and helper effect. Lexical guards
        are navigation, NOT a proved runtime predicate or interval certificate.
        Original AST/control flow is retained; dynamic values are never read.
        """
        top,row,syntax=self.syntax_function(function)
        parameters=[a.arg for a in syntax.args.posonlyargs+syntax.args.args]
        effects=[];edges=[];sources={};active=[]
        def path(node,aliases):
            if isinstance(node,ast.Name):return aliases.get(node.id)
            if isinstance(node,ast.Attribute):
                base=path(node.value,aliases)
                return None if base is None else base+'.'+node.attr
            return None
        def resolve(node,globals_):
            if isinstance(node,ast.Name):return globals_.get(node.id)
            if isinstance(node,ast.Attribute):
                base=resolve(node.value,globals_)
                return getattr(base,node.attr,None) if base is not None else None
            return None
        def expected_callee(expression,parent_binding):
            # Resolve source declarations/imports, not whichever current globals
            # happen to contain a function with a genuine pinned filename.
            tree=self.syntax[parent_binding['path']]
            if isinstance(expression,ast.Name):
                for item in tree.body:
                    if isinstance(item,(ast.FunctionDef,ast.ClassDef)) and item.name==expression.id:return parent_binding['path'],expression.id
                    if isinstance(item,ast.ImportFrom):
                        for alias in item.names:
                            if (alias.asname or alias.name)==expression.id:
                                if item.level:
                                    parent=Path(parent_binding['path']).parent
                                    for _ in range(item.level-1):parent=parent.parent
                                    target=parent.joinpath(*(item.module or '').split('.')).with_suffix('.py')
                                    return str(target.resolve()),alias.name
                                suffix='/'+item.module.replace('.','/')+'.py'
                                targets=[k for k in self.files if k.endswith(suffix)]
                                if len(targets)==1:return targets[0],alias.name
            raise ValueError('custom loaded helper has no independent original declaration/import edge: '+ast.unparse(expression))
        def walk(actual,origins,trail):
            fn,binding,node=self.syntax_function(actual);token=binding['path']+':'+binding['qualname']
            if token in active:raise ValueError('recursive helper needs explicit source proof: '+token)
            active.append(token);sources[token]=dict(binding,original_ast=ast.unparse(node))
            names=[a.arg for a in node.args.posonlyargs+node.args.args]
            if len(names)!=len(origins):raise ValueError('helper formal/actual count differs')
            aliases=dict(zip(names,origins));body=ast.Module(body=node.body,type_ignores=[])
            parents={child:parent for parent in ast.walk(body) for child in ast.iter_child_nodes(parent)}
            # Only unambiguous bare/attribute array aliases can change field
            # identity. Arithmetic/indexing results are scalar values. Keep
            # a full branch census; reject an alias that has different roots.
            for item in ast.walk(body):
                if isinstance(item,(ast.Assign,ast.AnnAssign)):
                    targets=item.targets if isinstance(item,ast.Assign) else [item.target]
                    origin=path(item.value,aliases)
                    if origin is not None:
                        for target in targets:
                            if isinstance(target,ast.Name):
                                if target.id in aliases and aliases[target.id] not in (None,origin):raise ValueError('ambiguous helper array alias: '+target.id)
                                aliases[target.id]=origin
            def reference_value(item):
                # Device array references may be forwarded only by the exact
                # Name/Attribute aliases and explicit authenticated helper edges
                # represented above. Array elements are values, not references.
                if item is None:return False
                if path(item,aliases) is not None:return True
                if isinstance(item,(ast.Tuple,ast.List,ast.Set)):return any(reference_value(value) for value in item.elts)
                if isinstance(item,ast.Dict):return any(reference_value(value) for value in item.values)
                if isinstance(item,ast.IfExp):return reference_value(item.body) or reference_value(item.orelse)
                if isinstance(item,(ast.BinOp,ast.BoolOp,ast.UnaryOp)):return any(reference_value(value) for value in ast.iter_child_nodes(item))
                return False
            for item in ast.walk(body):
                if isinstance(item,(ast.Assign,ast.AnnAssign)):
                    targets=item.targets if isinstance(item,ast.Assign) else [item.target]
                    if reference_value(item.value) and (path(item.value,aliases) is None or any(not isinstance(target,ast.Name) for target in targets)):
                        raise ValueError('unsupported device reference propagation/container/destructuring: '+binding['path']+':'+str(item.lineno))
                elif isinstance(item,ast.Return) and reference_value(item.value):
                    raise ValueError('device reference return needs explicit source ABI binding: '+binding['path']+':'+str(item.lineno))
                elif isinstance(item,ast.For) and reference_value(item.iter):
                    raise ValueError('device reference iteration needs explicit source ABI binding: '+binding['path']+':'+str(item.lineno))
            def guards(item):
                out=[];child=item
                while child in parents:
                    parent=parents[child]
                    if isinstance(parent,ast.If) and child is not parent.test:out.append({'condition':ast.unparse(parent.test),'branch':'else' if child in parent.orelse else 'then'})
                    elif isinstance(parent,(ast.For,ast.While)):out.append({'loop':ast.unparse(parent).split(':',1)[0]})
                    child=parent
                return list(reversed(out))
            for item in ast.walk(body):
                if isinstance(item,ast.Subscript):
                    base=item.value
                    while isinstance(base,ast.Subscript):base=base.value
                    origin=path(base,aliases)
                    if origin is None:continue
                    context=type(item.ctx).__name__
                    if isinstance(parents.get(item),ast.AugAssign):context='ReadWrite'
                    effects.append({'root_argument':origin,'source':binding,'line':item.lineno,'expression':ast.unparse(item),'context':context,'lexical_guards':guards(item),'helper_edges':trail})
                elif isinstance(item,ast.Call):
                    target=resolve(item.func,fn.__globals__);loaded=getattr(target,'func',getattr(target,'__func__',target))
                    # Arguments passed by array reference have no Subscript.
                    actual_origins=[path(arg,aliases) for arg in item.args]
                    if isinstance(getattr(loaded,'__code__',None),types.CodeType):
                        childfn,childrow,childnode=self.syntax_function(loaded)
                        expected_path,expected_name=expected_callee(item.func,binding)
                        if childrow['path']!=expected_path or childnode.name!=expected_name:raise ValueError('actual loaded callee differs from original declaration/import')
                        lowering=self.warp_function(target) if getattr(target,'adj',None) is not None else None
                        if require_warp_lowering and lowering is None:raise ValueError('actual custom helper has no authenticated Warp lowering')
                        childnames=[a.arg for a in childnode.args.posonlyargs+childnode.args.args]
                        if item.keywords or len(childnames)!=len(item.args):raise ValueError('helper keywords/defaults need explicit source binding')
                        edge={'caller':binding,'line':item.lineno,'callee':childrow,'arguments':[ast.unparse(a) for a in item.args],'formal_origins':dict(zip(childnames,actual_origins)),'lexical_guards':guards(item),'helper_lowering':lowering,'original_expected_callee':{'path':expected_path,'name':expected_name}}
                        edges.append(edge);walk(loaded,actual_origins,trail+[edge])
                    elif any(x is not None for x in actual_origins):
                        name=ast.unparse(item.func)
                        if name in ('wp.atomic_add','wp.atomic_sub','wp.atomic_min','wp.atomic_max','wp.atomic_exch','wp.atomic_cas') and actual_origins[0] is not None:
                            effects.append({'root_argument':actual_origins[0],'source':binding,'line':item.lineno,'expression':ast.unparse(item),'context':'ReadWrite','lexical_guards':guards(item),'helper_edges':trail,'source_builtin_atomic':name})
                        elif name not in ('len','wp.len'):
                            raise ValueError('unresolved reference-taking helper: '+binding['path']+':'+str(item.lineno)+':'+name)
            active.pop()
        primitive={'int','float','bool','str','bytes','wp.int32','wp.int64','wp.float32','scalar.wp_real'}
        origins=[None if a.annotation is not None and ast.unparse(a.annotation) in primitive else a.arg for a in syntax.args.posonlyargs+syntax.args.args]
        walk(top,origins,[])
        return {'schema':'authenticated-transitive-source-field-census-v1','root_source':row,'parameters':parameters,'effects':effects,'helper_edges':edges,'sources':list(sources.values()),'source_completeness':'all original function bodies and authentic loaded helper calls traversed; unresolved reference-taking helpers rejected','scope':'conservative symbolic source census; lexical guards omit dominance/return constraints and are not exact runtime read/write or epoch proof','device_execution':False,'full_F':False}

    def warp_function(self,object_):
        """Bind actual Warp lowering AST and entry function to pinned source.

        This rejects altered adj/source/transformers/argument labels/constants.
        Full generated module/PTX dependencies still need compile-chain evidence.
        """
        fn,row,node=self.syntax_function(object_)
        adj=getattr(object_,'adj',None)
        if adj is None or adj.func is not fn or getattr(adj,'transformers',None):raise ValueError('actual original Warp lowering/function/transformer differs')
        def clean(n):
            n=copy.deepcopy(n);n.decorator_list=[];return ast.dump(n,include_attributes=False)
        actual=getattr(adj,'tree',None)
        source=ast.parse(adj.source)
        if not isinstance(actual,ast.Module) or not actual.body or not source.body or clean(actual.body[0])!=clean(node) or clean(source.body[0])!=clean(node):raise ValueError('actual Warp lowering AST/source differs from original function')
        labels=tuple(a.arg for a in node.args.posonlyargs+node.args.args)
        if tuple(a.label for a in adj.args)!=labels:raise ValueError('actual Warp formal argument labels differ from source')
        return {'source':row,'lowering_ast_sha256':hashlib.sha256(clean(node).encode()).hexdigest(),'argument_labels':labels,'compile_dependency_scope':'actual lowering and loaded source binding; independent compiled module/PTX/source-helper chain evidence separately required'}
