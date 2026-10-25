"""Offline direct CUDA builds and load-only reuse of their ordinary artifacts."""
import os
from pathlib import Path
import subprocess
from .build_receipts import canonical, digest, file_record, compiler_dependencies, toolchain_record
from .product_cache import current_product_cache, compiler_environment_record


def receipted_artifact_build(*, commands, units, outputs, architecture,
        source_closure=None, source_aliases=None, verify, cwd=None, env=None, logs=None, required_symbols=()):
    """Execute or reuse a complete build under an explicit product scope.

    Source closure maps actual source paths to their already generated SHA256.
    Output paths never enter the cache identity. Every return has fresh private
    artifact files and has passed ``verify`` on those actual files; its caller
    keeps all original owner/module/resource/ABI gates there.
    """
    cache=current_product_cache()
    if cache is None:
        raise RuntimeError('direct product build requires an explicit product cache scope')
    from .compilation import compilation_allowed
    offline = compilation_allowed()
    commands=[list(map(str,command)) for command in commands]
    for command in commands:
        if any(arg in ('--use_fast_math','-use_fast_math','--fmad=false','--ftz=true',
                '--prec-div=false','--prec-sqrt=false') for arg in command):
            raise ValueError('direct native product violates FP64 compiler policy')
        for flag in ('--fmad=true','--ftz=false','--prec-div=true','--prec-sqrt=true'):
            if flag not in command:command.append(flag)
    units=[Path(p).resolve() for p in units]
    outputs={name:Path(p).resolve() for name,p in outputs.items()}
    expected_sources=[p.read_bytes() for p in units]
    closure=dict(source_closure or {})
    source_aliases=dict(source_aliases or {})
    if set(source_aliases)-set(closure):
        raise ValueError('logical source aliases must name current actual closure files')
    aliases={str(p):'@unit:'+str(i) for i,p in enumerate(units)}
    for command in commands:
        if '-o' in command:
            aliases[command[command.index('-o')+1]]='@output:'+str(len(aliases))
    # Output aliases must be stable by command position, not absolute owner path.
    normalized=[[aliases.get(arg,arg) for arg in command] for command in commands]
    compiler=commands[0][0]
    if any(command[0]!=compiler for command in commands):
        raise ValueError('direct native product requires one selected compiler')
    per_unit_flags=[]
    for unit in units:
        command=next((c for c in commands if str(unit) in c),None)
        if command is None:raise ValueError('compiler command missing actual source unit')
        flags=[];index=1
        while index<len(command):
            arg=command[index];index+=1
            if arg=='-o':index+=1;continue
            if arg in aliases or arg in ('-shared','-dc','-dlto','--ptx','-c'):continue
            if arg.endswith(('.o','.a')):continue
            flags.append(arg)
        per_unit_flags.append(flags)
    selection = {'schema':'maintained-direct-build-inputs-v1',
        'units':[{'bytes':len(raw),'sha256':digest(raw)} for raw in expected_sources],
        'closure':{source_aliases.get(k,k):v for k,v in closure.items()},
        'commands':normalized,'architecture':architecture,
        'policy':{'fma':True,'precise_div':True,'precise_sqrt':True,'ftz':False}}
    def current_inputs():
        if any(p.read_bytes()!=raw for p,raw in zip(units,expected_sources)):
            raise RuntimeError('generated CUDA source changed during product admission')
        for spelling,expected in closure.items():
            if digest(Path(spelling).read_bytes())!=expected:
                raise RuntimeError('project CUDA source closure changed during product admission')
        dependencies={}
        for unit,flags in zip(units,per_unit_flags):
            for row in compiler_dependencies(compiler,flags,[unit]):dependencies[row['path']]=row
        archives=sorted({arg for command in commands for arg in command if arg.endswith('.a') and Path(arg).is_file()})
        toolchain=toolchain_record(compiler,per_unit_flags[0])
        toolchain=dict(toolchain);toolchain['environment_sha256']=compiler_environment_record()['sha256']
        toolchain['environment_schema']='compiler-input-environment-v1'
        return {**selection,
            'dependencies':sorted([dict(row,path=source_aliases.get(row['path'],row['path']))
                for row in dependencies.values()],key=lambda row:row['path']),
            'archives':[file_record(p) for p in archives], 'toolchain':toolchain,
            'policy':{'fma':True,'precise_div':True,'precise_sqrt':True,'ftz':False}}
    # Child compilers must see exactly the bound compiler-input environment.
    # Callers' cache/tmp/worker variables may differ; compiler variables may not.
    if offline and env is not None:
        current_environment=compiler_environment_record()
        child_environment=compiler_environment_record(env)
        if child_environment!=current_environment:
            raise ValueError('direct builder child compiler environment differs from admitted inputs')
    identity=current_inputs() if offline else selection
    def validate():
        if current_inputs()!=identity:
            raise RuntimeError('CUDA source/dependency/toolchain/archive changed during product admission')
    def build():
        from .compilation import require_compilation
        require_compilation('receipted CUDA product', tuple(outputs))
        for index,command in enumerate(commands):
            result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True)
            if logs is not None:logs(index,result)
            if result.returncode:raise RuntimeError('native product compilation failed: '+(str(cwd) if cwd else str(outputs)))
        return {name:p.read_bytes() for name,p in outputs.items()}
    def materialize(artifacts):
        for name,raw in artifacts.items():
            path=outputs[name];path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
        verify()
    _,receipt,status=cache.obtain_artifacts(identity,required_symbols,names=outputs,
        build=build,validate=validate,exports=materialize)
    return {'receipt':receipt,'cache':status,'compile_commands_executed':[] if status['hit'] else commands}
