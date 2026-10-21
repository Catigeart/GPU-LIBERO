"""Strict cold build receipts and direct-address product cache. CPU-only API."""
import hashlib,json,os,shutil,subprocess,uuid,re
from pathlib import Path
from contextlib import contextmanager
import time
from functools import lru_cache

def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _file_signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@lru_cache(maxsize=8192)
def _file_digest(path, signature):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        if _file_signature(os.fstat(stream.fileno())) != signature:
            raise RuntimeError('build input changed before hashing: ' + str(path))
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
        if _file_signature(os.fstat(stream.fileno())) != signature:
            raise RuntimeError('build input changed while hashing: ' + str(path))
    return result.hexdigest()


def file_record(path):
    """Recheck file identity/metadata; hash bytes again after any file change.

    The bounded cache is process-local. Recently changed files always read
    bytes: filesystem timestamps can coalesce multiple writes in one tick.
    For older files, ctime catches edits with restored mtime; inode/device
    catch replacement. Dependency/tool discovery still runs on every admission.
    """
    path = Path(path).resolve()
    signature = _file_signature(path.stat())
    compute = _file_digest.__wrapped__ if time.time_ns() - signature[4] < 2_000_000_000 else _file_digest
    return {'path': str(path), 'bytes': signature[2], 'sha256': compute(str(path), signature)}

def _dependency_names(output):
    """Parse one fixed-target Make rule without evaluating Make expressions."""
    if type(output) is not str or '\0' in output:
        raise ValueError('invalid compiler dependency output')
    target,separator,body=output.partition(':')
    # CUDA 13.1 emits "CF_DEP_TARGET :"; older SDKs omit that space.
    # strip only the fixed target's surrounding whitespace, never its name.
    if not separator or '\n' in target or '\r' in target or target.strip()!='CF_DEP_TARGET':
        raise ValueError('compiler dependency target missing')
    words=[];word=[];index=0
    while index<len(body):
        c=body[index];index+=1
        if c=='\\':
            if index==len(body): raise ValueError('truncated compiler dependency escape')
            escaped=body[index];index+=1
            if escaped=='\r' and index<len(body) and body[index]=='\n':
                index+=1;escaped='\n'
            if escaped=='\n':
                # A Make continuation is whitespace, including with no indent.
                if word: words.append(''.join(word));word=[]
            elif escaped in (' ','\t','#','$','\\',':'):
                word.append(escaped)
            else:
                raise ValueError('unsupported compiler dependency escape')
        elif c=='$':
            if index==len(body) or body[index]!='$':
                raise ValueError('compiler dependency expression unsupported')
            word.append('$');index+=1
        elif c in ':#;|':
            raise ValueError('multiple rules or unsupported compiler dependency syntax')
        elif c in ' \t\r\n':
            if c in '\r\n' and body[index:].strip():
                raise ValueError('multiple compiler dependency rules')
            if word: words.append(''.join(word));word=[]
        elif ord(c)<32 or ord(c)==127:
            raise ValueError('invalid compiler dependency control character')
        else: word.append(c)
    if word: words.append(''.join(word))
    if not words: raise ValueError('compiler dependency list empty')
    return words

def compiler_dependencies(nvcc,flags,units):
    """Ask the actual compiler, including angle/vendor/macro includes.

    Never infer preprocessing from regex. Unresolved includes fail before a
    cache hit. -M output can escape spaces, #, $, backslashes and line breaks.
    """
    rows={}
    for unit in units:
        output=subprocess.check_output([nvcc,*flags,'-M','-MT','CF_DEP_TARGET',str(unit)],text=True)
        for name in _dependency_names(output):
            path=Path(name).resolve()
            if path!=Path(unit).resolve(): rows[str(path)]=file_record(path)
    return [rows[k] for k in sorted(rows)]

def toolchain_record(nvcc,flags=()):
    executable=shutil.which(str(nvcc)) or str(nvcc);path=Path(executable).resolve()
    version=subprocess.check_output([str(path),'--version'],text=True)
    # Pin all installed CUDA compiler components, plus host compiler/linker.
    # The environment digest binds PATH/CPATH/tool options without disclosing it.
    paths={path}
    host_compiler=None
    for index,flag in enumerate(flags):
        if flag.split('=',1)[0] in ('-ccbin','--compiler-bindir'):
            host_compiler=flag.split('=',1)[1] if '=' in flag else flags[index+1]
    if host_compiler:
        custom=Path(host_compiler).resolve()
        if custom.is_dir():paths.update(p.resolve() for p in custom.iterdir() if p.is_file())
        elif custom.is_file():paths.add(custom)
        else:raise ValueError('actual configured host compiler unavailable')
    for folder in (path.parent,path.parent.parent/'nvvm/bin',
                   path.parent.parent/'nvvm/libdevice',path.parent.parent/'nvvm/lib64'):
        if folder.is_dir(): paths.update(p.resolve() for p in folder.iterdir() if p.is_file())
    for name in ('gcc','g++','cc','c++','ld','as','collect2','ar'):
        found=shutil.which(name)
        if found: paths.add(Path(found).resolve())
    for name in ('g++','gcc'):
        found=shutil.which(name)
        if found:
            for helper in ('cc1','cc1plus','collect2','liblto_plugin.so'):
                target=subprocess.check_output([found,'-print-prog-name='+helper],text=True).strip()
                if Path(target).is_file(): paths.add(Path(target).resolve())
            for library in ('libstdc++.so','libgcc_s.so','libgcc.a','crtbegin.o','crtend.o'):
                target=subprocess.check_output([found,'-print-file-name='+library],text=True).strip()
                if Path(target).is_file():paths.add(Path(target).resolve())
    # CUDA implicit device/runtime linker inputs are outside -M's header set.
    # Include actual installed library files, including symlink destinations.
    for folder in (path.parent.parent/'lib64',path.parent.parent/'targets/x86_64-linux/lib'):
        if folder.is_dir():paths.update(p.resolve() for p in folder.iterdir() if p.is_file())
    # Pin loader/runtime dependencies of the actual compiler/linker programs.
    # Non-ELF archives/objects report no dynamic dependencies, which is expected.
    pending=list(paths);seen=set()
    while pending:
        binary=pending.pop()
        if binary in seen:continue
        seen.add(binary)
        if binary.suffix in ('.a','.o','.h','.txt','.profile'):continue
        result=subprocess.run(['ldd',str(binary)],capture_output=True,text=True)
        if result.returncode not in (0,1):raise ValueError('toolchain loader dependency inspection failed: '+str(binary))
        for token in re.findall(r'(?:=>\s+)?(/[^\s()]+)',result.stdout):
            target=Path(token)
            if target.is_file() and target.resolve() not in paths:
                paths.add(target.resolve());pending.append(target.resolve())
    return {'compiler':str(path),'version':version,'components':[file_record(p) for p in sorted(paths)],
            'environment_sha256':digest(canonical(dict(os.environ))),
            'schema':'strict-toolchain-components-env-v1'}

def build_identity(*,source,bridges,flags,toolchain,dependencies,archive,architecture):
    payload={'schema':'compiler-dependency-build-v1','source_sha256':digest(source.encode()),
             'bridges_sha256':[digest(x.encode()) for x in bridges],'flags':list(flags),
             'toolchain':toolchain,'dependencies':dependencies,'archive':archive,
             'architecture':architecture,'params_launcher_abi':'included-generated-source',
             'policy':{'fma':True,'precise_div':True,'precise_sqrt':True,'ftz':False}}
    return digest(canonical(payload)),payload

@contextmanager
def product_lock(folder,timeout=120):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);lock=folder/'build.lock'
    start=time.monotonic()
    while True:
        try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);break
        except FileExistsError:
            if time.monotonic()-start>timeout: raise TimeoutError('active/stale build lock: '+str(lock))
            time.sleep(.05)
    try: yield
    finally: os.close(fd);lock.unlink()

def read_product(folder,key):
    folder=Path(folder)
    try:
        receipt=json.loads((folder/'receipt.json').read_text());raw=(folder/'product.so').read_bytes()
    except (OSError,ValueError): return None
    try:
        if (type(receipt) is not dict or receipt.get('schema')!='complete-build-product-v1' or receipt.get('build_key')!=key or
                receipt.get('binary_sha256')!=digest(raw) or receipt.get('binary_bytes')!=len(raw) or
                receipt.get('identity_sha256')!=digest(canonical(receipt.get('identity'))) or
                digest(canonical(receipt.get('identity')))!=key): return None
    except (TypeError,ValueError):return None
    return raw,receipt

def publish_product(folder,key,identity,raw):
    folder=Path(folder);nonce=uuid.uuid4().hex
    receipt={'schema':'complete-build-product-v1','build_key':key,'identity':identity,
             'identity_sha256':digest(canonical(identity)),'binary_sha256':digest(raw),'binary_bytes':len(raw)}
    # Completion marker last. Readers must hold the build lock.
    blob=folder/(nonce+'.so');meta=folder/(nonce+'.json')
    blob.write_bytes(raw);meta.write_bytes(canonical(receipt))
    os.replace(blob,folder/'product.so');os.replace(meta,folder/'receipt.json')
    return receipt

def materialize_private(root,name,raw):
    """Never share a mutable .so path, hardlink, CDLL or owner across plans."""
    path=Path(root)/(name+'_'+uuid.uuid4().hex+'.so')
    with path.open('xb') as stream: stream.write(raw)
    return path
