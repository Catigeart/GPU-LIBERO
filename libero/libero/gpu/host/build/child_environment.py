"""Private cwd/config/cache environment for every new child process."""
import hashlib,os,shutil
from pathlib import Path
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def child_env(cwd):
    """Give each compiler/scientific child its own actual cwd, PWD and caches."""
    cwd = Path(cwd).resolve()
    env = os.environ.copy()
    base = os.environ.get('P06C_CHILD_ENV_ROOT')
    private = (Path(base) / hashlib.sha256(str(cwd).encode()).hexdigest()[:16]
               if base else cwd / 'child-environment')
    for key, name in (
        ('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),
        ('XDG_CACHE_HOME','cache'),('XDG_CONFIG_HOME','config'),
        ('XDG_DATA_HOME','data'),('XDG_STATE_HOME','state'),
        ('XDG_RUNTIME_DIR','runtime'),('CUDA_CACHE_PATH','cuda'),
        ('WP_CACHE_DIR','warp'),('WARP_CACHE_PATH','warp'),('NUMBA_CACHE_DIR','numba'),
        ('HOME','home'),('MPLCONFIGDIR','mpl'),
    ):
        p = private / name
        p.mkdir(parents=True, exist_ok=True)
        if key == 'XDG_RUNTIME_DIR':
            p.chmod(0o700)
        env[key] = str(p)
    config_path = env.get('LIBERO_CONFIG_PATH')
    if not config_path:
        from libero.libero import libero_config_path
        config_path = libero_config_path
    source = Path(config_path) / 'config.yaml'
    config = private / 'libero'
    config.mkdir(exist_ok=True)
    dest = config / 'config.yaml'
    if not dest.exists():
        shutil.copyfile(source, dest)
    if sha(source) != sha(dest):
        raise ValueError('child private config drift')
    env.update(PWD=str(cwd), LIBERO_CONFIG_PATH=str(config),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    return env

