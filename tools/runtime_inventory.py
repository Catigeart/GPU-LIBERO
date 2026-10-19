"""Inspect GPU imports and cold source composition without importing it."""
import argparse
import ast
from importlib.util import resolve_name
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / 'libero/libero/gpu'
ENTRYPOINTS = ('libero.gpu.host.factory', 'libero.gpu.host.adapters.environment.batch_env',
               'libero.gpu.host.runtime.step.fused_substep')


def inventory():
    sources = {}
    for path in sorted(GPU.rglob('*.py')):
        parts = list(path.relative_to(GPU).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        sources['.'.join(['libero', 'gpu', *parts])] = path
    rows = []
    for module, path in sources.items():
        tree = ast.parse(path.read_text())
        package = module if path.name == '__init__.py' else module.rsplit('.', 1)[0]
        dependencies, scripts = set(), set()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                if base:
                    names = [base] + [base + '.' + alias.name for alias in node.names]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module' and isinstance(node.args[0].value, str):
                    names = [node.args[0].value]
            dependencies.update(name for name in names if name in sources)
            scripts.update(name for name in names if name == 'scripts' or name.startswith('scripts.'))
        calls = {node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        rows.append(dict(module=module, path=str(path.relative_to(ROOT)),
            entry_functions=[node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))],
            local_dependencies=sorted(dependencies), script_dependencies=sorted(scripts),
            cold_source_composition=bool({'exec', 'compile'} & calls)))
    by_module = {row['module']: row for row in rows}
    pending, reached = list(ENTRYPOINTS), set()
    while pending:
        module = pending.pop()
        if module in reached or module not in by_module:
            continue
        reached.add(module)
        pending.extend(by_module[module]['local_dependencies'])
    for row in rows:
        row['statically_reachable'] = row['module'] in reached
    return dict(scope='Static imports, not an execution trace or proof that unreached modules are unused',
                entrypoints=list(ENTRYPOINTS), module_count=len(rows), reachable_count=len(reached),
                script_dependency_count=sum(len(row['script_dependencies']) for row in rows),
                cold_source_composition=[row['module'] for row in rows if row['cold_source_composition']], modules=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'modules'}))
    return 1 if record['script_dependency_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
