"""根据学习项目中的 Python 导入，同步生成依赖清单。

从项目脚本路径识别项目，并将清单保存到 requirements/<项目名>.txt。
本文件也可以作为统一运行入口：先同步依赖，再启动指定脚本。
"""

import ast
import importlib.util
import os
import runpy
import sys
import sysconfig
from pathlib import Path

PACKAGE_NAME_OVERRIDES = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}
CONDITIONAL_REQUIREMENTS = {
    "torch.utils.tensorboard": ("tensorboard", "protobuf"),
}
SKIP_DIRS = {".git", ".venv", "__pycache__", "build", "dist", "env", "venv"}


def find_project(script_file):
    """返回 learning/ 或 projects/ 下脚本所属的仓库根目录和项目目录。"""
    script = Path(script_file).resolve()
    for parent in script.parents:
        if parent.name == "common":
            continue
        if parent.parent.name in {"learning", "projects"}:
            repo_root = parent.parent.parent
            if (repo_root / "requirements").is_dir():
                return repo_root, parent
    return None, None


def _python_files(project_root):
    return sorted(
        path for path in project_root.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.parts)
    )


def _imports(paths):
    roots, complete = set(), set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    complete.add(alias.name)
                    roots.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                complete.add(node.module)
                roots.add(node.module.split(".", 1)[0])
    return roots, complete


def _is_stdlib(module):
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    if spec is None:
        return False
    if spec.origin in ("built-in", "frozen", None):
        return True
    try:
        origin = Path(spec.origin).resolve()
        stdlib = Path(sysconfig.get_path("stdlib")).resolve()
        relative = origin.relative_to(stdlib)
    except (AttributeError, TypeError, ValueError):
        return False
    return not any(part in {"site-packages", "dist-packages"} for part in relative.parts)


def _distribution_for(module):
    if module in PACKAGE_NAME_OVERRIDES:
        return PACKAGE_NAME_OVERRIDES[module]
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata
    packages_distributions = getattr(metadata, "packages_distributions", None)
    if packages_distributions:
        matches = packages_distributions().get(module, [])
        if matches:
            return matches[0]
    for distribution in metadata.distributions():
        top_level = distribution.read_text("top_level.txt") or ""
        if module in top_level.splitlines():
            return distribution.metadata["Name"]
    try:
        return metadata.distribution(module).metadata["Name"]
    except metadata.PackageNotFoundError:
        return module


def _requirements_for(project_root):
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata
    paths = _python_files(project_root)
    roots, complete = _imports(paths)
    local_names = {path.stem for path in paths} | {"learning", "requirements"}
    distributions = set()
    for module in roots - local_names:
        if not _is_stdlib(module):
            distributions.add(_distribution_for(module))
    for prefix, required in CONDITIONAL_REQUIREMENTS.items():
        if any(item == prefix or item.startswith(prefix + ".") for item in complete):
            distributions.update(required)
    result = []
    for distribution in distributions:
        try:
            installed = metadata.distribution(distribution)
            name = installed.metadata.get("Name") or distribution
            result.append("{}=={}".format(name, installed.version))
        except metadata.PackageNotFoundError:
            result.append(distribution)
    return sorted(set(result), key=str.lower)


def sync_for_script(script_file):
    """创建或更新本项目的依赖清单；内容未变化时不重写文件。"""
    repo_root, project_root = find_project(script_file)
    if project_root is None:
        return None
    output = repo_root / "requirements" / (project_root.name + ".txt")
    version = ".".join(str(value) for value in sys.version_info[:3])
    lines = [
        "# Python 版本：{}".format(version),
        "# 由 requirements/sync_requirements.py 自动生成",
    ] + _requirements_for(project_root)
    content = "\n".join(lines) + "\n"
    if not output.exists() or output.read_text(encoding="utf-8") != content:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print("[依赖清单] 已更新 {}".format(output.relative_to(repo_root)))
    return output

def main(argv=None):
    """先同步依赖，再运行指定的项目脚本。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("用法：python requirements/sync_requirements.py <项目脚本.py> [脚本参数 ...]")
        return 0 if args else 2
    target = Path(args[0]).expanduser().resolve()
    if not target.is_file() or target.suffix != ".py":
        raise SystemExit("未找到 Python 文件：{}".format(target))
    repo_root, project_root = find_project(target)
    if project_root is None:
        raise SystemExit("脚本不在 learning/ 或 projects/ 项目目录中：{}".format(target))
    sync_for_script(target)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(target.parent) not in sys.path:
        sys.path.insert(0, str(target.parent))
    sys.argv = [str(target)] + args[1:]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
