#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
separate_assets.py

用途：
1. 扫描仓库中的权重/训练数据/大文件
2. 将这些文件移动或复制到仓库外的 assets 目录
3. 生成 manifest，方便读者按原路径放回
4. 可选地更新 .gitignore，避免再次提交大文件
5. 提供 verify 模式，检查读者是否已把权重放到正确位置

示例：
  扫描：
    python separate_assets.py scan --repo .

  分离（移动到仓库外）：
    python separate_assets.py separate --repo . --assets-dir ../LightStab_assets --move

  分离（只复制，保留仓库内原文件）：
    python separate_assets.py separate --repo . --assets-dir ../LightStab_assets --copy

  校验读者是否放对：
    python separate_assets.py verify --repo . --manifest weights_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXTENSIONS = {
    ".pth", ".pt", ".ckpt", ".bin", ".onnx", ".engine", ".trt",
    ".safetensors", ".h5", ".hdf5", ".pb", ".tflite",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".mypy_cache",
    ".pytest_cache", "node_modules", "build", "dist", ".venv", "venv"
}

DEFAULT_SIZE_MB = 50


@dataclass
class AssetRecord:
    rel_path: str
    size_bytes: int
    sha256: str
    reason: str
    asset_path: str   # 放在 assets-dir 下的相对路径，默认与原始 rel_path 一致


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_files(repo: Path, exclude_dirs: set[str]) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for name in files:
            yield root_path / name


def is_candidate(path: Path, repo: Path, min_size_mb: int, extensions: set[str]) -> tuple[bool, str]:
    ext = path.suffix.lower()
    size_bytes = path.stat().st_size
    size_limit = min_size_mb * 1024 * 1024
    rel = path.relative_to(repo).as_posix()

    if ext in extensions:
        return True, f"extension={ext}"
    if size_bytes >= size_limit:
        return True, f"size>={min_size_mb}MB"
    return False, ""


def scan_assets(repo: Path, min_size_mb: int, extensions: set[str], exclude_dirs: set[str]) -> List[AssetRecord]:
    records: List[AssetRecord] = []
    for path in iter_files(repo, exclude_dirs):
        ok, reason = is_candidate(path, repo, min_size_mb, extensions)
        if not ok:
            continue
        rel = path.relative_to(repo).as_posix()
        records.append(
            AssetRecord(
                rel_path=rel,
                size_bytes=path.stat().st_size,
                sha256=sha256sum(path),
                reason=reason,
                asset_path=rel,
            )
        )
    records.sort(key=lambda x: x.size_bytes, reverse=True)
    return records


def write_manifest(records: List[AssetRecord], repo: Path, manifest_path: Path) -> None:
    payload = {
        "repo_name": repo.name,
        "total_files": len(records),
        "total_size_bytes": sum(r.size_bytes for r in records),
        "files": [asdict(r) for r in records],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_gitignore(repo: Path, records: List[AssetRecord]) -> None:
    gitignore = repo / ".gitignore"
    existing = set()
    if gitignore.exists():
        existing = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}

    additions = []
    marker = "# Added by separate_assets.py"
    if marker not in existing:
        additions.append(marker)
    for r in records:
        if r.rel_path not in existing:
            additions.append(r.rel_path)

    if additions:
        with gitignore.open("a", encoding="utf-8") as f:
            if gitignore.stat().st_size > 0:
                f.write("\n")
            for line in additions:
                f.write(line + "\n")


def move_or_copy_assets(repo: Path, assets_dir: Path, records: List[AssetRecord], mode: str) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)

    for r in records:
        src = repo / r.rel_path
        dst = assets_dir / r.asset_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[skip] 源文件不存在: {src}")
            continue

        if mode == "move":
            shutil.move(str(src), str(dst))
            print(f"[move] {src} -> {dst}")
        elif mode == "copy":
            shutil.copy2(src, dst)
            print(f"[copy] {src} -> {dst}")
        else:
            raise ValueError(f"未知模式: {mode}")


def write_setup_readme(repo: Path, manifest_path: Path, output_path: Path, records: List[AssetRecord]) -> None:
    lines = []
    lines.append(f"# {repo.name} 权重与大文件放置说明")
    lines.append("")
    lines.append("本项目的 Git 仓库只保留代码，不直接包含模型权重、训练数据和其他大文件。")
    lines.append("请先从作者提供的网盘下载这些文件，然后按下面的路径放回项目目录。")
    lines.append("")
    lines.append("## 读者使用步骤")
    lines.append("")
    lines.append("1. 克隆代码仓库。")
    lines.append("2. 从网盘下载权重压缩包或单独文件。")
    lines.append("3. 将下载得到的文件放到下面列出的**目标路径**。")
    lines.append("4. 放好后，运行校验命令：")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python separate_assets.py verify --repo . --manifest {manifest_path.name}")
    lines.append("```")
    lines.append("")
    lines.append("如果输出 `所有文件都存在`，说明路径基本正确。")
    lines.append("")
    lines.append("## 推荐目录结构")
    lines.append("")
    lines.append("下面这些路径是文件应该放回项目中的位置：")
    lines.append("")
    lines.append("| 文件名 | 目标路径 | 大小 | SHA256 |")
    lines.append("|---|---|---:|---|")

    for r in records:
        name = Path(r.rel_path).name
        lines.append(f"| {name} | `{r.rel_path}` | {human_size(r.size_bytes)} | `{r.sha256}` |")

    lines.append("")
    lines.append("## 建议的网盘目录组织")
    lines.append("")
    lines.append("为了让读者更容易还原，建议你在网盘里保持与仓库相同的目录结构，例如：")
    lines.append("")
    lines.append("```text")
    for r in records:
        lines.append(r.rel_path)
    lines.append("```")
    lines.append("")
    lines.append("这样读者下载后可以直接解压到项目根目录。")
    lines.append("")
    lines.append("## 给作者的建议")
    lines.append("")
    lines.append("- 仓库里只保留代码、配置文件、示例脚本和 README。")
    lines.append("- 权重文件放在网盘，并在主 README 中添加下载链接。")
    lines.append("- 不要把这些大文件重新提交到 Git。")
    lines.append("- 如果以后要在 GitHub 管理大文件，可以改用 Git LFS。")
    lines.append("")
    lines.append("## 自动生成说明")
    lines.append("")
    lines.append(f"此文档由 `separate_assets.py` 根据 `{manifest_path.name}` 自动生成。")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_scan_table(records: List[AssetRecord]) -> None:
    if not records:
        print("未发现候选权重/大文件。")
        return
    print(f"发现 {len(records)} 个候选文件：")
    print("-" * 120)
    print(f"{'size':>12}  {'reason':<18}  path")
    print("-" * 120)
    for r in records:
        print(f"{human_size(r.size_bytes):>12}  {r.reason:<18}  {r.rel_path}")
    print("-" * 120)
    total = sum(r.size_bytes for r in records)
    print(f"总大小: {human_size(total)}")


def load_manifest(manifest_path: Path) -> List[AssetRecord]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [AssetRecord(**item) for item in data.get("files", [])]


def verify_assets(repo: Path, manifest_path: Path) -> int:
    records = load_manifest(manifest_path)
    missing = []
    hash_mismatch = []

    for r in records:
        path = repo / r.rel_path
        if not path.exists():
            missing.append(r.rel_path)
            continue
        actual = sha256sum(path)
        if actual != r.sha256:
            hash_mismatch.append((r.rel_path, r.sha256, actual))

    if not missing and not hash_mismatch:
        print("所有文件都存在，且 SHA256 校验通过。")
        return 0

    if missing:
        print("以下文件缺失：")
        for item in missing:
            print(f"  - {item}")

    if hash_mismatch:
        print("以下文件存在，但 SHA256 不匹配：")
        for rel_path, expected, actual in hash_mismatch:
            print(f"  - {rel_path}")
            print(f"    expected: {expected}")
            print(f"    actual  : {actual}")

    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="扫描、分离并校验仓库中的权重/大文件。")
    sub = parser.add_subparsers(dest="command", required=False)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo", type=Path, default=Path("/home/liutao/video/LightStab"), help="项目根目录")
        p.add_argument("--min-size-mb", type=int, default=DEFAULT_SIZE_MB, help="超过该大小也会被识别为候选文件")
        p.add_argument(
            "--ext",
            nargs="*",
            default=sorted(DEFAULT_EXTENSIONS),
            help="视为权重/大文件的扩展名列表，例如 .pth .ckpt .npy",
        )

    p_scan = sub.add_parser("scan", help="扫描候选权重/大文件")
    add_common(p_scan)

    p_sep = sub.add_parser("separate", help="分离候选权重/大文件")
    add_common(p_sep)
    p_sep.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("/home/liutao/video/LightStab_assets"),
        help="仓库外的保存目录，例如 ../LightStab_assets"
    )
    mode = p_sep.add_mutually_exclusive_group(required=False)
    mode.add_argument("--move", action="store_true", help="移动文件到 assets-dir")
    mode.add_argument("--copy", action="store_true", help="复制文件到 assets-dir")
    p_sep.add_argument("--manifest", type=Path, default=Path("weights_manifest.json"), help="生成的 manifest 路径（默认在 repo 下）")
    p_sep.add_argument("--readme", type=Path, default=Path("WEIGHTS_SETUP.md"), help="生成的说明文档路径（默认在 repo 下）")
    p_sep.add_argument("--update-gitignore", action="store_true", default=True, help="将这些文件路径追加到 .gitignore")

    p_verify = sub.add_parser("verify", help="校验读者是否把文件放到了正确位置")
    p_verify.add_argument("--repo", type=Path, default=Path("/home/liutao/video/LightStab"), help="项目根目录")
    p_verify.add_argument("--manifest", type=Path, default=Path("/home/liutao/video/LightStab/weights_manifest.json"), help="manifest 文件路径")

    return parser


def main() -> int:
    parser = build_parser()

    # 如果没有传任何参数，默认执行 separate
    if len(sys.argv) == 1:
        args = parser.parse_args([
            "separate",
            "--repo", "/home/liutao/video/LightStab",
            "--assets-dir", "/home/liutao/video/LightStab_assets",
            "--move",
            "--update-gitignore"
        ])
    else:
        args = parser.parse_args()

    if args.command == "verify":
        return verify_assets(args.repo.resolve(), args.manifest.resolve())

    repo = args.repo.resolve()
    if not repo.exists():
        print(f"项目目录不存在: {repo}", file=sys.stderr)
        return 2

    extensions = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    records = scan_assets(repo, args.min_size_mb, extensions, DEFAULT_EXCLUDE_DIRS)

    if args.command == "scan":
        print_scan_table(records)
        return 0

    if args.command == "separate":
        mode = "copy" if args.copy else "move"
        assets_dir = args.assets_dir.resolve()
        manifest_path = args.manifest if args.manifest.is_absolute() else repo / args.manifest
        readme_path = args.readme if args.readme.is_absolute() else repo / args.readme

        print_scan_table(records)
        if not records:
            print("未发现需要分离的文件。")
            return 0

        move_or_copy_assets(repo, assets_dir, records, mode)
        write_manifest(records, repo, manifest_path)
        write_setup_readme(repo, manifest_path, readme_path, records)

        if args.update_gitignore:
            update_gitignore(repo, records)
            print(f".gitignore 已更新: {repo / '.gitignore'}")

        print(f"manifest 已生成: {manifest_path}")
        print(f"说明文档已生成: {readme_path}")
        print(f"外部 assets 目录: {assets_dir}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
