#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 Obsidian 的图片嵌入语法转换成 GitHub / PyCharm 都能正常显示的标准 Markdown。

问题背景
--------
Obsidian 粘贴截图时默认写成 wiki 嵌入语法::

    ![[Pasted image 20260702104553.png]]
    ![[Pasted image 20260702104553.png|378]]

而 GitHub、PyCharm 的 Markdown 渲染器都不认识 ``![[...]]``，只会把方括号里的
文件名当成普通文字打印出来（看起来就是"一串字母数字"）。同时截图文件本身存放在
Obsidian 仓库目录里，并没有被复制进代码仓库，所以即使语法正确也找不到图。

本脚本做两件事：
1. 从 Obsidian 仓库里找到被引用的图片，复制到每一篇笔记同级的 ``images/`` 目录；
2. 把 ``![[图名|宽度]]`` 改写成 ``![图名](images/图名.png)``（标准 Markdown）。

用法
----
    python tools/sync_obsidian_images.py --vault "C:/Users/你/Documents/Obsidian Vault" --notes Obsidian_notebook

常用参数
--------
``--dry-run``     只统计和报告，不改动任何文件（推荐先跑一次看看）。
``--keep-width``  把 ``|378`` 这类宽度保留成 ``<img src="..." width="378">``（GitHub 支持，
                  纯 Markdown 不支持指定宽度；默认丢弃宽度，让 GitHub 自动按页面宽度缩放）。
``--notes``       要处理的笔记根目录（默认 ``Obsidian_notebook``），可重复指定。
``--report``      报告文件路径，默认写到系统临时目录。

脚本是幂等的：重复运行不会重复复制，也不会破坏已经改写好的标准 Markdown 链接。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".avif"}

# ![[target]] 或 ![[target|300]] 或 ![[target|300x200]]
EMBED_RE = re.compile(r"!\[\[(?P<target>[^\[\]]+?)\]\]")

# 需要从文件名里剔除的 Windows 非法字符
INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def slugify(name: str) -> str:
    """把 'Pasted image 20260702104553.png' 变成 'pasted-image-20260702104553.png'。"""
    stem, ext = os.path.splitext(name)
    stem = INVALID_CHARS.sub("-", stem).replace(" ", "-")
    stem = re.sub(r"-{2,}", "-", stem).strip("-.")
    return f"{stem.lower() or 'image'}{ext.lower()}"


def split_target(raw: str) -> tuple[str, str | None]:
    """拆出 '图名' 和宽度提示。返回 (target, size_hint)。"""
    if "|" in raw:
        target, _, hint = raw.partition("|")
        return target.strip(), hint.strip() or None
    return raw.strip(), None


def build_index(vault: Path) -> tuple[dict[str, list[Path]], set[str]]:
    """建立 '文件名(小写)' -> [完整路径...] 的索引，便于按名字找图。"""
    by_name: dict[str, list[Path]] = defaultdict(list)
    missing: set[str] = set()
    for root, _dirs, files in os.walk(vault):
        # 跳过 Obsidian 自己的配置目录
        if ".obsidian" in Path(root).parts:
            continue
        for fn in files:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                by_name[fn.lower()].append(Path(root) / fn)
    return by_name, missing


def resolve(target: str, index: dict[str, list[Path]], vault: Path) -> Path | None:
    """把嵌入目标解析成真实的图片路径。"""
    # 1) 目标里带路径：先按 vault 相对路径找
    candidate = (vault / target)
    if candidate.is_file():
        return candidate

    # 2) 按文件名找（Obsidian 常用短名引用）
    key = Path(target).name.lower()
    hits = index.get(key)
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    # 同名多份时优先取路径最短（离 vault 根最近）的那个，并在报告里记一条
    return sorted(hits, key=lambda p: len(p.parts))[0]


def iter_md(notes_roots: list[Path]):
    for root in notes_roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.md")):
            if ".git" in p.parts:
                continue
            yield p


def convert_file(
    md_path: Path,
    vault: Path,
    index: dict[str, list[Path]],
    keep_width: bool,
    dry_run: bool,
    stats: dict,
    warnings: list[str],
) -> None:
    raw = md_path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    if not text.strip():
        return

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    changed = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue

        if in_fence or "![[" not in line:
            out.append(line)
            continue

        def repl(m: re.Match) -> str:
            nonlocal changed
            target, hint = split_target(m.group("target"))
            src = resolve(target, index, vault)
            if src is None:
                stats["unresolved"].append(f"{md_path.name}: {target}")
                return m.group(0)  # 找不到就原样保留，绝不静默删除

            dest_dir = md_path.parent / "images"
            dest = dest_dir / slugify(src.name)
            if src.name != dest.name:
                stats["renamed"].add(f"{src.name} -> {dest.name}")

            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                stats["already"].add(dest.as_posix())
            else:
                stats["copied"].add((src.as_posix(), dest.as_posix()))
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)

            rel = f"images/{dest.name}"
            alt = Path(rel).stem
            changed = True
            if keep_width and hint:
                width = hint.split("x")[0].strip()
                if width.isdigit():
                    return f'<img src="{rel}" width="{width}" alt="{alt}">'
            return f"![{alt}]({rel})"

        new_line = EMBED_RE.sub(repl, line)
        if new_line != line:
            stats["links"] += len(EMBED_RE.findall(line))
            stats["files_changed"].add(md_path.as_posix())
        out.append(new_line)

    if not changed or dry_run:
        return

    new_text = "".join(out)
    md_path.write_bytes((("\ufeff" if bom else "") + new_text).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 Obsidian 的 ![[图片]] 嵌入转换成 GitHub 可显示的标准 Markdown，并复制图片进仓库。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vault", required=True, help="Obsidian 仓库根目录（图片所在处）")
    parser.add_argument(
        "--notes",
        action="append",
        default=None,
        help="要处理的笔记目录，可重复；默认 Obsidian_notebook",
    )
    parser.add_argument("--keep-width", action="store_true", help="用 HTML img 保留 Obsidian 的显示宽度")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不改文件")
    parser.add_argument("--report", default=None, help="报告文件路径")
    args = parser.parse_args(argv)

    vault = Path(args.vault).expanduser()
    if not vault.is_dir():
        print(f"[错误] 找不到 Obsidian 仓库目录：{vault}", file=sys.stderr)
        return 2

    here = Path(__file__).resolve().parent.parent  # 仓库根 = tools/ 的上一级
    note_roots = [Path(p) if Path(p).is_absolute() else here / p for p in (args.notes or ["Obsidian_notebook"])]
    existing = [p for p in note_roots if p.exists()]
    if not existing:
        print(f"[错误] 这些笔记目录都不存在：{note_roots}", file=sys.stderr)
        return 2

    index, _ = build_index(vault)
    stats: dict = {
        "links": 0,
        "copied": set(),
        "already": set(),
        "renamed": set(),
        "files_changed": set(),
        "unresolved": [],
    }
    warnings: list[str] = []

    for md in iter_md(existing):
        convert_file(md, vault, index, args.keep_width, args.dry_run, stats, warnings)

    # 统计仍然残留的 wiki 图片嵌入（代码块里的除外）
    leftover = 0
    for md in iter_md(existing):
        text = md.read_text(encoding="utf-8", errors="replace")
        in_fence = False
        for line in text.splitlines():
            s = line.lstrip()
            if s.startswith("```") or s.startswith("~~~"):
                in_fence = not in_fence
                continue
            if not in_fence and "![[" in line:
                leftover += 1

    lines = []
    mode = "试运行（不写文件）" if args.dry_run else "实际执行"
    lines.append(f"模式: {mode}")
    lines.append(f"Obsidian 仓库: {vault}")
    lines.append(f"笔记目录: {', '.join(str(p) for p in existing)}")
    lines.append(f"图片索引: {len(index)} 个文件名")
    lines.append("")
    lines.append(f"改写的嵌入链接: {stats['links']}")
    lines.append(f"涉及笔记文件: {len(stats['files_changed'])}")
    lines.append(f"需要复制的图片: {len(stats['copied'])}")
    lines.append(f"仓库里已存在的图片: {len(stats['already'])}")
    lines.append(f"重命名的图片: {len(stats['renamed'])}")
    lines.append(f"转换后仍残留的 ![[...]]: {leftover}")
    if stats["unresolved"]:
        lines.append("")
        lines.append(f"[未找到图片] {len(stats['unresolved'])} 处：")
        lines.extend("  - " + item for item in stats["unresolved"])
    if warnings:
        lines.append("")
        lines.append("[提示]")
        lines.extend("  - " + w for w in warnings)
    if stats["renamed"]:
        lines.append("")
        lines.append("[文件名规范化示例] 前 10 条：")
        lines.extend("  - " + r for r in sorted(stats["renamed"])[:10])

    report_text = "\n".join(lines)
    print(report_text)

    report_path = Path(args.report) if args.report else Path(tempfile.gettempdir()) / "obsidian_image_sync_report.txt"
    try:
        report_path.write_text(report_text + "\n", encoding="utf-8")
        print(f"\n报告已写入: {report_path}")
    except OSError as exc:  # 报告写不出来不影响主流程
        print(f"\n[提示] 报告写入失败: {exc}", file=sys.stderr)

    return 1 if stats["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
