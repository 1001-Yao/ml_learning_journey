#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修正 Markdown 里公式的写法，让 GitHub 能正确渲染。

为什么需要它
------------
Obsidian 渲染公式很宽容，GitHub（GFM + MathJax）却有几条硬性要求。下面的结论
都经过 GitHub 官方 Markdown 渲染接口 ``POST /markdown`` 实测：

1. ``$$`` 必须独占一行，且**前后各有一个空行**，否则 ``$$`` 会被并进普通段落:
       a  $$...$$  b   ->  <p>a<br> $$<br> E=mc^2<br> $$</p>    不渲染
2. ``$$`` 后面**不能紧跟空行**，否则 ``$$`` 自己变成一个段落:
       $$ ⏎ (空行) ⏎ E=mc^2 ⏎ $$  ->  <p>$$</p> <p>E=mc^2<br> $$</p>   不渲染
3. 公式块内部不能有空行（会把块切断）。
4. 行内公式只支持 ``$...$``，**不支持** ``\\( ... \\)``；写成 ``\\( i \\)`` 时 GitHub
   会把反斜杠丢掉，显示成普通文字 ``( i )``。``$`` 不能紧贴汉字，两侧需留空格。

Obsidian 自己不需要这些空行，所以从 Obsidian 粘过来的笔记很容易只有一部分
公式失效——正是"有的笔记公式显示不出来"。

本脚本做三件事（幂等，可反复运行）：
- 给 ``$$`` 块前后补空行；
- 去掉 ``$$`` 后 / 闭合 ``$$`` 前的多余空行；
- 把 ``\\( ... \\)`` 改写成 ``$ ... $``（必要时补空格）。

块内部仍有空行的，脚本不擅自改写（会把一个公式拆错），只列出来让你手动处理。

用法
----
    python tools/normalize_markdown_math.py --notes Obsidian_notebook
    python tools/normalize_markdown_math.py --notes Obsidian_notebook --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 紧贴 $ 会让行内公式失效的字符（汉字、字母、数字、下划线等）
GLUE = re.compile(r"[0-9A-Za-z_\u4e00-\u9fff]")
PAREN_MATH = re.compile(r"\\\((?P<body>.+?)\\\)")
# 单行行内公式 $...$（不含 $$）
INLINE_RE = re.compile(r"(?<![\\$])\$(?!\$)(?P<body>[^$\n]+?)\$(?!\$)")


def is_fence(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("```") or s.startswith("~~~")


def fix_blocks(lines: list[str], report: dict, where: str) -> list[str]:
    """补空行、去紧贴的空行。返回新的行列表。"""
    out: list[str] = []
    in_fence = False
    i = 0
    n = len(lines)

    def ensure_blank_before(lineno: int) -> None:
        if out and out[-1].strip() != "":
            out.append("\n")
            report["blank_added"] += 1
            report["touched"].add(lineno)

    while i < n:
        line = lines[i]
        if is_fence(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        stripped = line.strip()

        # 单行 $$...$$
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            ensure_blank_before(i)
            out.append(line)
            if i + 1 < n and lines[i + 1].strip() != "":
                if not line.endswith("\n"):
                    out.append("\n")
                out.append("\n")
                report["blank_added"] += 1
                report["touched"].add(i)
            report["blocks"] += 1
            i += 1
            continue

        if stripped != "$$":
            out.append(line)
            i += 1
            continue

        # 多行公式块：找闭合
        j = i + 1
        while j < n and lines[j].strip() != "$$":
            if is_fence(lines[j]):  # 公式块里出现代码围栏，说明没闭合
                break
            j += 1
        if j >= n or lines[j].strip() != "$$":
            report["skipped"].append(f"{where}:{i + 1} $$ 未闭合，跳过")
            out.append(line)
            i += 1
            continue

        body = lines[i + 1:j]
        lead = 0
        while lead < len(body) and body[lead].strip() == "":
            lead += 1
        trail = 0
        while trail < len(body) - lead and body[len(body) - 1 - trail].strip() == "":
            trail += 1

        if lead or trail:
            report["blank_removed"] += lead + trail
            report["touched"].add(i)

        inner = body[lead:len(body) - trail]
        if any(b.strip() == "" for b in inner):
            report["needs_manual"].append(
                f"{where}:{i + 1} 公式块内部有空行（会把公式切成两段），需人工整理"
            )

        ensure_blank_before(i)
        out.append(line)
        out.extend(inner)
        out.append(lines[j])

        if j + 1 < n and lines[j + 1].strip() != "":
            if not lines[j].endswith("\n"):
                out.append("\n")
            out.append("\n")
            report["blank_added"] += 1
            report["touched"].add(i)

        report["blocks"] += 1
        i = j + 1

    return out


def normalize_inline(line: str) -> tuple[str, int]:
    """整理行内公式：去掉 $ 内侧多余空格；$ 紧贴汉字/字母时在外侧补空格。

    返回 (新行, 改写次数)。只在非代码段调用。
    """
    changed = 0
    out: list[str] = []
    pos = 0
    for m in INLINE_RE.finditer(line):
        body = m.group("body")
        stripped = body.strip()
        if not stripped:  # $ $ 之类，不动
            continue
        prefix = line[pos:m.start()]
        expr = f"${stripped}$"
        before = prefix[-1:] if prefix else ""
        after = line[m.end():m.end() + 1]
        if before and GLUE.match(before):
            expr = " " + expr
        if after and GLUE.match(after):
            expr = expr + " "
        if expr != m.group(0) or before or after:
            pass
        out.append(prefix)
        out.append(expr)
        pos = m.end()
        if expr != m.group(0):
            changed += 1
    if pos == 0:
        return line, 0
    out.append(line[pos:])
    return "".join(out), changed


def convert_paren_math(lines: list[str], report: dict) -> list[str]:
    """把 \\( x \\) 改成 $x$，并整理行内公式（跳过代码块、行内代码和 $$ 公式体）。"""
    out: list[str] = []
    in_fence = False
    in_math = False
    for line in lines:
        if is_fence(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if line.strip() == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_math:
            out.append(line)
            continue

        # 按反引号切成 代码/非代码 交替段，只处理非代码段
        segments = line.split("`")
        new_segments = []
        for idx, seg in enumerate(segments):
            if idx % 2 == 1:
                new_segments.append(seg)
                continue

            if "\\(" in seg:

                def repl(m: re.Match) -> str:
                    body = m.group("body").strip()
                    if not body:
                        return m.group(0)
                    report["paren_math"] += 1
                    return f"${body}$"

                seg = PAREN_MATH.sub(repl, seg)

            seg, changed = normalize_inline(seg)
            report["inline_fixed"] += changed
            new_segments.append(seg)

        out.append("`".join(new_segments))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="修正公式写法，使 GitHub 能渲染 $$...$$ 与行内公式。")
    parser.add_argument("--notes", action="append", default=None, help="笔记目录，默认 Obsidian_notebook，可重复")
    parser.add_argument("--dry-run", action="store_true", help="只统计不改文件")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent.parent
    roots = [Path(p) if Path(p).is_absolute() else here / p for p in (args.notes or ["Obsidian_notebook"])]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("[错误] 笔记目录不存在", file=sys.stderr)
        return 2

    report = {
        "blocks": 0,
        "blank_added": 0,
        "blank_removed": 0,
        "paren_math": 0,
        "inline_fixed": 0,
        "touched": set(),
        "files_changed": 0,
        "needs_manual": [],
        "skipped": [],
    }

    files = sorted(p for r in roots for p in r.rglob("*.md") if ".git" not in p.parts)
    for md in files:
        raw = md.read_bytes()
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        if "$" not in text and "\\(" not in text:
            continue

        lines = text.splitlines(keepends=True)
        before = report["touched"]
        report["touched"] = set()

        new_lines = fix_blocks(lines, report, md.name)
        new_lines = convert_paren_math(new_lines, report)
        new_text = "".join(new_lines)

        if new_text != text:
            report["files_changed"] += 1
        if not args.dry_run and new_text != text:
            md.write_bytes((("\ufeff" if bom else "") + new_text).encode("utf-8"))
        before |= report["touched"]
        report["touched"] = before

    print(f"扫描笔记: {len(files)} 篇")
    print(f"模式: {'试运行（不写文件）' if args.dry_run else '实际执行'}")
    print(f"检查公式块: {report['blocks']} 个")
    print(f"补空行: {report['blank_added']} 行    删除多余空行: {report['blank_removed']} 行")
    print(f"改写 \\( ... \\) 行内公式: {report['paren_math']} 处")
    print(f"整理行内公式边界（去内侧空格/补外侧空格）: {report['inline_fixed']} 处")
    print(f"改动文件: {report['files_changed']} 篇")
    if report["needs_manual"]:
        print(f"\n[需人工处理] {len(report['needs_manual'])} 处：")
        for s in report["needs_manual"]:
            print("  - " + s)
    if report["skipped"]:
        print(f"\n[已跳过] {len(report['skipped'])} 处：")
        for s in report["skipped"][:10]:
            print("  - " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
