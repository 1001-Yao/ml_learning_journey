#### 文档说明

此学习文档使用 Markdown 语法编写。其中《吴恩达机器学习》笔记多为文字记录，
而《李沐动手学深度学习》则对采用截图老师 ppt 记录。

#### 关于笔记里的图片

笔记最初是在 Obsidian 里写的，粘贴截图时 Obsidian 会把图片存成
`Pasted image 20260702104553.png` 这样的文件，并在正文里写成 wiki 嵌入语法：

```
![[Pasted image 20260702104553.png]]
![[Pasted image 20260702104553.png|378]]
```

GitHub 和 PyCharm 的 Markdown 渲染器都不认识 `![[...]]`，只会把方括号里的文件名
当成普通文字打印出来（看起来就是一串字母数字）；而且截图原本放在 Obsidian 仓库目录里，
并没有提交到本仓库，所以即使语法正确也找不到图片。

现在已统一改成标准 Markdown 图片语法，图片文件也一并提交进仓库：

```
![pasted-image-20260702104553](images/pasted-image-20260702104553.png)
```

- 每篇笔记的同级目录下都有一个 `images/` 文件夹，放这篇笔记用到的截图；
- 图片文件名统一规范成小写、空格换成 `-`，避免 GitHub 链接里出现 `%20`；
- 这样 GitHub 网页、PyCharm 预览、Typora、Obsidian 都能正常显示。

#### 以后新增笔记怎么办

从 Obsidian 复制过来的新笔记仍然会带 `![[...]]`。提交前在仓库根目录跑一次
同步脚本即可自动修正（脚本是幂等的，可以反复运行）：

```bash
python tools/sync_obsidian_images.py --vault "C:/Users/姚国庆/Documents/Obsidian Vault"
```

先看效果可以加 `--dry-run` 只统计不改文件。
