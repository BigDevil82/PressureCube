# 整页图片提示词

## 可复用骨架

将以下内容改写为当前页面的完整提示词。先确定文案再生成，不让模型自行编造研究成果。

```text
Use case: productivity-visual / academic research presentation.
Output: ONE complete 16:9 slide, high-resolution raster, not a slide collage.
Audience: Chinese academic research progress review.

Typography: All Chinese titles, body text, captions, table cells and chart labels
must use Microsoft YaHei (微软雅黑), clean modern sans-serif Chinese glyphs.
Bold titles and regular body text. No Songti, SimSun, Ming, calligraphic or serif
Chinese lettering. Mathematical symbols must remain clear and consistent.

Visual style: White background, deep academic blue titles, black body text,
restrained red emphasis. Match the reference's composition and visual density,
while overriding its font, page numbers and unwanted footnotes.
Use whitespace and flat academic layouts, not rounded UI cards.

Visible copy: Supply the exact title, diagram labels, body copy and numeric data.
Composition: Specify the locations and proportions needed by this page's content.
Scientific charts: Keep correct units, direct labels and proportional marks.
Use a common zero baseline for bar comparisons unless a different scale is
explicitly justified by the scientific content. No invented data or fake curves.

Strict exclusions: NO page number, page counter, corner number badge, footer
pagination, source line, citation footer, code filename, implementation path,
production note, AI disclosure, or 'schematic only' disclaimer. Do not add text
beyond the supplied visible copy. No unrequested institution logo.
Retain only research conditions needed to interpret the result, integrated into
the content rather than appended as authoring commentary.
```

## 页面调整

- 背景页：研究对象的结构图或真实照片配合问题与阶段目标。不要伪造项目实景。
- 方法页：少量步骤加清晰连接，解释输入、主要计算和输出。优先概念结构图，避免生成看似真实的有限元数值云图。
- 结果页：一个主对比图配合必要的约束或响应指标。数值多时用短表，不用缩小字体容纳整张实验日志。
- 字体局部修订：明确“只将所有中文替换为微软雅黑无衬线字形，保留文案、数字、布局与图像”，然后检查全页，不只检查标题。
- 删除注释：直接删除页码、来源、实现路径及制作说明，并恢复留白。不要用另一条解释性文字替代。
- 参考图输入：明确其用途是配色和构图参考；用户默认偏好优先于图中的字体、页码和页脚。

## 证据与页面文案

内部来源表记录：结论、原始文件、配置、指标、比较口径、是否本次复现。这些用于检查与回答追问，不自动投放到幻灯片上。

必要条件应保留。例如“固定拓扑，比较截面分组收益”和“单方案评估耗时”决定结论的含义，应作为图标题或算例设置。代码路径、复核过程、生成方式与“来源于本机文件”等说明不帮助听众理解研究，不放上页面。

无法证实的数字应删去或缩小结论范围。不能通过删掉来源与注释，把独立模型写成部署模型、把单方案加速写成全流程加速，或把既有记录写成本次重新验证。

## Windows图片封装

脚本需要本机安装PowerPoint。创建一个UTF-8 JSON清单，图片按明确顺序列出，不通过通配符选择，以免混入草稿。`image`路径相对清单文件目录，`notes`可省略。来源可以放入`notes`，不显示在放映页。

```json
{
  "slides": [
    {"image": "01_background.png", "notes": "Internal source and speaker notes."},
    {"image": "02_method.png"}
  ]
}
```

```powershell
& '<skill-dir>/scripts/package-slides.ps1' `
    -ManifestPath '<report-dir>/slides.json' `
    -OutputPath '<report-dir>/report.pptx' `
    -ExportPdf
```

输出为PPTX、可选同名PDF及`.preview/<PPTX文件名>/`中的逐页检查图。脚本保持16:9和图片比例，不压扁或裁掉页面。默认拒绝覆盖已有成品，修改时使用新文件名。脚本不新增来源、页码或字体属性；图片内的字体必须在图像生成阶段处理。
