# Image generation prompts

Mode: built-in image_gen. Five independent full-slide raster generations.

## Slide 2 precision edit

Edit this Chinese academic slide preserving the entire layout, all factual numbers, all existing text and visual style except the following precision fixes. Change ONLY: 1. The vertical label of the sampling cube currently "内压 / MPa" to "内压 / kPa". 2. Replace the text "Mises 应力" under the top finite element cabin with "结构响应示意". 3. Replace "Mises 应力" to the right of the bottom finite element cabin with "响应示意". 4. Add a small readable grey note just above the footer at bottom right near the page number but without overlap: "流程及结构图均为示意". Keep existing training and genetic search pipeline, title, numbered sections, all formulas and thresholds unchanged. Do not change bars or any other visual. No real numerical finite element fields are claimed in these diagrams. Keep output 16:9 and full resolution.

## Slide 1

Use case: productivity-visual. Generate ONE complete Chinese academic research progress PowerPoint slide as a high resolution raster image, 16:9 landscape 1920x1080 or higher. White background, deep academic royal blue #125CA5 large top-left title, fine light blue horizontal divider beneath title, black Chinese body, restrained dark red emphasis, bottom-right blue small square page number 01. Inspired by a Chinese civil engineering university academic report. No university logo, no invented seals. All Chinese must be crisp, correct and fully legible, generous whitespace, no rounded UI cards. This is page 1 of a consistent five-page deck. Main title exact: "研究背景与阶段目标". Secondary heading exact: "极端环境下模块化增压建筑的轻量化设计". Left half: a large precise architectural engineering cutaway illustration of a rectangular modular pressurized steel building cabin with silver steel frame, closely spaced secondary members, pale transparent envelope and a few windows, isometric view, white background, small outward red arrows normal to surfaces to signify internal pressure. Label under illustration "增压舱结构示意". Small labels near diagram: "内外压差" "构件截面" "梁间距". This is a conceptual illustration, do not make it look like a real simulation output or add a colorbar.
Right half: two vertically spaced unboxed problem-and-method statements with blue headings and black two-line detail:
"问题一：优化迭代计算成本高"
"拓扑与截面组合多，每个候选方案均需结构性能评估"
"新增研究：代理模型辅助优化" (red)
then
"问题二：同类构件统一截面存在冗余"
"不同位置受力不同，统一选型限制进一步减重"
"新增研究：构件分组优化" (red).
Full-width bottom above page number, separated by a thin gray rule, one bold clear line: "目标：在应力与位移约束下，降低结构质量与在线优化成本".
Below in medium font exact formula "min m(x)     uσ ≤ 1.0     uδ ≤ 1.0".
Academic, polished, faithful Chinese typesetting. Do not add fabricated numbers, external claims, extra paragraphs, GNN or RL.

## Slide 2

Use case: productivity-visual. Generate ONE complete 16:9 Chinese academic research PowerPoint slide raster, 1920x1080 or higher. Match previous slide visual design: pure white background, deep blue #125CA5 large top-left Chinese title, fine light-blue horizontal divider at y=120 approx, black body, dark red key emphasis, blue square bottom-right page number 02. Elegant Chinese academic report, readable professional Chinese serif/Song body and bold blue heading, restrained lines, no logo, no cards. Title exact "代理模型辅助优化方法". Subtitle exact "离线学习结构响应，在线批量筛选候选方案".
Main body two horizontal process lanes, flowing left to right with simple blue connecting arrows. Top lane label "离线训练". Four clearly separated unboxed steps, each with a relevant small technical diagram (parameter axes, steel cabin FE mesh, input feature matrix, residual neural network):
1 "参数空间采样" small text "LHS采样：尺寸、内压、拓扑、截面"
2 "有限元标签" small text "强度组合 uσ / 使用组合 uδ"
3 "特征表达" small text "全局参数 + 各组截面 + 空间几何"
4 "ResNet回归" small text "应力、位移分别建模".
Between rows a slim textual note exact "数据积累：20万基础样本 + 3万应力高误差区域补样".
Bottom lane label "在线优化". Four unboxed steps with blue arrows:
"遗传算法生成方案" then "代理模型批量预测" then "质量最小化筛选" then "有限元复核".
A red return connector from final FEA verification back toward design with label "超限时增大分组截面，重新校核". This is a method flowchart, arrows are meaningful connections, not decorative.
Below lanes, a clean flat highlighted key-point area without box, three concise statements each on separate line:
"质量由几何模型直接计算，代理模型预测 uσ 与 uδ"
"代理搜索采用 0.9 惩罚阈值，最终以有限元利用率 ≤ 1.0 判断可行"
"现有在线接口采用 K折 ResNet 集成，单目标特征精简用于独立实验".
Last line smaller gray: "强度组合 1.3DL + 1.5LL + 1.25PL    使用组合 DL + LL + PL".
Fit all text with generous spacing, no overlaps. Do not invent speedup numbers. Keep flow diagrams visually dominant with exact meaningful labels.

## Slide 3

Use case: productivity-visual. One complete high resolution 16:9 Chinese academic slide, 1920x1080 or higher. Same deck white background, deep blue #125CA5 large bold Chinese title at top-left, light blue fine rule below header, black Song/serif Chinese body, dark red key highlights, bottom-right blue page square 03, no logo. Title "构件分组优化方法". Subtitle "以受力规律发现分组，以几何原型适应拓扑变化".
Main upper body a two-stage technical workflow, using meaningful blue arrows and simple outlined flowchart nodes with sharp corners, spacious. First row label "离线规则发现", steps:
"代表性结构有限元分析"
"应力、位移与空间特征"
"对称等价组归并"
"Ward层次聚类".
Second row label "在线分组与设计", steps:
"保存各组几何原型"
"新构件匹配最近原型"
"各组独立选择截面"
"有限元复核".
Connect end of first row down to beginning of second row with a clearly routed nonoverlapping connection.
Below main flow, on left show one isometric rectangular steel cabin wireframe with beams distinguished in three colors only, labeled "构件分组示意", not a simulation output; next to it a small conceptual inset of identical positions under two different member spacings, label "构件数量改变，分组规则仍可复用". No fictitious result values.
On right lower half use exact clear text:
"分组变量：4类构件细分为9组"
"框架梁2组 / 主梁2组 / 次梁3组 / 支撑梁2组"
"几何判别：位置、方向、长度与所在表面"
"新模型分组无需预先进行有限元分析".
Bottom takeaway in blue/red large text:
"在同类构件内部按需求选型，释放进一步减重空间".
Very small but readable source footer "实现：discovery.py / application.py / grouped_workflow.py".
Do not claim every group must use a different section; do not claim full symmetry enforcement online; do not display stress weights or invented performance numbers. All Chinese exact and legible.

## Slide 4

Use case: productivity-visual. One complete high-resolution 16:9 academic Chinese PowerPoint raster slide, 1920x1080 or higher. Consistent white background, large deep blue #125CA5 Chinese title top-left, thin pale-blue horizontal line, black Song/serif body, red selected numbers, blue page square 04 lower-right. No logo, no rounded cards. Title "代理模型成效：精度与单方案评估效率".
Two balanced unboxed sections in main body.
Left section headline "分组响应预测精度". A clean professional scientific table with just three horizontal rules (not full grid) and EXACT entries:
columns "指标" "应力利用率" "位移利用率"
row "样本数" "23万" "20万"
row "输入维数" "210" "236"
row "测试集 R²" "0.9850" "0.9898"
row "MAE" "0.0563" "0.0606".
Below table readable footnote "独立单目标实验，数据集不同，非同一联合模型".
Then a compact blue textual callout "应力特征从536维精简至210维" with a simple visual reduction diagram of two feature stacks, not a fabricated scatter plot.
Right section headline "实际候选方案对照". Exact case line "7.2 × 3.6 × 3.3 m，内压20 kPa".
A horizontal bar chart ONLY TWO bars, exact values:
"有限元评估" 0.703 seconds, gray-blue long bar
"代理模型评估" 0.055 seconds, red short bar.
Bars share ZERO origin, second length exactly about 7.82% of first. Direct data labels "0.703 s" and "0.055 s". Chart x-axis "单次评估耗时 / s", axis range 0 to 0.8 ticks at 0,0.2,0.4,0.6,0.8.
Above or beside chart red callout "约12.7倍" with black small descriptor "单方案评估加速".
Below chart a small clear response table:
columns "利用率" "有限元" "代理模型"
row "应力 uσ" "0.730" "0.714"
row "位移 uδ" "0.556" "0.536".
Across slide bottom, separated by thin gray line, two short limitations in readable small type:
"耗时为本机热启动中位数：FEA重复3次，代理重复30次；含建模与质量计算，不含加载与离线训练。"
"实测调用当前部署模型，网格0.5 m；单方案加速不等同于完整优化加速。"
Small source line if room "来源：项目训练记录与2026-10-07单方案实测".
Exact numbers, no invented model accuracy, no full optimization speedup claim, chart must be numerically proportional, no decorative graphs. Layout professionally readable and not crowded.

## Slide 5

Use case: productivity-visual. ONE complete high resolution 16:9 Chinese academic PowerPoint slide, 1920x1080 or higher. Same five-page deck white background, deep blue #125CA5 bold Chinese title, fine blue divider below heading, black Chinese Song body, red performance highlights, blue bottom-right page square 05. No logos, no rounded cards.
Title "构件分组成效：同拓扑下进一步减重"
Subtitle "三个工程原型对比，4类统一截面与9组独立截面".
Main left 58% width: grouped COLUMN bar chart, truthful zero baseline and common vertical scale from 0 to25 t. Y-axis "结构质量 / t", ticks0,5,10,15,20,25. Legend gray-blue "不分组" and dark red "分组". Three pairs at case labels:
"7.2 m / 20 kPa": grayblue11.120 and red10.235
"9.6 m / 35 kPa": grayblue17.450 and red15.962
"12.0 m / 40 kPa": grayblue22.052 and red19.858.
Label each bar accurately above it with numeric value (3 decimals); red saving annotations above each pair "减重8.0%" "减重8.5%" "减重9.9%".
Thin light horizontal gridlines, scientific chart, no broken axis and no fake data. Bars scaled correctly to shared zero baseline.
Main right 38% width: heading "分组方案有限元校核". Clean three-line table exact:
columns "原型" "uσ" "uδ"
rows "7.2 m" "0.977" "0.919"
"9.6 m" "0.960" "0.670"
"12.0 m" "0.983" "0.672".
Under table exact bold line "三个方案均满足利用率 ≤ 1.0".
Then separated by whitespace a concrete case callout:
"12.0 m原型"
"22.052 t → 19.858 t"
"减少2.194 t，约9.9%" in large red type.
Below in black "主梁与支撑梁按组配置不同截面".
Full width bottom under thin gray line:
"同一原型固定拓扑，仅调整分组截面；宽3.6 m、高3.3 m，有限元网格0.5 m。"
"结果来源：项目既有算例记录；减重率以不分组优化方案为基准。"
Bottom takeaway in blue if enough room "分组扩大截面设计自由度，有限元复核控制最终可行性".
No extra unsupported conclusions, keep all required numeric labels accurate and all text legible.
