# WebCAD HTTP 对接

## 当前范围

独立模块位于 `src/integrations/webcad`，只负责最终结构模型到平台实体构件的转换与 HTTP 调用，不修改优化、有限元或 Streamlit 现有逻辑。

依据：`docs/WebCAD对接` 下的两份 PDF、任务说明和登录示例。实时 `cad.node.describe_type` 的 `createSchema` 优先于静态文档。只使用 HTTP，不引入 MCP 客户端。

## 登录与连接

浏览器先登录平台并打开目标模型编辑页，保持打开。根目录 `.env.local` 已由 Git 忽略，可设置如下内容（替换占位值）：

```dotenv
WEBCAD_BASE_URL=https://124.223.165.124:4006
WEBCAD_USERNAME=你的账号
WEBCAD_PASSWORD=你的密码
# 或直接设置 WEBCAD_ACCESS_TOKEN，优先于账号密码
# 多个编辑页面时必须指定目标会话
# WEBCAD_SESSION_ID=实际会话id
# 推荐配置受信任证书链
# WEBCAD_CA_BUNDLE=证书文件路径
# 仅联调环境在明确接受风险时使用，默认开启证书验证
# WEBCAD_VERIFY_TLS=false
```

密码加密严格参照所附 JS：随机 32 字符 AES 密钥、AES-ECB/PKCS7 加密密码、RSA PKCS1v15 加密 Base64 密钥，二者以 `|` 连接。根据平台技术人员 2026-10-22 的调整，登录路径改为 `/local-api/auth/login`（原 `/dev-api/auth/login`）；会话与工具调用仍使用 `/api/cad-agent/...`。Bearer token 只保留在内存中，不输出或写入导出记录。公钥、clientId、tenantId 来自示例，部署环境是否相同仍需用实际登录验证；后两者支持环境变量覆盖。登录不自动刷新或重试。

CLI 也兼容 `.env.local` 中的 `username`、`password` 简写；规范的 `WEBCAD_*` 环境变量优先。CLI 默认地址取所附 JS 的示例地址，可用 `--base-url` 或 `WEBCAD_BASE_URL` 指定实际环境；Python API 则要求显式配置地址。

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py probe
# 使用已提供的 CA，不修改系统证书库
.\.venv\Scripts\python.exe scripts\webcad.py probe --ca-bundle "docs\WebCAD对接\2.证书安装\ca_all.crt"
```

该命令只读：获取会话、构件类型及方管、H 型钢、工字钢、槽钢共八种梁柱的实时 Schema，输出到 `results/webcad`。没有会话、多个未指定会话、认证失败或 Schema 不符合文档时中止，不任意选择最后一个页面。

## 第一根测试构件

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py trial --confirm-create
```

创建任务说明中的一根示例方钢管梁：100×100×5 mm、外圆角 8 mm，从 `(0,0,3000)` 到 `(4000,0,3000)` mm，旋转 0°。这是文档联调样例，不是优化方案；在目标试验模型中执行。创建前用实时 Schema 校验，成功后调用 `cad.document.save`。在浏览器检查尺寸、位置、朝向和刷新后持久化。试建记录为 `results/webcad/trial_manifest.json`。

## 完整钢结构试建

```powershell
.\.venv\Scripts\python.exe -u scripts\webcad.py cabin --confirm-create --ca-bundle "docs\WebCAD对接\2.证书安装\ca_all.crt" --cleared-manifest results/webcad/trial_manifest.json
```

`cabin` 读取当前 `export_geometry.py` 的默认工程模型，创建钢结构骨架，不包括舱壁板、玻璃、连接节点和机电构件，也不冒充优化后的最终方案。当前默认参数为长 8.5 m、宽 3.7 m、高 3.3 m、主梁间距 1.05 m、支撑梁最大间距 0.5 m，右侧第 1、2 跨窗洞。原始模型共有 906 个实际杆段：102 个方管梁段、84 个方管柱段、413 个 H 型钢梁段、307 个槽钢梁段；杆段数量不等于完整工程构件数量。

这是显式的视觉试建模式：`MappingConfig.trial_mode=True`，同时开启方管理论参考圆角。缺少用户旋转映射的杆段先用平台 0°；世界坐标、工作面行为、水平 FRAME 杆件分类和局部轴对应仍记录为 `warnings`，不会标为已经验证。未知截面尺寸仍然阻止创建，不通过试建模式绕过。已给定的 `rotations` 优先于临时 0°。正式导出默认关闭试建模式，仍需要解决 `pending`。

创建前统一校验全部类型的实时 Schema；之后串行发送创建请求，只检查每次响应并写记录，不逐根调用读取工具。每 25 根显示一次进度，全部成功后保存。计划与记录分别为 `results/webcad/cabin_plan.json`、`results/webcad/export_manifest.json`。

如果旧的单根测试构件已由用户删除，`--cleared-manifest` 会在同一目标文档中用一次 `cad.node.list` 确认旧记录内全部 nodeId 已不存在，再把旧记录标为 `cleared`。它不删除任何远程构件；未删除、目标文档不符或旧记录包含不确定创建结果时拒绝自动清理。没有旧记录需要处理时省略该参数。

编辑页尚未建立 CAD Agent 会话时，可添加 `--wait-for-editor 120` 有限等待；期间只查询会话，不重复登录或发送创建请求。超时仍无会话则停止，不绕过连接条件。

后续整舱重试状态（2026-10-22）：用户已授权清理旧测试梁后继续，但新请求在 TLS 握手阶段失败，未登录、删除或新增构件。直连诊断排除了仅由环境代理导致的可能：服务器叶证书 Authority Key Identifier 为 `93f2b2891b5dc212e7e55ba25f71a0b23dc45ce4`，所提供 CA 的 Subject Key Identifier 为 `3167d62a46a0d9a7eb4b135855468946263f181b`，使用所提供 CA 公钥验证叶证书签名失败。颁发者名称相同不能证明密钥相同；本机信任库也只有当前这份 CA。需平台提供可验证当前服务器证书链的 CA/中间证书，或恢复匹配原 CA 的服务器证书。保持 TLS 验证，不使用 `verify=False` 绕过。

## 最终模型与 Streamlit 入口

直接读取原始 `StructuralModel` 中每根实际杆件的 `section`、`beam_type`、`on_face` 和 `section_assignment.beam_groups`，不再按四大类别硬编码截面值，因而保留构件分组优化后的真实截面差异。跳过虚拟杆，拒绝带 `parent_id` 的有限元网格模型。当前逐条导出原始几何杆段，不自动合并成四根整柱。

分组优化入口是 `AppSession().grouping.optimization_result.model`（即 `GroupedOptimizationResult.model`）；普通优化应用最优参数并重新生成后使用 `AppSession().model`。不要使用 `analysis.meshed_model`。调用方应只把最终校核通过的方案交给正式导出。

```python
from src.integrations.webcad import (
    MappingConfig, WebCADClient, WebCADConfig,
    build_export_plan, export_to_webcad,
)

# mapping_data 来自人工核实后保存的 JSON，不能凭空补圆角或角度。
mapping = MappingConfig(**mapping_data)
plan = build_export_plan(final_model, mapping)
# Streamlit 可先显示 plan["pending"] 和构件表，提供确认按钮。
client = WebCADClient(WebCADConfig.from_env())
try:
    result = export_to_webcad(
        final_model, client, mapping,
        "results/webcad/final/export_manifest.json",
        on_progress=lambda done, total: progress.progress(done / total),
    )
finally:
    client.close()
```

API 不依赖 Streamlit，也不自行读取 `.env.local`；应用按既有配置机制创建 `WebCADConfig`，CLI 才加载该文件。`on_progress` 可直接用于 UI 进度。优化器和现有界面尚未自动触发导出，避免未核实配置就修改远程模型。

### 离线预览

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py preview
.\.venv\Scripts\python.exe scripts\webcad.py preview --mapping 已核实映射.json
```

CLI 预览使用现有 `scripts/export_geometry.py` 的默认工程模型，只用于适配检查，不冒充最终优化结果。真实方案调用 `build_export_plan(final_model, mapping)`。所有预览坐标统一由 m 换算为 mm，不调用平台。

`MappingConfig` 字段：

- `section_rules`：以真实截面名称为键；`sectionParams` 提供经确认的圆角或尺寸修正，单位 mm。标准补充库未匹配到的开口截面还需 `dimensions_confirmed: true`，表示已确认本地尺寸含义与平台一致。
- `rotations`：按预览生成的 `rotation_key` 填角度，键格式为 `构件类型:面:轴:正负方向`。不默认把所有截面设为 0°。
- `horizontal_frames_confirmed`：确认水平 FRAME 杆件按普通钢梁创建。
- `world_placement_confirmed`：实测确认世界坐标、工作面和截面定位行为后开启。
- `use_box_reference_radius`：显式启用 GB/T 3094—2012 表 A.1 注中的理论计算参考圆角，默认关闭；开启不表示已确认厂家实测圆角或供货工艺。
- `trial_mode`：仅用于视觉试建，允许把未核实的放置约定与临时 0° 朝向记录为警告；不放宽截面尺寸和 Schema 校验，不用于宣称实体与有限元完全一致。

预览保留所有 `pending`，正式导出拒绝任何未确认项；不补猜测的半径、旋转角、材质或工作面 ID。方管映射 `squareTubeSteel*`；根据新提供的 GB/T 11263—2024，HShape 对应 `hShapedSteel*`，不能按普通工字钢 `gongShapedSteel*` 的字段传入；槽钢映射 `channelSteel*`。钢柱业务编码按实时 Schema 的 `businessQuery` 使用 `15-002`，不沿用静态文档的 `02-003`。

### 标准几何补充库

`src/data/steel_section_geometry.json` 独立保存三本标准的尺寸及出处，覆盖两套分析库所用的 21 种 H 型钢、20 种槽钢和 23 种方管尺寸/参考记录。按完整数值尺寸匹配，不按排序索引或截面显示名称猜测型号。H 型钢的 t1、t2、r 及槽钢的 t、r、r1 已核对原表；圆角的参考属性仍遵从标准条文。

槽钢 BIM 几何采用标准翼缘厚度，分析库现有的 `tf=tw` 回退保持原样；例如 20b 的标准 t=11 mm，而旧分析 t=9 mm。导出每根构件的 `section_geometry.analysis_differences` 会记录这项区别。方管的参考圆角、允许上限、表中是否列出，以及原库面积/参考面积也会分别记录，不把参考几何宣称为实测产品。

GB/T 3094—2012 表 A.1 注用于理论重量计算：S≤6 时 R=1.5S，S>6 时 R=2S；5.4 条表 3 另给允许上限。150×150×4/5/6 未直接列入尺寸表，原库面积也与参考计算不完全一致，供货规格及采用参考圆角须明确确认。默认不自动开启 `use_box_reference_radius`。

再生数据与参数核对报告：

```powershell
.\.venv\Scripts\python.exe scripts\sync_section_geometry.py
```

提取脚本需要 `.venv` 中的 PyMuPDF，运行导出则只读取已生成的 JSON，不依赖 PDF 解析。核对报告位于 `results/webcad/section_standards/section_parameters_report.md`，不纳入 Git。原始 `steel_sections*.json` 和分析流程没有改动。

正式命令行导出需提供由最终模型生成并审阅的计划：

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py export --plan 最终导出计划.json --confirm-create
```

## 记录与重复导出

每个构件发送前记录 `in_flight`，成功后记录 `node_id`、源杆件、分组、截面、实际返回属性。超时或响应不完整标为 `uncertain`，不自动重试，也不在部分失败后保存。全部成功才调用保存，保存失败保留已创建构件记录。

已有同路径或同目录同目标模型的记录时拒绝再次导出；CLI 与 UI 应保持目标模型对应的记录目录固定。这不是服务端幂等保证：移动/删除记录或使用其他目录不能防重。首版不提供自动删除或断点续传，须先人工核对并清理本次记录中的构件，不能误删用户原有构件。没有事务或自动回滚承诺。

## 实测状态与待确认

2026-10-16 的联调中，使用提供的 CA 后 TLS 校验通过，但旧 `/dev-api/auth/login` 返回“用户不存在/密码错误”。Windows 系统 `USERNAME` 与本地 `username` 的同名冲突已修正：简写先转换为 `WEBCAD_*`，不读取或覆盖系统登录名。

2026-10-22，平台技术人员调整后，按要求把登录前缀改为 `/local-api`，已实测完成：

- 使用 `2.证书安装/ca_all.crt`，全程开启 TLS 证书校验；原加密算法、公钥、clientId 和 tenantId 可用，账号密码登录成功。
- 获取到唯一编辑会话，目标文档为 `mcp-demo`。
- `cad.node.list_types` 和六种钢梁、钢柱的 `cad.node.describe_type` 均成功，Schema 已保存到 `results/webcad`。方管外圆角、工字钢/槽钢两个圆角均为必填项，不能省略或臆造。
- 实时 Schema 校验后成功创建一根 `structure.squareTubeSteelGirder`。截面为 100×100×5 mm、外圆角 8 mm，起点 `(0,0,3000)`、终点 `(4000,0,3000)` mm，定位 0、旋转 0°；创建响应中的实际参数与请求一致。
- 创建的 `nodeId` 为 `7341552054505472_M09_YH-s`；`cad.document.save` 返回成功；随后独立调用 `cad.node.list` 再次查询到该构件。

会话、类型、Schema、创建记录与只读复核分别位于 `results/webcad/sessions.json`、`types.json`、`structure.*.json`、`trial_manifest.json` 和 `node_list_after_trial.json`，不纳入 Git。尚未通过浏览器刷新验证持久化，也未导入完整增压舱。

离线默认模型预览读取到 906 根实际杆件，生成待确认列表。单构件联调时 21 项 WebCAD 测试通过；标准参数补充后共 30 项测试通过，覆盖原两套库全部 H/槽钢匹配、方管参考圆角显式选择、出处保留及分析数据不变。现有分组优化、RFEM 导出与工程布局测试在前次开发时已通过。

仍须联调确认：

1. 在浏览器检查试建实体的可视几何、坐标与截面朝向，并通过刷新检查保存后的持久化；API 成功不等同于这些人工验收已完成。
2. 单构件详细读取工具：文档使用 `get_by_id`，旧接入说明列 `get`，工具名称及入参仍须核实。本次只读复核使用已确认可用的 `cad.node.list`。
3. HShape 的 `l` 对应标准 t2 已核对；槽钢 BIM 标准厚度与原分析回退值不完全相同，尚未统一两者，不应宣称实体与分析截面完全相同。
4. 方管圆角采用理论计算参考值还是厂家值、GB/T 3094 冷拔工艺是否为实际供货标准、未列尺寸供货约定尚待确认；默认不启用参考圆角。
5. 本地 `FRAME` 同时包含竖向与水平杆件。暂按竖向 FRAME 为柱、水平为梁生成候选计划，需确认与四角柱业务定义一致；原始杆段也不等同四根完整柱。
6. OpenSees 按轴线方向和所属面定义局部轴；平台零旋转、正角方向、槽钢开口方向及形心定位是否与之对应，必须用样例实测。没有证据前不自动计算平台旋转角。
