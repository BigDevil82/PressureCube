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

密码加密严格参照所附 JS：随机 32 字符 AES 密钥、AES-ECB/PKCS7 加密密码、RSA PKCS1v15 加密 Base64 密钥，二者以 `|` 连接。登录路径为 `/dev-api/auth/login`，Bearer token 只保留在内存中，不输出或写入导出记录。公钥、clientId、tenantId 来自示例，部署环境是否相同仍需用实际登录验证；后两者支持环境变量覆盖。登录不自动刷新或重试。

CLI 也兼容 `.env.local` 中的 `username`、`password` 简写；规范的 `WEBCAD_*` 环境变量优先。CLI 默认地址取所附 JS 的示例地址，可用 `--base-url` 或 `WEBCAD_BASE_URL` 指定实际环境；Python API 则要求显式配置地址。

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py probe
# 使用已提供的 CA，不修改系统证书库
.\.venv\Scripts\python.exe scripts\webcad.py probe --ca-bundle "docs\WebCAD对接\2.证书安装\ca_all.crt"
```

该命令只读：获取会话、构件类型及六种梁柱的实时 Schema，输出到 `results/webcad`。没有会话、多个未指定会话、认证失败或 Schema 不符合文档时中止，不任意选择最后一个页面。

## 第一根测试构件

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py trial --confirm-create
```

创建任务说明中的一根示例方钢管梁：100×100×5 mm、外圆角 8 mm，从 `(0,0,3000)` 到 `(4000,0,3000)` mm，旋转 0°。这是文档联调样例，不是优化方案；在目标试验模型中执行。创建前用实时 Schema 校验，成功后调用 `cad.document.save`。在浏览器检查尺寸、位置、朝向和刷新后持久化。试建记录为 `results/webcad/trial_manifest.json`。

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

- `section_rules`：以真实截面名称为键；`sectionParams` 提供经确认的圆角或尺寸修正，单位 mm；工字钢和槽钢还需 `dimensions_confirmed: true`，表示已确认本地尺寸含义与平台一致。
- `rotations`：按预览生成的 `rotation_key` 填角度，键格式为 `构件类型:面:轴:正负方向`。不默认把所有截面设为 0°。
- `horizontal_frames_confirmed`：确认水平 FRAME 杆件按普通钢梁创建。
- `world_placement_confirmed`：实测确认世界坐标、工作面和截面定位行为后开启。

预览保留所有 `pending`，正式导出拒绝任何未确认项；不补猜测的半径、旋转角、材质或工作面 ID。方管映射 `squareTubeSteel*`，HShape 暂按任务要求候选映射 `gongShapedSteel*`，槽钢映射 `channelSteel*`。由于本地类名和平台 H/工字钢分类存在差异，HShape 映射必须人工确认后才能写入。

正式命令行导出需提供由最终模型生成并审阅的计划：

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py export --plan 最终导出计划.json --confirm-create
```

## 记录与重复导出

每个构件发送前记录 `in_flight`，成功后记录 `node_id`、源杆件、分组、截面、实际返回属性。超时或响应不完整标为 `uncertain`，不自动重试，也不在部分失败后保存。全部成功才调用保存，保存失败保留已创建构件记录。

已有同路径或同目录同目标模型的记录时拒绝再次导出；CLI 与 UI 应保持目标模型对应的记录目录固定。这不是服务端幂等保证：移动/删除记录或使用其他目录不能防重。首版不提供自动删除或断点续传，须先人工核对并清理本次记录中的构件，不能误删用户原有构件。没有事务或自动回滚承诺。

## 实测状态与待确认

2026-10-16：示例地址可达；正常证书验证提示缺少可信证书链；仅对一次不带凭据的诊断请求临时关闭验证后，`GET /api/cad-agent/sessions` 返回 HTTP 401，错误为 `Invalid or missing WebCAD admin token`。当前尚未证明登录、有效会话、Schema、创建和保存成功。

用户选择保持证书校验并提供 `2.证书安装/ca_all.crt` 后，使用该 CA 的 TLS 校验已通过；加密登录返回业务 `code: 500`、消息“用户不存在/密码错误”，未返回 token。已停止继续重试，待核对账号与登录配置。不使用关闭证书验证的方式发送凭据。

离线默认模型预览读取到 906 根实际杆件，生成待确认列表；未执行远程创建。新增测试覆盖登录加密、CA 传递、梁柱和截面映射、实时 Schema 校验、重复导出、超时及保存失败记录；现有分组优化、RFEM 导出与工程布局回归测试也已通过。

仍须联调确认：

1. 实际登录凭据、示例公钥及客户端配置适用性；TLS 证书链；有效 token 是否具有该 HTTP 入口权限。
2. 目标编辑会话及六种类型的实时 Schema；文档使用 `get_by_id`，旧接入说明列 `get`，读取工具名称及入参须核实，不自动猜测调用。
3. 本地 HShape 的 `tf` 沿用原数据字段 `l`；槽钢缺少 `t` 时本地用腹板厚回退。平台要求平均腿厚等定义，不能未经确认当作完全相同。
4. 型钢圆角未保存在 `SectionProperties`；应按真实型号补齐，不编造标准值。
5. 本地 `FRAME` 同时包含竖向与水平杆件。暂按竖向 FRAME 为柱、水平为梁生成候选计划，需确认与四角柱业务定义一致；原始杆段也不等同四根完整柱。
6. OpenSees 按轴线方向和所属面定义局部轴；平台零旋转、正角方向、槽钢开口方向及形心定位是否与之对应，必须用样例实测。没有证据前不自动计算平台旋转角。
