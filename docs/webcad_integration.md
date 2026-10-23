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
.\.venv\Scripts\python.exe scripts\webcad.py probe --ca-bundle "docs\WebCAD对接\2.证书安装\ca.crt"
```

该命令只读：获取会话、构件类型及方管、H 型钢、工字钢、槽钢共八种梁柱的实时 Schema，输出到 `results/webcad`。没有会话、多个未指定会话、认证失败或 Schema 不符合文档时中止，不任意选择最后一个页面。

## 第一根测试构件

```powershell
.\.venv\Scripts\python.exe scripts\webcad.py trial --confirm-create
```

创建任务说明中的一根示例方钢管梁：100×100×5 mm、外圆角 8 mm，从 `(0,0,3000)` 到 `(4000,0,3000)` mm，旋转 0°。这是文档联调样例，不是优化方案；在目标试验模型中执行。创建前用实时 Schema 校验，成功后调用 `cad.document.save`。在浏览器检查尺寸、位置、朝向和刷新后持久化。试建记录为 `results/webcad/trial_manifest.json`。

## 完整钢结构试建

完整模型优先使用下文的批量接口。下面的串行命令保留用于单构件诊断，不建议用于完整舱体：

```powershell
.\.venv\Scripts\python.exe -u scripts\webcad.py cabin --confirm-create --ca-bundle "docs\WebCAD对接\2.证书安装\ca.crt" --cleared-manifest results/webcad/trial_manifest.json
```

`cabin` 读取当前 `export_geometry.py` 的默认工程模型，创建钢结构骨架，不包括舱壁板、玻璃、连接节点和机电构件，也不冒充优化后的最终方案。当前默认参数为长 8.5 m、宽 3.7 m、高 3.3 m、主梁间距 1.05 m、支撑梁最大间距 0.5 m，右侧第 1、2 跨窗洞。原始模型共有 906 个实际杆段：102 个方管梁段、84 个方管柱段、413 个 H 型钢梁段、307 个槽钢梁段；杆段数量不等于完整工程构件数量。

这是显式的视觉试建模式：`MappingConfig.trial_mode=True`，同时开启方管理论参考圆角。缺少用户旋转映射的杆段先用平台 0°；世界坐标、工作面行为、水平 FRAME 杆件分类和局部轴对应仍记录为 `warnings`，不会标为已经验证。未知截面尺寸仍然阻止创建，不通过试建模式绕过。已给定的 `rotations` 优先于临时 0°。正式导出默认关闭试建模式，仍需要解决 `pending`。

创建前统一校验全部类型的实时 Schema；之后串行发送创建请求，只检查每次响应并写记录，不逐根调用读取工具。每 25 根显示一次进度，全部成功后保存。计划与记录分别为 `results/webcad/cabin_plan.json`、`results/webcad/export_manifest.json`。

如果旧的单根测试构件已由用户删除，`--cleared-manifest` 会在同一目标文档中用一次 `cad.node.list` 确认旧记录内全部 nodeId 已不存在，再把旧记录标为 `cleared`。它不删除任何远程构件；未删除、目标文档不符或旧记录包含不确定创建结果时拒绝自动清理。没有旧记录需要处理时省略该参数。

编辑页尚未建立 CAD Agent 会话时，可添加 `--wait-for-editor 120` 有限等待；期间只查询会话，不重复登录或发送创建请求。超时仍无会话则停止，不绕过连接条件。

后续整舱重试状态（2026-10-22）：用户已授权清理旧测试梁后继续，但新请求在 TLS 握手阶段失败，未登录、删除或新增构件。直连诊断排除了仅由环境代理导致的可能：服务器叶证书 Authority Key Identifier 为 `93f2b2891b5dc212e7e55ba25f71a0b23dc45ce4`，所提供 CA 的 Subject Key Identifier 为 `3167d62a46a0d9a7eb4b135855468946263f181b`，使用所提供 CA 公钥验证叶证书签名失败。颁发者名称相同不能证明密钥相同；本机信任库也只有当前这份 CA。需平台提供可验证当前服务器证书链的 CA/中间证书，或恢复匹配原 CA 的服务器证书。保持 TLS 验证，不使用 `verify=False` 绕过。

补充证据：单构件成功记录的本地保存时间为 2026-10-22 10:32:16（北京时间），当前叶证书 NotBefore 为同日 15:41:04（北京时间），不能视作上午成功时同一份有效证书。连续三次直连取得相同 SHA-256 指纹 `a24fe49de5ff1a01372251fde23e83a0697931a28ad856c9d196bea820b3eefb`，服务端当前只发送一张叶证书；SAN 仅含 `10.1.1.110/111/112/113/115`，未含访问入口 `124.223.165.124`。即使补齐 CA，公网 IP 的名称校验也需要处理。由部署、证书重载还是网关切换导致，客户端无法确定；应由平台恢复原入口证书，或提供匹配当前证书的可信链及覆盖实际访问地址的证书/正式域名。

平台随后确认重新生成证书并提供 `2.证书安装/ca.crt`。已按用户授权安装至 `Cert:\CurrentUser\Root`，指纹（SHA-1）为 `1D0A1C5A3F4AF68CC3FF76F09BC1CD6F764DCF48`，SHA-256 为 `17278cf1f38a66f5a87c9e0206634c8a4f8061d604f018cdc1ccda75c739d800`。新 CA 的 SKI 与服务器 AKI 一致，签名验证通过；使用新 CA 的公网请求报错已变为 `IP address mismatch`，原因仍是服务器 SAN 没有公网 IP。文档内网示例 `https://10.1.1.112:3001` 从本机直连超时，不能作为当前替代入口。尚未删除旧测试梁或创建整舱；需平台为实际公网地址签发匹配 SAN 的服务器证书，或提供可达且证书匹配的正式入口。客户端现保留具体 TLS 错误，区分 CA 链和地址不匹配，不统一建议更换 CA。

2026-10-22 平台再次替换服务器证书后，公网入口的 CA 链和 IP 地址校验均已通过，账号登录成功。当前叶证书 SHA-256 为 `aac340d58184c02687ff2beeeecb8847cf858fe74b0b709c4178ef469fefc656`，SAN 已包含 `124.223.165.124`。前述 TLS 阻塞已解除，无需关闭证书校验。刷新编辑页后恢复了 `mcp-demo` 会话；一次列表查询确认旧测试梁已不存在，原 `trial_manifest.json` 已保留并标为 `cleared`，未删除其他构件。

随后整舱请求通过实时 Schema 校验并开始创建。前 380 个杆段返回成功及 nodeId，第 381 个请求返回 HTTP 404，重新查询编辑会话列表为空。导出记录 `results/webcad/export_manifest.json` 状态为 `interrupted`，包含 380 个 `created` 和 1 个 `failed`；尚未调用整舱保存，其余 525 个杆段尚未发送。会话消失的具体原因待平台或浏览器侧核实，不能将已返回成功视为已持久化。恢复连接后须先核对已创建构件，再决定清理或续建，不能直接重跑整舱命令。此次相关 37 项单元测试通过，未修改分析或优化逻辑。

### Python 批量创建

2026-10-23 技术人员提供 `docs/WebCAD对接/mcp批量创建` 中的转换脚本和调用格式后，新增独立模块 `src/integrations/webcad/batch.py`。HTTP 地址和调用信封保持原样，工具名改为 `cad.node.create_batch`，其参数为：

```json
{
  "nodeList": [{"type": "structure.squareTubeSteelGirder", "params": {}}],
  "components": [],
  "batchSize": 100,
  "skipCutRelations": true
}
```

以上 `params` 仅作结构占位，实际必须包含原计划中的完整截面、端点和业务参数。一次提交全部杆段，由平台按 `batchSize` 分片保存，不是 Python 并发调用或循环发送单根创建。`skipCutRelations=true` 沿用技术人员示例，跳过切割关系，不代表已验证连接节点的加工几何。

```powershell
.\.venv\Scripts\python.exe -u scripts\webcad.py cabin --batch --batch-size 100 --confirm-create --ca-bundle "docs\WebCAD对接\2.证书安装\ca.crt" --wait-for-editor 120
# 最终优化方案：使用实际 final_model 生成并审阅的计划，不使用 cabin 默认试建模型
.\.venv\Scripts\python.exe -u scripts\webcad.py export --plan 最终导出计划.json --batch --confirm-create --ca-bundle "docs\WebCAD对接\2.证书安装\ca.crt"
```

同一目标已有未清理的导出记录时仍拒绝重复创建。此次仅在列表确认上次的 380 个 ID 均不存在后，使用 `--cleared-manifest results/webcad/export_manifest.json` 保留并核销旧中断记录；不会远程删除其他构件。已标记 `cleared` 的记录不要再次传给该参数。

转换保留 m→mm 后的原坐标与截面，新增稳定的 `params.id=cabin-{source_id}`、`compentName`、`detailName` 和 `userData.originId` 等字段。906 条 `nodeList` 与所附 JS 生成文件逐项一致。按技术人员说明，这几种钢构件使用世界坐标几何，不能只凭截面和长度跨位置共享 `shapeReuseKey`，本实现不设置该键。文档指出 BeamNode 路径可能忽略截面旋转，因此批量入口暂拒绝非 0°，不宣称已解决朝向映射；材质仍使用平台默认值。

批量函数 `export_batch_plan(plan, client, manifest_path, batch_size=100, on_progress=None)` 可直接从 Streamlit 调用，`plan` 使用 `build_export_plan(final_model, mapping)` 生成。`on_progress` 仅在整体完成时回调，不模拟服务端实时进度。CLI 批量请求超时默认 600 秒，可通过 `--request-timeout` 调整；超时或中断不自动重试。分片可能已经保存，即使最终失败也不能假设全部回滚。

本次实测目标仍为 `mcp-demo`：创建前节点数为 0；批量响应为 `total=906`、`successCount=906`、`failureCount=0`、`failures=[]`。返回的 `nodeIds` 与请求中的 `params.id` 完全一致；一次整体列表核对确认新增 ID 集合一致，包含 102 个方管梁段、84 个方管柱段、413 个 H 型钢梁段和 307 个槽钢梁段，随后 `cad.document.save` 成功返回。未逐根读取，也未重现此前 380 根时会话中断的问题；长期内存占用及刷新后持久化仍待人工验收。

静态文件未描述批量返回值 Schema；上述返回字段及保留 `params.id` 的行为来自本次实时响应。Python 在后续调用中校验成功/失败数量及 ID 集合，返回结构变化或部分成功时保留原始结果并停止，不擅自假定创建全部成功。请求文件为 `results/webcad/cabin_plan.batch.json`（含 `name` 和 `arguments`），执行记录为 `results/webcad/batch_manifest.json`，包含原请求、响应、整体核对节点及保存响应，均不纳入 Git。45 项相关单元测试通过，未修改结构分析和优化流程。

### 切割与定位实测（2026-10-23）

更新后的《WebCAD MCP 建模指南》共 46 页：第 5 页明确 `cutIdsA` 为本构件被谁切、`cutIdsB` 为本构件切谁；第 32–33 页给出 `engine.cutFace`，目标构件用 `faceIds` 引用切割面，面上用 `cutIds` 引用目标。它提供的是显式切割关系，未说明如何自动识别主次构件和选择连接方案。

新增诊断脚本 `scripts/check_webcad_cutting.py`，不修改正式导出映射、分析或优化。第一轮要求保存文档为空，第二轮只允许第一轮测试节点存在；固定 ID、显式目标文档、运行记录防重，不自动删除或重发写请求。先校验实时 Schema，再以 `skipCutRelations=false` 批量创建，总计 17 根测试梁和 2 个切割面。读取保存文档中的 BREP，通过现有 Gmsh 4.15.0 的 OpenCASCADE 接口测量体积、相交体积和包围盒，不只检查请求参数。

```powershell
# 省略 --confirm-create 仅输出请求预览，不连接平台
.\.venv\Scripts\python.exe scripts\check_webcad_cutting.py --confirm-create --document-id 实际目标文档ID --ca-bundle "docs\WebCAD对接\2.证书安装\ca.crt"
# 第一轮结束后，追加双向引用、切割面和定位补充实验
.\.venv\Scripts\python.exe scripts\check_webcad_cutting.py --confirm-create --follow-up --document-id 实际目标文档ID --ca-bundle "docs\WebCAD对接\2.证书安装\ca.crt"
```

实测引擎版本为 engine 0.1.590 / CAD 0.1.329：

| 测试 | 保存 BREP 的几何结果 | 判定范围 |
| --- | --- | --- |
| 无关系，仅 `skipCutRelations=false` | 正交 H 型梁仍相交 175190.208 mm³（175.190 cm³） | 不会仅凭相交自动消除重叠 |
| 被切梁填写 `cutIdsA=[主梁ID]` | 相交体积为 0，主梁体积不变，被切梁减少对应体积 | 已验证单向 A 引用可执行布尔差 |
| 仅切割梁填写 `cutIdsB=[次梁ID]` | 相交体积未变 | 当前批量路径下不能依赖 B 单向引用 |
| 双方分别填写 `cutIdsA/cutIdsB` | 相交体积为 0 | 双向引用可用 |
| 按指南样例构造切割面，面和梁双向引用 | 梁由 2000 mm 截为 1000 mm，体积减半，保持一个实体 | 已验证这一组切割面方向与保留侧 |
| `baseType=5` | 沿 X 梁的截面定位向 Y 偏移，而非 Z 向贴顶 | 中上是截面局部定位，不能直接等同世界顶面 |
| `crosssectionRotationAngle=90` | 保存的角度仍为 0，截面包围盒未改变 | 当前 BeamNode 批量路径旋转未生效 |
| 显式工作面及通用 `transform` 补充测试 | 未得到预期的截面转向；非单位 transform 的 BREP/显示约定未完全确认 | 不作为生产定位方案 |

H150×100 沿 X 梁的实测外包范围为 Y 向 150 mm、Z 向 100 mm；因此不能直接用名义截面高度推定世界 Z 向尺寸。两种截面在同一标高传 `baseType=5` 后，顶面仍分别为 3050 和 3100 mm。第一轮切割面的输入法向与保存平面法向不一致且未截断；第二轮采用指南中同类起终点/normal 的样例写法后成功。只确认这一组样例，不能把法向语义或双向引用必要性单独归因于本实验，其他方向、保留侧、间隙及一般化接口仍待确认。

特别注意：H 梁实体布尔差后被切梁含 6 个实体，虽然消除了重叠，但不是一根连续加工件。生产阶段应优先按连接面截断或计算端点，必要时按明确的节点设计开缺口；不把“零相交体积”当成可加工验收。按外轮廓反算定位、主次关系选择和加工间隙仍由本地规则决定。

本次用户在页面删除并保存后，`cad.node.list` 仍返回 906 个旧 ID，`get_by_id` 也可返回旧参数；对其中一个已授权的旧 ID 调用删除成功，列表数量仍未变，但 `cad.node.list_faces` 报找不到宿主。保存文档却仅含文件夹，故判断存在列表/参数登记与实际文档不同步的现象，具体缓存机制待平台确认。未继续循环删除旧记录。正式批量导出的列表核对目前只能作为登记一致性检查，不能代替实际几何验收或删除确认。

两轮请求均全部成功，保存文档核对存在 19 个测试节点；其 `hasBackendSave=false`，该字段及保存接口是否真正触发后台落库仍须平台确认，不能仅凭返回序列化文档就承诺刷新后持久化。测试件保留供浏览器查看，后台保存需人工确认。请求、原始文档及测量结果位于 `results/webcad/cut_tests/experiment.json`、`followup.json`，简报为 `report.md`，均不跟踪。相关单元测试共 50 项通过。

## 本地精确定位与端部截断

2026-10-23 新增 `production.py`（布局转换）、`profiles.py`（解析截面）与 `production_export.py`（创建和 BREP 验收），不再依赖平台钢梁旋转和自动切割。原 `cabin/export` 命令保留为原生钢构件的视觉试建；需要外尺寸、面平齐和无贯穿的几何模型时使用独立的 `scripts/webcad_production.py`。

### 几何约定

- 输入必须为未网格化的轴线模型，单位 m；输出统一为 mm，X 为宽度、Y 为长度、Z 为高度。
- 尺寸基准为钢骨架外包面，尚不包含壁板、玻璃、保温层和连接件。主舱长度不包含原模型单独定义的扩展段。
- 同直线、同分组、同截面、同类型及同所属面的连续杆段先合并；不跨优化分组或截面变化合并。各加工件的 `stock_id/source_ids` 可追溯至这一组原始杆段。
- 截面高度沿所属面法向，宽度在面内且与杆轴垂直，两个截面轴与正向杆轴构成右手系。槽钢开口按面中心方向统一；H 型钢及方管保留内外圆角，槽钢保留 1:10 翼缘斜度和圆角。当前 H150×100、方管 150×150×6 和槽钢 20b 的解析面积均与实测原生截面 BREP 一致，方管理论参考圆角的供货确认要求不变。
- 按各自实际截面尺寸向舱内偏移，使顶、底和各侧面的外表面平齐；主舱在与扩展段共享的边界优先定位。不是给所有构件统一减去同一个尺寸。
- 柱优先于同类水平杆，随后按边框、主梁、次梁、支撑排序；同级采用固定的轴向和坐标顺序。低优先构件在高优先构件的外包接触面处计算平直端部，不做空心型钢相减，不生成缺口碎片。
- `gap_mm=0` 表示几何接触，可显式设置正装配间隙；不擅自填入焊缝或加工余量。若整个连续杆会被吃掉、外尺寸/所属面平齐不符或仍有包围盒体积重叠，拒绝导出。
- 靠边的内部排架若因边框内移而放不下，会整排联动调整，而不是静默删杆。当前默认模型的 Y=8400 mm 一排调整为 Y=8250 mm；外尺寸不变，原轴网不再被无条件视为加工定位。

### 平台表示与核对

平台创建类型为 `engine.prism`，草图中显式给出世界坐标、圆弧、内孔及工作面法向，沿杆轴按本地净长拉伸。H 型钢和空心方管试样已实测核对朝向、圆角、内孔和体积。批量接口仅传 `id/sketch/lengthStart/lengthEnd`：本次发现 `engine.prism` 拒绝 `name/userData` 等额外参数，因此不能假称它仍是平台原生钢梁业务节点，截面、分组、源杆和加工长度保存在外部清单中。正式请求设置 `skipCutRelations=true`，平台无需再剪切实体。

导出前使用实际保存文档而不是可能过期的 `cad.node.list` 检查空模型；只允许通过显式记录列表清理本次试样，不按通配符删除。每次请求保留日志，不自动重试。已存在导出记录的目录禁止覆盖原计划和清单；新一轮方案使用新的输出目录。

返回后逐件读取保存文档内的 BREP（不是逐件远程查询），使用现有 Gmsh/OpenCASCADE 验证单实体、体积与解析截面面积×净长一致、包围盒与计划一致，并复核包围盒无重叠；随后对可能接触的实体测量真实最短距离，零间隙模式下要求整个构架连通，不只检查包围盒相接。通用 `transform` 固定为单位阵，不重复变换几何。

### 实测结果

当前主舱参数为 8.5×3.7×3.3 m，扩展段沿负 Y 方向为 1.1 m，整体外包范围为 X=[0,3700]、Y=[-1100,8500]、Z=[0,3300] mm。906 个原始杆段合并为 126 条连续参考杆，截断后为 442 个实体件：边框 40、主梁 33、次梁 111、支撑 258。所有连续杆均保留，没有整条杆被吃掉；最短加工件 50 mm。

已清理记录中的 19 个切割测试节点和 2 个截面试样，完成全部 442 件的创建。BREP 核对全部通过：每件为一个实体，最大包围盒误差约 1e-7 mm，无包围盒体积重叠；926 对实际接触检查全部通过，构架为一个包含 442 件的连通整体，无孤立件。批量创建 11.95 秒，创建/保存响应及逐件几何核对共 38.11 秒，追加接触核对为 18.70 秒。钢件几何理论质量约 11157.98 kg，不含板、玻璃、连接件或损耗，不能与轴线分析中的名义截面质量混为一谈。

保存接口仍返回 `hasBackendSave=false`，所以仅确认前端文档和保存返回的几何，不承诺后台落库或刷新后持久化；需在页面手动保存并由平台确认。结果位于 `results/webcad/production/plan.json`、`request.json`、`cut_list.csv`、`export_manifest.json` 和 `report.md`，不纳入 Git。相关测试共 64 项通过。

### 使用方法

```powershell
# 离线计算，不接触平台；输出目录不能已有导出记录
.\.venv\Scripts\python.exe scripts\webcad_production.py --output-dir results/webcad/new_production
# 向空白目标文档导入；页面必须保持打开
.\.venv\Scripts\python.exe -u scripts\webcad_production.py --confirm-create --document-id 实际文档ID --ca-bundle "docs\WebCAD对接\2.证书安装\ca.crt" --output-dir results/webcad/new_production
# 可选：--gap-mm 1；或 --plan 来自最终优化模型的生产计划.json
```

默认 CLI 仍使用 `export_geometry.py` 的当前示例模型，不冒充最优解。最终优化和 Streamlit 直接调用：

```python
from src.integrations.webcad import (
    ProductionConfig, build_production_plan, export_production_plan,
)

plan = build_production_plan(final_model, ProductionConfig(gap_mm=0))
# 展示 plan["grid_adjustments"]、加工清单与约定，确认后调用
record = export_production_plan(
    plan, client, "results/webcad/final_production/export_manifest.json",
    document_id=target_document_id,
    on_progress=lambda done, total: progress.progress(done / total),
)
```

这仍是几何深化，不代表连接设计和生产图纸已经完成：焊缝、螺栓、节点板、供货圆角及公差需另行确认；门窗参数仍是原轴网名义尺寸，不保证最终净空。原 OpenSees/RFEM 模型未修改，因偏心、实际跨度和末跨排架位置已变化，投入生产前须回传进行结构一致性复核。

## 原生钢构件与 Streamlit 入口

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

已有同路径或同目录同目标模型的记录时拒绝再次导出；CLI 与 UI 应保持目标模型对应的记录目录固定。这不是服务端幂等保证：移动/删除记录或使用其他目录不能防重。首版不提供自动删除或断点续传，须先人工核对并清理本次记录中的构件，不能误删用户原有构件。没有事务或自动回滚承诺。以上逐构件状态适用于串行接口；批量接口保存整包请求/响应和整体节点核对结果，平台本身可能已经分片保存，不能套用“部分失败后未保存”的串行假设。

## 实测状态与待确认

2026-10-16 的联调中，使用提供的 CA 后 TLS 校验通过，但旧 `/dev-api/auth/login` 返回“用户不存在/密码错误”。Windows 系统 `USERNAME` 与本地 `username` 的同名冲突已修正：简写先转换为 `WEBCAD_*`，不读取或覆盖系统登录名。

2026-10-22，平台技术人员调整后，按要求把登录前缀改为 `/local-api`，已实测完成：

- 使用 `2.证书安装/ca_all.crt`，全程开启 TLS 证书校验；原加密算法、公钥、clientId 和 tenantId 可用，账号密码登录成功。
- 获取到唯一编辑会话，目标文档为 `mcp-demo`。
- `cad.node.list_types` 和六种钢梁、钢柱的 `cad.node.describe_type` 均成功，Schema 已保存到 `results/webcad`。方管外圆角、工字钢/槽钢两个圆角均为必填项，不能省略或臆造。
- 实时 Schema 校验后成功创建一根 `structure.squareTubeSteelGirder`。截面为 100×100×5 mm、外圆角 8 mm，起点 `(0,0,3000)`、终点 `(4000,0,3000)` mm，定位 0、旋转 0°；创建响应中的实际参数与请求一致。
- 创建的 `nodeId` 为 `7341552054505472_M09_YH-s`；`cad.document.save` 返回成功；随后独立调用 `cad.node.list` 再次查询到该构件。

会话、类型、Schema、创建记录与只读复核分别位于 `results/webcad/sessions.json`、`types.json`、`structure.*.json`、`trial_manifest.json` 和 `node_list_after_trial.json`，不纳入 Git。以上为首次单构件联调记录；后续整舱中断情况见“完整钢结构试建”，尚未完成完整增压舱的保存和持久化验收。

离线默认模型预览读取到 906 根实际杆件，生成待确认列表。单构件联调时 21 项 WebCAD 测试通过；标准参数补充后共 30 项测试通过，覆盖原两套库全部 H/槽钢匹配、方管参考圆角显式选择、出处保留及分析数据不变。现有分组优化、RFEM 导出与工程布局测试在前次开发时已通过。

仍须联调确认：

1. 在浏览器检查试建实体的可视几何、坐标与截面朝向，并通过刷新检查保存后的持久化；API 成功不等同于这些人工验收已完成。
2. 已验证 `cad.node.get_by_id({"nodeId": "..."})` 可用，但删除后的旧参数仍可能被返回；列表及参数缓存与保存文档不一致的问题仍待平台确认。详见切割与定位实测，不以参数回显代替几何验收。
3. HShape 的 `l` 对应标准 t2 已核对；槽钢 BIM 标准厚度与原分析回退值不完全相同，尚未统一两者，不应宣称实体与分析截面完全相同。
4. 方管圆角采用理论计算参考值还是厂家值、GB/T 3094 冷拔工艺是否为实际供货标准、未列尺寸供货约定尚待确认；默认不启用参考圆角。
5. 本地 `FRAME` 同时包含竖向与水平杆件。暂按竖向 FRAME 为柱、水平为梁生成候选计划，需确认与四角柱业务定义一致；原始杆段也不等同四根完整柱。
6. OpenSees 按轴线方向和所属面定义局部轴；平台零旋转、正角方向、槽钢开口方向及形心定位是否与之对应，必须用样例实测。没有证据前不自动计算平台旋转角。
