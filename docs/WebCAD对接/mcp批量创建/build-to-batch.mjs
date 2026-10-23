#!/usr/bin/env node
/**
 * 把 cabin_plan.json 的 members 转成 WebCAD「批量创建」入参 JSON：{ nodeList, components }。
 *
 * 目标格式与引擎批量管线一致（IBatchRequest，见
 * src/engine/packages/ui/src/nodeFactory/nodeFactoryOfflineCreate.ts:6-18）：
 *   { nodeList: [{ type, params }], components?: [...] }
 * 该 JSON 可直接粘进「节点工厂 → 批量创建」面板，或作为第三方导入的输入，
 * 底层走 NodeFactoryApi.createNodesBatchAsync（分批 200 + 批间让出主线程 + 关 history），
 * 而不是 906 次 cad.node.create。
 *
 * 用法：
 *   node scripts/build-cabin-batch.mjs                        # -> cabin_plan.batch.json
 *   node scripts/build-cabin-batch.mjs --out=xxx.json
 *   node scripts/build-cabin-batch.mjs --split=200            # 额外切分成分批文件
 *   node scripts/build-cabin-batch.mjs --in=other.json
 *
 * 关于 shapeReuseKey（重要，勿"优化"）：
 *   这 4 种类型的几何是**按世界坐标**建模的（descriptor 把 startPoint/endPoint 原样写进
 *   sketchData，transform 为单位阵，见 plugin-structure/nodeDescriptors/steelGirderDescriptors.ts:293-333
 *   与 plugin-common/bodys/beamCreate.ts:14-44）。引擎约定"同 shapeReuseKey ⇒ 同本地几何"，
 *   所以只有 (type, 截面, 起点, 终点) 完全相同的构件才能共享 key，否则克隆件会跑到源件的位置上。
 *   本脚本因此只对"完全重复"的构件派生 key（cabin_plan.json 实测 0 组重复）。
 */

import { readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

// ---------------------------------------------------------------------------
// 类型表：镜像 steelGirderDescriptors.ts / steelPillarDescriptors.ts 的 sectionKeys
// ---------------------------------------------------------------------------

/** @type {Record<string, { label: string, componentCode: string, sectionKeys: string[], positionDependentGeometry: boolean }>} */
const NODE_TYPES = {
  "structure.squareTubeSteelGirder": {
    label: "方管梁",
    componentCode: "15-001-0001",
    sectionKeys: [
      "sideLengthA",
      "sideLengthB",
      "outerArcRadiusR",
      "wallThicknessT",
    ],
    positionDependentGeometry: true,
  },
  "structure.channelSteelGirder": {
    label: "槽钢梁",
    componentCode: "15-001-0002",
    sectionKeys: [
      "legWidthB",
      "height",
      "radiusR",
      "legArcRadiusR1",
      "thicknessD",
      "thicknessT",
    ],
    positionDependentGeometry: true,
  },
  "structure.hShapedSteelGirder": {
    label: "H 型钢梁",
    componentCode: "15-001-0004",
    sectionKeys: ["widthB", "height", "radius", "thicknessT1", "thicknessT2"],
    positionDependentGeometry: true,
  },
  "structure.squareTubeSteelPillar": {
    label: "方管柱",
    componentCode: "15-002-0001",
    sectionKeys: [
      "sideLengthA",
      "sideLengthB",
      "outerArcRadiusR",
      "wallThicknessT",
    ],
    positionDependentGeometry: true,
  },
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const parseArgs = (argv) => {
  const options = {
    in: "cabin_plan.json",
    out: "cabin_plan.batch.json",
    split: 0,
    idPrefix: "cabin-",
  };
  for (const arg of argv) {
    const match = /^--([a-zA-Z-]+)=(.*)$/.exec(arg);
    if (!match) continue;
    const [, key, value] = match;
    if (key === "in") options.in = value;
    else if (key === "out") options.out = value;
    else if (key === "split")
      options.split = Math.max(0, Math.floor(Number(value) || 0));
    else if (key === "id-prefix") options.idPrefix = value;
  }
  return options;
};

const options = parseArgs(process.argv.slice(2));
const inputPath = resolve(process.cwd(), options.in);
const outputPath = resolve(process.cwd(), options.out);

// ---------------------------------------------------------------------------
// 校验
// ---------------------------------------------------------------------------

const isFinitePoint = (value) =>
  !!value &&
  ["x", "y", "z"].every((axis) => Number.isFinite(Number(value[axis])));

const samePoint = (a, b) => a.x === b.x && a.y === b.y && a.z === b.z;

const round = (value, digits = 6) => Number(Number(value).toFixed(digits));

/** 几何签名：仅用于统计"有多少种不同几何"，不作为 shapeReuseKey 使用。 */
const geometrySignature = (type, params) => {
  const { startPoint: s, endPoint: e, sectionParams } = params;
  const length = Math.hypot(e.x - s.x, e.y - s.y, e.z - s.z);
  const section = Object.keys(sectionParams)
    .sort()
    .map((key) => `${key}=${round(sectionParams[key])}`)
    .join(",");
  return `${type}|${section}|L=${round(length, 3)}`;
};

/** 完全重复签名：同 type + 截面 + baseType + 起点 + 终点 ⇒ 本地几何完全一致，可共享 shapeReuseKey。 */
const duplicateSignature = (type, params) => {
  const { startPoint: s, endPoint: e, sectionParams, baseType } = params;
  const section = Object.keys(sectionParams)
    .sort()
    .map((key) => `${key}=${round(sectionParams[key])}`)
    .join(",");
  return JSON.stringify([
    type,
    section,
    round(baseType ?? 0),
    [s.x, s.y, s.z],
    [e.x, e.y, e.z],
  ]);
};

// ---------------------------------------------------------------------------
// 转换
// ---------------------------------------------------------------------------

const raw = JSON.parse(readFileSync(inputPath, "utf8"));
const members = Array.isArray(raw)
  ? raw
  : (raw.members ?? raw.nodeList ?? raw.records ?? []);
if (!Array.isArray(members) || members.length === 0) {
  console.error(`未在 ${options.in} 中找到 members / nodeList / records 数组`);
  process.exit(1);
}

const errors = [];
const warnings = [];
/** @type {Array<{ type: string, params: Record<string, unknown> }>} */
const nodeList = [];
const typeCounts = {};
const geometryCounts = new Map();
const duplicateGroups = new Map();
const seenIds = new Set();

members.forEach((member, index) => {
  const where = `members[${index}] (source_id=${member?.source_id ?? "?"})`;
  const type = member?.type;
  const definition = NODE_TYPES[type];
  if (!definition) {
    errors.push(`${where}: 不支持的 type=${type}`);
    return;
  }
  const params = member?.params;
  if (!params || typeof params !== "object") {
    errors.push(`${where}: 缺少 params`);
    return;
  }

  const { startPoint, endPoint, sectionParams } = params;
  if (!isFinitePoint(startPoint)) {
    errors.push(`${where}: startPoint 不是有限 x/y/z`);
    return;
  }
  if (!isFinitePoint(endPoint)) {
    errors.push(`${where}: endPoint 不是有限 x/y/z`);
    return;
  }
  if (samePoint(startPoint, endPoint)) {
    errors.push(`${where}: startPoint 与 endPoint 重合，descriptor 会拒绝创建`);
    return;
  }
  if (!sectionParams || typeof sectionParams !== "object") {
    errors.push(`${where}: 缺少 sectionParams`);
    return;
  }
  for (const key of definition.sectionKeys) {
    const value = Number(sectionParams[key]);
    if (!Number.isFinite(value) || value <= 0) {
      errors.push(
        `${where}: sectionParams.${key} 必须是正有限数（当前 ${JSON.stringify(sectionParams[key])}）`,
      );
    }
  }

  // 业务编码一致性（不阻断，只提示：descriptor 的 businessQuery 就是这套编号）
  if (
    params.componentCode &&
    params.componentCode !== definition.componentCode
  ) {
    warnings.push(
      `${where}: componentCode=${params.componentCode} 与 ${type} 期望的 ${definition.componentCode} 不一致`,
    );
  }

  const id = `${options.idPrefix}${member.source_id ?? index + 1}`;
  if (seenIds.has(id)) {
    errors.push(`${where}: 生成的 id 重复：${id}（请检查 source_id）`);
  }
  seenIds.add(id);

  const filledSection = {};
  for (const key of definition.sectionKeys)
    filledSection[key] = Number(sectionParams[key]);

  const outParams = {
    id,
    startPoint: {
      x: Number(startPoint.x),
      y: Number(startPoint.y),
      z: Number(startPoint.z),
    },
    endPoint: {
      x: Number(endPoint.x),
      y: Number(endPoint.y),
      z: Number(endPoint.z),
    },
    sectionParams: filledSection,
    baseType: Number(params.baseType ?? 0),
    ...(params.crosssectionRotationAngle !== undefined
      ? { crosssectionRotationAngle: Number(params.crosssectionRotationAngle) }
      : {}),
    // 保留业务编码（批量管线会把它们当 businessProperties 合并到节点上）
    ...(params.typeCode ? { typeCode: params.typeCode } : {}),
    ...(params.componentCode ? { componentCode: params.componentCode } : {}),
    // 可读名称 / 溯源信息
    compentName: definition.label,
    ...(member.section_name ? { detailName: String(member.section_name) } : {}),
    userData: {
      originId: id,
      ...(member.group_id ? { groupId: member.group_id } : {}),
      ...(member.beam_type ? { beamType: member.beam_type } : {}),
      ...(member.face ? { face: member.face } : {}),
      ...(member.section_name ? { sectionName: member.section_name } : {}),
    },
  };

  const signature = duplicateSignature(type, outParams);
  const bucket = duplicateGroups.get(signature);
  if (bucket) {
    bucket.count += 1;
    outParams.shapeReuseKey = bucket.key; // 与首件几何完全一致，可安全复用
  } else {
    duplicateGroups.set(signature, {
      count: 1,
      key: `cabin-dup-${duplicateGroups.size + 1}`,
    });
  }

  nodeList.push({ type, params: outParams });
  typeCounts[type] = (typeCounts[type] ?? 0) + 1;
  const geo = geometrySignature(type, outParams);
  geometryCounts.set(geo, (geometryCounts.get(geo) ?? 0) + 1);
});

if (errors.length) {
  console.error(`校验失败，共 ${errors.length} 条：`);
  for (const error of errors.slice(0, 50)) console.error(`  - ${error}`);
  if (errors.length > 50)
    console.error(`  ... 其余 ${errors.length - 50} 条省略`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// 输出
// ---------------------------------------------------------------------------

const payload = { components: [], nodeList };
writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");

const parts = [];
if (options.split > 0) {
  const total = Math.ceil(nodeList.length / options.split);
  const dir = dirname(outputPath);
  const base = basename(outputPath).replace(/\.json$/i, "");
  for (let i = 0; i < total; i += 1) {
    const chunk = nodeList.slice(i * options.split, (i + 1) * options.split);
    const partPath = join(
      dir,
      `${base}.part-${String(i + 1).padStart(2, "0")}of${String(total).padStart(2, "0")}.json`,
    );
    writeFileSync(
      partPath,
      `${JSON.stringify({ components: [], nodeList: chunk }, null, 2)}\n`,
      "utf8",
    );
    parts.push({ path: partPath, count: chunk.length });
  }
}

// ---------------------------------------------------------------------------
// 报告
// ---------------------------------------------------------------------------

const dupGroupCount = [...duplicateGroups.values()].filter(
  (group) => group.count > 1,
).length;
const dupMemberCount = [...duplicateGroups.values()]
  .filter((group) => group.count > 1)
  .reduce((sum, group) => sum + group.count, 0);
const nonZeroRotation = nodeList.filter(
  (node) => Number(node.params.crosssectionRotationAngle) !== 0,
).length;
const missingMaterial = nodeList.filter(
  (node) =>
    !node.params.materialId && !node.params.businessProperties?.materialId,
).length;

console.log(
  `输入: ${options.in}  (units=${raw.units ?? "未声明"}, test_only=${raw.test_only ?? "未声明"}, warnings=${raw.warnings?.length ?? 0})`,
);
console.log(`输出: ${options.out}  nodeList=${nodeList.length}  components=0`);
for (const [type, count] of Object.entries(typeCounts)) {
  console.log(`  - ${NODE_TYPES[type].label.padEnd(6, " ")} ${type}  ${count}`);
}
console.log(`几何签名（type+截面+长度）: ${geometryCounts.size} 种`);
console.log(`完全重复构件: ${dupMemberCount} 条 / ${dupGroupCount} 组`);
if (parts.length) {
  console.log(`分批文件 (每批 ${options.split}):`);
  for (const part of parts) console.log(`  - ${part.path}  (${part.count})`);
}

const notes = [
  "shapeReuseKey 只写给「type+截面+起点+终点」完全相同的构件：这 4 种类型的几何按世界坐标建模（transform 为单位阵），" +
    "跨位置复用会让克隆件出现在源件位置。本次数据无完全重复构件，因此未写入任何 key。",
  "crosssectionRotationAngle 全部为 " +
    `${nonZeroRotation === 0 ? "0" : "非 0"}` +
    "；注意 BeamNode 路径当前并不应用该字段（steelGirderDescriptors.ts:365-368 只在非 Beam 分支赋值），" +
    "非 0 值会被静默忽略。",
  missingMaterial === nodeList.length
    ? "所有构件未携带 materialId / lineMaterialId：导入后使用文档默认材质与线材质，需要分材质时在导入后统一赋值。"
    : `${nodeList.length - missingMaterial} 条构件携带了材质，其余使用文档默认材质。`,
];
if (raw.test_only === true)
  notes.push(
    "cabin_plan.json 标记 test_only=true，导入前请确认这批数据可用于目标模型。",
  );

console.log("提示:");
for (const note of notes) console.log(`  - ${note}`);
console.log(
  [
    "",
    "下一步：打开 mcp-demo → 节点工厂「批量创建」面板粘贴该 JSON（或走第三方导入）→",
    parts.length
      ? "按 part 顺序逐批导入，每批完成后先保存，再导入下一批。"
      : "导入完成后执行一次保存。",
  ].join("\n"),
);
