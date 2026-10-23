import CryptoJS from 'crypto-js';
import crypto from 'node:crypto';

// ---------------------------------------------------------------------------
// 配置（可通过环境变量覆盖）
// ---------------------------------------------------------------------------
const BASE_URL = (process.env.CAD_SCRIPT_BASE_URL ?? 'https://124.223.165.124:4006/').replace(/\/$/, '');
const USERNAME = 'leng123456';
const PASSWORD = 'leng123456';
const CLIENT_ID = 'e5cd7e4891bf95d1d19206ce24a7b32e';
const TENANT_ID = '000000';
const PUBLIC_KEY = `-----BEGIN RSA PUBLIC KEY-----\n${'MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC/XwRpoHnvtFxR5cSzw0m220gpyydQ0fO/u2NqajTsa8A3+nm6LHAufv5rxLWafb3x7OYUnMv/5Llief/tZHNrf7ElQYRtHLk21C/gIJUkwb1F2cns6E3IKoTJOdBUbPyTy0zFeexxznCkIqmy53zLw4yJFYGK53h0OZdrUbq/ZwIDAQAB'}\n-----END RSA PUBLIC KEY-----`;
const NODE_TYPE = 'structure.tConcretePillar';

// 节点创建默认参数
const NODE_CREATE_PARAMS = {
  startPoint: { x: 0, y: 0, z: 0 },
  endPoint: { x: 0, y: 0, z: 3000 },
  lengthB1: 200,
  lengthB2: 200,
  lengthB3: 200,
  lengthB4: 200,
  lengthA1: 400,
  lengthA2: 200,
  lengthA3: 400,
  baseType: 0,
  concreteType: 'ConcretePillar',
  crosssectionRotationAngle: 0
};

// TLS 自签证书时跳过验证
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

// ---------------------------------------------------------------------------
// HTTP 工具
// ---------------------------------------------------------------------------

const log = {
  info: (msg) => console.log(`\x1b[36m[INFO]\x1b[0m ${msg}`),
  ok: (msg) => console.log(`\x1b[32m[OK]\x1b[0m   ${msg}`),
  err: (msg) => console.error(`\x1b[31m[ERR]\x1b[0m  ${msg}`),
  step: (n, msg) => console.log(`\n\x1b[1m\x1b[33m[${n}]\x1b[0m ${msg}`)
};

const http = async (url, { method = 'GET', headers = {}, body } = {}) => {
  const opts = {
    method,
    headers: { 'content-type': 'application/json', ...headers }
  };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(url, opts);
  const text = await resp.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Non-JSON response (${resp.status}): ${text.slice(0, 200)}`);
  }
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${data?.message ?? data?.error ?? text.slice(0, 200)}`);
  }
  return data;
};

// ---------------------------------------------------------------------------
// 步骤
// ---------------------------------------------------------------------------

// 1. 登录
const stepLogin = async () => {
  log.step(1, `登录接口 POST ${BASE_URL}/dev-api/auth/login`);
  const encryptedPassword = encryptPassword(PASSWORD, PUBLIC_KEY);
  const resp = await http(`${BASE_URL}/dev-api/auth/login`, {
    method: 'POST',
    body: {
      tenantId: TENANT_ID,
      username: USERNAME,
      password: encryptedPassword,
      rememberMe: true,
      clientId: CLIENT_ID,
      grantType: 'password'
    }
  });
  if (!resp?.data?.access_token && !resp?.data?.accessToken) {
    throw new Error(`登录响应缺少 access_token，完整响应: ${JSON.stringify(resp).slice(0, 500)}`);
  }
  const accessToken = String(resp.data.access_token ?? resp.data.accessToken);
  log.ok(`登录成功，accessToken 前 20 位: ${accessToken.slice(0, 20)}...`);
  return accessToken;
  log.ok(`登录成功，accessToken 前 20 位: ${String(resp.data.accessToken).slice(0, 20)}...`);
  return String(resp.data.accessToken);
};

// 2. 获取 session
const stepGetSession = async (accessToken) => {
  log.step(2, `获取 session 列表 GET ${BASE_URL}/api/cad-agent/sessions`);
  const authHeaders = { Authorization: `Bearer ${accessToken}` };
  const resp = await http(`${BASE_URL}/api/cad-agent/sessions`, { headers: authHeaders });
  const sessions = resp?.sessions;
  if (!Array.isArray(sessions) || sessions.length === 0) {
    throw new Error('没有可用的 CAD Agent session（请先在浏览器中打开 WebCAD 页面并建立连接）');
  }
  const lastSession = sessions[sessions.length - 1];
  const sessionId = lastSession.id;
  log.ok(`获取到 ${sessions.length} 个 session，使用最后一个: ${sessionId}`);
  return { sessionId, authHeaders };
};

// 3. 获取节点类型列表
const stepListNodeTypes = async ({ sessionId, authHeaders }) => {
  log.step(3, `获取节点类型列表 POST ${BASE_URL}/api/cad-agent/sessions/${sessionId}/calls`);
  const resp = await http(`${BASE_URL}/api/cad-agent/sessions/${encodeURIComponent(sessionId)}/calls`, {
    method: 'POST',
    headers: authHeaders,
    body: { tool: 'cad.node.list_types', args: {} }
  });
  if (resp?.ok !== true) {
    throw new Error(`获取节点类型失败: ${JSON.stringify(resp)}`);
  }
  const types = resp?.value ?? [];
  log.ok(`获取到 ${Array.isArray(types) ? types.length : 0} 种节点类型`);
  if (Array.isArray(types) && types.length > 0) {
    log.info(`前 5 种类型: ${types.slice(0, 5).map((t) => t.type ?? t).join(', ')}`);
  }
  return { nodeTypes: types, sessionId, authHeaders };
};

// 4. 获取类型 schema
const stepDescribeType = async ({ sessionId, authHeaders, nodeType }) => {
  log.step(4, `获取类型 schema POST ${BASE_URL}/api/cad-agent/sessions/${sessionId}/calls`);
  const resp = await http(`${BASE_URL}/api/cad-agent/sessions/${encodeURIComponent(sessionId)}/calls`, {
    method: 'POST',
    headers: authHeaders,
    body: { tool: 'cad.node.describe_type', args: { type: nodeType } }
  });
  if (resp?.ok !== true) {
    throw new Error(`获取类型 schema 失败 (type=${nodeType}): ${JSON.stringify(resp)}`);
  }
  const schema = resp?.value;
  log.ok(`获取到 ${nodeType} 的 schema`);
  if (schema?.create) {
    log.info(`create schema 字段: ${Object.keys(schema.create.properties ?? {}).join(', ')}`);
  }
  return { schema, sessionId, authHeaders };
};

// 5. 创建节点
const stepCreateNode = async ({ sessionId, authHeaders, nodeType }) => {
  log.step(5, `创建节点 POST ${BASE_URL}/api/cad-agent/sessions/${sessionId}/calls`);
  const resp = await http(`${BASE_URL}/api/cad-agent/sessions/${encodeURIComponent(sessionId)}/calls`, {
    method: 'POST',
    headers: authHeaders,
    body: { tool: 'cad.node.create', args: { type: nodeType, params: NODE_CREATE_PARAMS } }
  });
  if (resp?.ok !== true) {
    throw new Error(`创建节点失败: ${JSON.stringify(resp)}`);
  }
  log.ok(`节点 ${nodeType} 创建成功`);
  return resp;
};

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------

const main = async () => {
  console.log('\n' + '='.repeat(60));
  console.log('  CAD Agent 自动化脚本');
  console.log(`  Base URL : ${BASE_URL}`);
  console.log(`  Username : ${USERNAME}`);
  console.log(`  Node Type: ${NODE_TYPE}`);
  console.log('='.repeat(60));

  try {
    // 1
    const accessToken = await stepLogin();

    // 2
    const { sessionId, authHeaders } = await stepGetSession(accessToken);

    // 3
    const { nodeTypes } = await stepListNodeTypes({ sessionId, authHeaders });

    // 校验目标类型是否存在
    const typeExists = Array.isArray(nodeTypes) && nodeTypes.some((t) => (t.type ?? t) === NODE_TYPE);
    if (!typeExists) {
      log.err(`目标类型 ${NODE_TYPE} 不在节点类型列表中，跳过后续步骤`);
      process.exit(1);
    }

    // 4
    await stepDescribeType({ sessionId, authHeaders, nodeType: NODE_TYPE });

    // 5
    await stepCreateNode({ sessionId, authHeaders, nodeType: NODE_TYPE });

    console.log('\n' + '='.repeat(60));
    console.log('  全部步骤完成');
    console.log('='.repeat(60) + '\n');
  } catch (error) {
    log.err(error.message);
    console.error(error);
    process.exit(1);
  }
};

// ---------------------------------------------------------------------------
// 密码加密工具函数
// ---------------------------------------------------------------------------

/**
 * 随机生成 32 位字符串
 */
const generateRandomString = () => {
  // 使用 node:crypto 替代 Math.random，更安全
  const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const bytes = crypto.randomBytes(32);
  let result = '';
  for (let i = 0; i < 32; i++) {
    result += characters.charAt(bytes[i] % characters.length);
  }
  return result;
};

/**
 * 随机生成 AES 密钥（CryptoJS WordArray）
 */
const generateAesKey = () => {
  return CryptoJS.enc.Utf8.parse(generateRandomString());
};

/**
 * AES-ECB 加密（PKCS7 填充）
 */
const encryptWithAes = (message, aesKey) => {
  const encrypted = CryptoJS.AES.encrypt(message, aesKey, {
    mode: CryptoJS.mode.ECB,
    padding: CryptoJS.pad.Pkcs7
  });
  return encrypted.toString();
};

/**
 * CryptoJS WordArray 转 Base64 字符串
 */
const encryptBase64 = (wordArray) => {
  return CryptoJS.enc.Base64.stringify(wordArray);
};

/**
 * 用 node:crypto 自动将 PKCS#1 公钥转换为可用格式
 */
const parsePublicKey = (publicKeyPem) => {
  // 先尝试直接导入（PKCS#8 / SPKI 格式）
  try {
    return crypto.createPublicKey(publicKeyPem);
  } catch {
    // PKCS#1 格式需要转换：去掉头尾并改为 PKCS#8 PEM
    const body = publicKeyPem
      .replace(/-----(BEGIN|END) RSA PUBLIC KEY-----/g, '')
      .replace(/\s/g, '');
    const pkcs8Pem =
      '-----BEGIN PUBLIC KEY-----\n' +
      body.match(/.{1,64}/g).join('\n') +
      '\n-----END PUBLIC KEY-----';
    return crypto.createPublicKey(pkcs8Pem);
  }
};

/**
 * RSA 公钥加密（PKCS1v15 填充，与前端 JSEncrypt 行为一致）
 */
const rsaEncrypt = (plainText, publicKeyPem) => {
  const key = parsePublicKey(publicKeyPem);
  const buffer = Buffer.from(plainText, 'utf8');
  const encrypted = crypto.publicEncrypt(
    {
      key,
      padding: crypto.constants.RSA_PKCS1_PADDING
    },
    buffer
  );
  return encrypted.toString('base64');
};

/**
 * 加密密码（与前端登录加密逻辑一致）
 *  1. 随机生成 AES 密钥
 *  2. 用 AES-ECB 加密明文密码
 *  3. 用 RSA 公钥加密 AES 密钥（先 Base64 编码）
 *  4. 返回 `{RSA加密的AES密钥}|{AES加密的密码}`
 */
const encryptPassword = (password, publicKey) => {
  const aesKey = generateAesKey();
  const aesEncrypted = encryptWithAes(password, aesKey);
  const encryptedAesKey = rsaEncrypt(encryptBase64(aesKey), publicKey);
  return `${encryptedAesKey}|${aesEncrypted}`;
};

// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------

main();
