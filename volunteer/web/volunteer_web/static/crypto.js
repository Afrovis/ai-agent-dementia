// Wire format for volunteer clip encryption (HANDOFF.md section 5.4).
// Mirrors volunteer/common/volunteer_common/crypto.py exactly: same AAD
// strings, same IV/tag layout, same length-prefixed result stream. No
// import/export syntax, so this file works both as a plain browser
// <script> (globals below) and as a CommonJS module for `node --test`.

const AES_KEY_BYTES = 32;
const GCM_IV_BYTES = 12;
const RESULT_BLOCK_BYTES = 4 * 1024 * 1024;
const LENGTH_PREFIX_BYTES = 4;

function chunkAad(submissionId, n) {
  return new TextEncoder().encode(`chunk:${submissionId}:${n}`);
}

function resultBlockAad(submissionId, index, isLast) {
  return new TextEncoder().encode(`result:${submissionId}:${index}:${isLast ? 1 : 0}`);
}

function generateSessionSecrets() {
  return {
    k: crypto.getRandomValues(new Uint8Array(AES_KEY_BYTES)),
    d: crypto.getRandomValues(new Uint8Array(AES_KEY_BYTES)),
  };
}

async function importAesKey(rawKey) {
  return crypto.subtle.importKey("raw", rawKey, "AES-GCM", false, ["encrypt", "decrypt"]);
}

async function importPublicKeySpki(derBytes) {
  return crypto.subtle.importKey(
    "spki",
    derBytes,
    { name: "RSA-OAEP", hash: "SHA-256" },
    false,
    ["encrypt"],
  );
}

async function wrapKey(publicKey, k) {
  const wrapped = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, publicKey, k);
  return new Uint8Array(wrapped);
}

// `iv` is always caller-supplied (never generated inside this function), so
// it matches volunteer_common.crypto.encrypt_chunk's signature exactly and
// the same call can build either a real upload (random IV) or a fixed test
// vector (deterministic IV).
async function encryptChunk(k, submissionId, n, plaintext, iv) {
  const key = await importAesKey(k);
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv, additionalData: chunkAad(submissionId, n) },
    key,
    plaintext,
  );
  const out = new Uint8Array(GCM_IV_BYTES + ciphertext.byteLength);
  out.set(iv, 0);
  out.set(new Uint8Array(ciphertext), GCM_IV_BYTES);
  return out;
}

async function decryptChunk(k, submissionId, n, ciphertextWithIv) {
  const key = await importAesKey(k);
  const iv = ciphertextWithIv.slice(0, GCM_IV_BYTES);
  const ciphertext = ciphertextWithIv.slice(GCM_IV_BYTES);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv, additionalData: chunkAad(submissionId, n) },
    key,
    ciphertext,
  );
  return new Uint8Array(plaintext);
}

function parseResultBlocks(data) {
  const blocks = [];
  let offset = 0;
  let index = 0;
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  while (offset < data.length) {
    if (offset + LENGTH_PREFIX_BYTES > data.length) {
      throw new Error("truncated result stream: missing length prefix");
    }
    const length = view.getUint32(offset, false);
    offset += LENGTH_PREFIX_BYTES;
    const bodyLen = GCM_IV_BYTES + length;
    if (offset + bodyLen > data.length) {
      throw new Error("truncated result stream: missing block body");
    }
    const body = data.slice(offset, offset + bodyLen);
    offset += bodyLen;
    const isLast = offset >= data.length;
    blocks.push({ index, isLast, ciphertextWithIv: body });
    index += 1;
  }
  return blocks;
}

async function decryptResultStream(k, submissionId, data) {
  const key = await importAesKey(k);
  const blocks = parseResultBlocks(data);
  const parts = [];
  for (const block of blocks) {
    const iv = block.ciphertextWithIv.slice(0, GCM_IV_BYTES);
    const ciphertext = block.ciphertextWithIv.slice(GCM_IV_BYTES);
    const plaintext = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv, additionalData: resultBlockAad(submissionId, block.index, block.isLast) },
      key,
      ciphertext,
    );
    parts.push(new Uint8Array(plaintext));
  }
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

// Only used to build the committed test vectors (scripts/make_vector.mjs).
// Production JS never encrypts a result -- that happens in the worker.
async function encryptResultBlocks(k, submissionId, plaintext, ivSource, blockSize) {
  blockSize = blockSize || RESULT_BLOCK_BYTES;
  const key = await importAesKey(k);
  const total = plaintext.length;
  const blockCount = Math.max(1, Math.ceil(total / blockSize));
  const parts = [];
  for (let index = 0; index < blockCount; index += 1) {
    const start = index * blockSize;
    const end = Math.min(start + blockSize, total);
    const blockPlaintext = plaintext.slice(start, end);
    const isLast = index === blockCount - 1;
    const iv = ivSource();
    const ciphertext = new Uint8Array(
      await crypto.subtle.encrypt(
        { name: "AES-GCM", iv, additionalData: resultBlockAad(submissionId, index, isLast) },
        key,
        blockPlaintext,
      ),
    );
    const lengthPrefix = new Uint8Array(LENGTH_PREFIX_BYTES);
    new DataView(lengthPrefix.buffer).setUint32(0, ciphertext.byteLength, false);
    parts.push(lengthPrefix, iv, ciphertext);
  }
  const total_len = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total_len);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

const api = {
  AES_KEY_BYTES,
  GCM_IV_BYTES,
  RESULT_BLOCK_BYTES,
  LENGTH_PREFIX_BYTES,
  chunkAad,
  resultBlockAad,
  generateSessionSecrets,
  importPublicKeySpki,
  wrapKey,
  encryptChunk,
  decryptChunk,
  parseResultBlocks,
  decryptResultStream,
  encryptResultBlocks,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = api;
}
if (typeof window !== "undefined") {
  window.VolunteerCrypto = api;
}
