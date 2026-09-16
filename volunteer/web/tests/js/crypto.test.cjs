// `node --test web/tests/js` (HANDOFF.md section 8). Exercises
// `static/crypto.js` directly, with no browser and no Python process:
// self round trips, tamper detection, and a fixed Python-encrypted chunk
// fixture (generated once with volunteer_common.crypto.encrypt_chunk) to
// confirm the two languages agree on the wire format.

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const cryptoLib = require(
  path.join(__dirname, "..", "..", "volunteer_web", "static", "crypto.js"),
);

function b64ToBytes(b64) {
  return new Uint8Array(Buffer.from(b64, "base64"));
}

function bytes(length, offset) {
  return Uint8Array.from({ length }, (_, i) => (i + offset) % 256);
}

test("chunk round trip with crypto.js alone", async () => {
  const key = crypto.getRandomValues(new Uint8Array(cryptoLib.AES_KEY_BYTES));
  const iv = crypto.getRandomValues(new Uint8Array(cryptoLib.GCM_IV_BYTES));
  const plaintext = new TextEncoder().encode("a browser-recorded chunk");
  const encrypted = await cryptoLib.encryptChunk(key, "sub-js", 0, plaintext, iv);
  const decrypted = await cryptoLib.decryptChunk(key, "sub-js", 0, encrypted);
  assert.deepEqual(decrypted, plaintext);
});

test("chunk decryption fails when the AAD (submission id) does not match", async () => {
  const key = crypto.getRandomValues(new Uint8Array(cryptoLib.AES_KEY_BYTES));
  const iv = crypto.getRandomValues(new Uint8Array(cryptoLib.GCM_IV_BYTES));
  const encrypted = await cryptoLib.encryptChunk(
    key,
    "sub-js",
    0,
    new TextEncoder().encode("payload"),
    iv,
  );
  await assert.rejects(() => cryptoLib.decryptChunk(key, "sub-js-other", 0, encrypted));
});

test("result stream round trip across two blocks", async () => {
  const key = crypto.getRandomValues(new Uint8Array(cryptoLib.AES_KEY_BYTES));
  const plaintext = new TextEncoder().encode(
    "a plaintext long enough to span more than one small block",
  );
  let counter = 0;
  const stream = await cryptoLib.encryptResultBlocks(
    key,
    "sub-js",
    plaintext,
    () => bytes(cryptoLib.GCM_IV_BYTES, counter++ * 7),
    16,
  );
  const decrypted = await cryptoLib.decryptResultStream(key, "sub-js", stream);
  assert.deepEqual(decrypted, plaintext);
});

test("a truncated result stream fails to decrypt instead of playing short", async () => {
  const key = crypto.getRandomValues(new Uint8Array(cryptoLib.AES_KEY_BYTES));
  const plaintext = new TextEncoder().encode("some plaintext bytes for a result block");
  let counter = 0;
  const stream = await cryptoLib.encryptResultBlocks(
    key,
    "sub-js",
    plaintext,
    () => bytes(cryptoLib.GCM_IV_BYTES, counter++ * 7),
    16,
  );
  const truncated = stream.slice(0, stream.length - 4);
  await assert.rejects(() => cryptoLib.decryptResultStream(key, "sub-js", truncated));
});

test("decrypts a chunk that volunteer_common.crypto encrypted in Python", async () => {
  // Fixture made with: crypto.encrypt_chunk(bytes(range(32)), "py-fixture-1",
  // 7, b"python-encrypted chunk for the js test", iv=bytes(range(50, 62)))
  const key = b64ToBytes("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=");
  const ciphertextWithIv = b64ToBytes(
    "MjM0NTY3ODk6Ozw9tiKbqKpSrIYDZJj/K5manEbQc/TnJoeNvmgr7N9WuxhQIL4uZJe0AxTCkOqWvWJMqHCnHPfQ",
  );
  const plaintext = await cryptoLib.decryptChunk(key, "py-fixture-1", 7, ciphertextWithIv);
  assert.equal(new TextDecoder().decode(plaintext), "python-encrypted chunk for the js test");
});
