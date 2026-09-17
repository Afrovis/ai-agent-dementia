#!/usr/bin/env node
// Generates the committed cross-language crypto vectors (HANDOFF.md section
// 5.4). Run in node:22-slim, from `volunteer/`:
//
//   docker run --rm -v "$PWD":/v -w /v node:22-slim node scripts/make_vector.mjs
//
// Uses WebCrypto through static/crypto.js -- the same code the browser
// ships -- with fixed, non-secret key/IV bytes so the output is
// deterministic and safe to commit. `common/tests/test_vectors.py` decrypts
// both files and asserts the plaintext.

import { createRequire } from "node:module";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const cryptoLib = require(
  path.join(here, "..", "web", "volunteer_web", "static", "crypto.js"),
);
const vectorsDir = path.join(here, "..", "common", "tests", "vectors");

function bytes(length, offset) {
  return Uint8Array.from({ length }, (_, i) => (i + offset) % 256);
}

function toB64(uint8) {
  return Buffer.from(uint8).toString("base64");
}

async function main() {
  const key = bytes(cryptoLib.AES_KEY_BYTES, 0);
  const submissionId = "sub-vector-1";

  // Chunk vector: one chunk, fixed IV, short plaintext.
  const chunkPlaintext = new TextEncoder().encode("hello volunteer chunk vector");
  const chunkN = 2;
  const chunkIv = bytes(cryptoLib.GCM_IV_BYTES, 100);
  const chunkCiphertext = await cryptoLib.encryptChunk(
    key,
    submissionId,
    chunkN,
    chunkPlaintext,
    chunkIv,
  );
  writeFileSync(
    path.join(vectorsDir, "chunk_vector.json"),
    JSON.stringify(
      {
        key: toB64(key),
        submission_id: submissionId,
        n: chunkN,
        plaintext: toB64(chunkPlaintext),
        ciphertext_with_iv: toB64(chunkCiphertext),
      },
      null,
      2,
    ) + "\n",
  );

  // Result vector: plaintext long enough to span two blocks at a tiny
  // block size, so the fixture stays small while still exercising the
  // length-prefix and per-block AAD logic with more than one block.
  const resultPlaintext = new TextEncoder().encode(
    "hello volunteer result vector, spanning two blocks!",
  );
  const blockSize = 32; // 51-byte plaintext -> exactly two blocks (32 + 19)
  let ivCounter = 0;
  const resultStream = await cryptoLib.encryptResultBlocks(
    key,
    submissionId,
    resultPlaintext,
    () => bytes(cryptoLib.GCM_IV_BYTES, 200 + ivCounter++ * cryptoLib.GCM_IV_BYTES),
    blockSize,
  );
  writeFileSync(
    path.join(vectorsDir, "result_vector.json"),
    JSON.stringify(
      {
        key: toB64(key),
        submission_id: submissionId,
        block_size: blockSize,
        plaintext: toB64(resultPlaintext),
        stream: toB64(resultStream),
      },
      null,
      2,
    ) + "\n",
  );

  console.log("wrote chunk_vector.json and result_vector.json");
}

main();
