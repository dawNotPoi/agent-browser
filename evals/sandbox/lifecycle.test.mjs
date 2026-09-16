import test from 'node:test';
import assert from 'node:assert/strict';
import { executeGuest } from './lifecycle.mjs';

test('cancellation waits for guest cleanup before artifact collection can start', async () => {
  const events = [];
  let attempts = 0;
  const command = {
    async wait() { events.push('wait'); if (++attempts === 1) throw new Error('cancelled'); return { exitCode: 130 }; },
    async kill(signal) { events.push(signal); },
  };
  await assert.rejects(executeGuest({ async runCommand() { return command; } }, {}, new AbortController().signal), /cancelled/);
  assert.deepEqual(events, ['wait', 'SIGTERM', 'wait']);
});

test('unresponsive cleanup is bounded and force-killed', async () => {
  const events = [];
  const command = { async wait() { throw new Error('timeout'); }, async kill(signal) { events.push(signal); } };
  await assert.rejects(executeGuest({ async runCommand() { return command; } }, {}, new AbortController().signal), /timeout/);
  assert.deepEqual(events, ['SIGTERM', 'SIGKILL']);
});
