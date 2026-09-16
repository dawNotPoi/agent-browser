/** Stop the guest gracefully before collecting its artifacts on cancellation. */
export async function executeGuest(sandbox, parameters, signal) {
  const command = await sandbox.runCommand({ ...parameters, detached: true, signal });
  try {
    return await command.wait({ signal });
  } catch (error) {
    try {
      await command.kill('SIGTERM');
      await command.wait({ signal: AbortSignal.timeout(45_000) });
    } catch {
      try { await command.kill('SIGKILL'); } catch { /* VM shutdown is the final bound. */ }
    }
    throw error;
  }
}
