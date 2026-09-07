// Opt-in data contribution pipeline — UPLOAD IS INTENTIONALLY STUBBED.
//
// Flow that already works end-to-end offline:
//   user opts in (Settings) -> every new scan is inserted into `upload_queue`
//   -> flushQueue() drains pending rows whenever we decide to try.
//
// TODO(backend): when the ingestion service exists:
//   1. Replace UPLOAD_ENDPOINT with the real URL.
//   2. Implement multipart upload of photo bytes + JSON metadata below.
//   3. On success call markUploaded(resultId); keep failures pending.
//   4. Add exponential backoff / Wi-Fi-only gating before production release.

import { nextPendingUploads } from "../db/db";

export const UPLOAD_ENDPOINT = "https://api.tirat.example/v1/contributions"; // placeholder
export function getUploadEndpoint(): string { return UPLOAD_ENDPOINT; }

export async function flushQueue(): Promise<{ uploaded: number; note: string }> {
  const batch = nextPendingUploads(10);
  if (batch.length === 0) return { uploaded: 0, note: "Queue empty" };

  // STUB: pretend success for at most one item so the status flags can be
  // exercised in demos, then stop. No network calls are made anywhere here yet.
  if (__DEV__) console.log(
    `[uploader] STUB flush of ${batch.length} queued sample(s) to ${UPLOAD_ENDPOINT} skipped`,
  );
  return { uploaded: 0, note: `Backend (${UPLOAD_ENDPOINT}) not wired up yet` };
}

