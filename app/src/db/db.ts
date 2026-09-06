// Local persistence — expo-sqlite, fully offline.
//
// Three tables:
//   results      every scan the user has taken (with photo path on disk)
//   upload_queue photos+results queued for the (stubbed) opt-in improvement program
//   kv           tiny settings store (language, opt-in flag) — no extra dependency

import * as SQLite from "expo-sqlite";
import { HISTORY_LIMIT } from "../config";
import { deleteFile } from "../util/fs";

export const db = SQLite.openDatabaseSync("tirat.db");

export type Verdict = "pure" | "adulterated" | "wood" | "gypsum" | "inconclusive";
export type FoodType = "teff" | "redchili";

export interface HistoryRow {
  id: number;
  created_at: string; // ISO 8601
  photo_path: string;
  verdict: Verdict;
  adulterant: string | null;
  confidence: number;
  est_pct: number | null;
  food_type: FoodType;
  queued: 0 | 1;
}

export function initDb(): void {
  db.execSync(`
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      photo_path TEXT NOT NULL,
      verdict TEXT NOT NULL,
      adulterant TEXT,
      confidence REAL NOT NULL,
      est_pct INTEGER
    );

    CREATE TABLE IF NOT EXISTS upload_queue (
      result_id INTEGER PRIMARY KEY REFERENCES results(id),
      status TEXT NOT NULL DEFAULT 'pending',
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS kv (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  `);

  // Migration: food_type was added when red-chili support shipped. Old
  // installs keep every row — new column defaults to 'teff'.
  const cols = db.getAllSync<{ name: string }>(`PRAGMA table_info(results)`);
  if (!cols.some((c) => c.name === "food_type")) {
    db.runSync(
      `ALTER TABLE results ADD COLUMN food_type TEXT NOT NULL DEFAULT 'teff'`,
    );
  }
}

// ---------------- results ----------------

export function saveResult(r: {
  photoPath: string;
  verdict: Verdict;
  adulterant: string | null;
  confidence: number;
  estPct: number | null;
  foodType: FoodType;
}): number {
  const res = db.runSync(
    `INSERT INTO results (created_at, photo_path, verdict, adulterant, confidence, est_pct, food_type)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [
      new Date().toISOString(),
      r.photoPath,
      r.verdict,
      r.adulterant,
      r.confidence,
      r.estPct,
      r.foodType,
    ],
  );
  pruneHistory();
  return Number(res.lastInsertRowId);
}

export function getHistory(limit = 200): HistoryRow[] {
  return db.getAllSync<HistoryRow>(
    `SELECT r.*, CASE WHEN q.result_id IS NULL THEN 0 ELSE 1 END AS queued
     FROM results r
     LEFT JOIN upload_queue q ON q.result_id = r.id
     ORDER BY r.id DESC
     LIMIT ?`,
    [limit],
  );
}

/** Keeps only the newest HISTORY_LIMIT rows, deleting their photos from disk too. */
function pruneHistory(): void {
  const stale = db.getAllSync<{ id: number; photo_path: string }>(
    `SELECT id, photo_path FROM results WHERE id NOT IN
     (SELECT id FROM results ORDER BY id DESC LIMIT ?)`,
    [HISTORY_LIMIT],
  );
  for (const row of stale) {
    try {
      deleteFile(row.photo_path);
    } catch {
      // best-effort cleanup; DB row removal below is what matters
    }
    db.runSync(`DELETE FROM upload_queue WHERE result_id = ?`, [row.id]);
    db.runSync(`DELETE FROM results WHERE id = ?`, [row.id]);
  }
}

export function clearHistory(): void {
  const rows = db.getAllSync<{ photo_path: string }>(`SELECT photo_path FROM results`);
  for (const row of rows) {
    try {
      deleteFile(row.photo_path);
    } catch {
      // ignore missing files
    }
  }
  db.runSync(`DELETE FROM upload_queue`);
  db.runSync(`DELETE FROM results`);
}

// ---------------- upload queue ----------------

export function enqueueForUpload(resultId: number): void {
  // Only queue if the user opted in. The actual upload is a stub (src/upload).
  if (getKV("opt_in") !== "1") return;
  db.runSync(
    `INSERT OR IGNORE INTO upload_queue (result_id, status, created_at) VALUES (?, 'pending', ?)`,
    [resultId, new Date().toISOString()],
  );
}

export function pendingUploadCount(): number {
  const rows = db.getFirstSync<{ n: number }>(
    `SELECT COUNT(*) AS n FROM upload_queue WHERE status = 'pending'`,
  );
  return rows?.n ?? 0;
}

export function markUploaded(resultId: number): void {
  db.runSync(`UPDATE upload_queue SET status = 'uploaded' WHERE result_id = ?`, [resultId]);
}

export function nextPendingUploads(limit = 10): Array<{ result_id: number; photo_path: string }> {
  return db.getAllSync<{ result_id: number; photo_path: string }>(
    `SELECT q.result_id AS result_id, r.photo_path AS photo_path
     FROM upload_queue q JOIN results r ON r.id = q.result_id
     WHERE q.status = 'pending' LIMIT ?`,
    [limit],
  );
}

// ---------------- settings (kv) ----------------

export function getKV(key: string): string | null {
  const row = db.getFirstSync<{ value: string }>(`SELECT value FROM kv WHERE key = ?`, [key]);
  return row?.value ?? null;
}

export function setKV(key: string, value: string): void {
  db.runSync(
    `INSERT INTO kv (key, value) VALUES (?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
    [key, value],
  );
}
