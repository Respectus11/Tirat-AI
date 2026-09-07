// Thin file-system helpers built on expo-file-system/legacy.
//
// WHY /legacy: SDK 57 moved the modern File/Directory API to the top level and
// re-exported the classic functions ONLY from this subpath (top-level ones throw).
// The legacy API is stable, typed, and everything an MVP needs.

import * as FileSystem from "expo-file-system/legacy";

const PHOTOS_DIR = `${FileSystem.documentDirectory}photos`;

export async function ensurePhotosDir(): Promise<string> {
  // intermediates:true => no error when it already exists
  await FileSystem.makeDirectoryAsync(PHOTOS_DIR, { intermediates: true });
  return PHOTOS_DIR;
}

/**
 * Persist a capture into permanent document storage.
 * Camera pictures live in the app CACHE, which the OS may clear at any time —
 * anything we want in history must be copied out before we rely on the path.
 */
export async function persistPhoto(tempUri: string): Promise<string> {
  const dir = await ensurePhotosDir();
  const dest = `${dir}/tirat_${Date.now()}.jpg`;
  await FileSystem.copyAsync({ from: tempUri, to: dest });
  return dest;
}


export function deleteFile(uri: string): void {
  // Fire-and-forget helper for pruning; callers already handle errors.
  void FileSystem.deleteAsync(uri, { idempotent: true });
}
