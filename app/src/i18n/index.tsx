// i18n: Amharic-first with English fallback.
//
// WHY Amharic default (not system-follow): the primary users are Ethiopian flour
// buyers; defaulting to አማርኛ guarantees the critical instructions are never
// lost on a phone whose system language happens to be English. A manual
// language switch lives in Settings and overrides the default.

import { I18n } from "i18n-js";
import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { getKV, setKV } from "../db/db";
import am from "./am.json";
import en from "./en.json";

export type Language = "am" | "en";
const LANG_KEY = "language";
const DEFAULT_LANG: Language = "am"; // product decision — see comment above

const i18n = new I18n({ am, en });
i18n.enableFallback = true;
i18n.defaultLocale = "en";

interface LocaleCtx {
  lang: Language;
  setLang: (l: Language) => void;
  t: (key: string, vars?: Record<string, unknown>) => string;
}

const Ctx = createContext<LocaleCtx>({
  lang: DEFAULT_LANG,
  setLang: () => undefined,
  t: (key) => key,
});

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Language>(DEFAULT_LANG);

  // Restore persisted choice once at startup.
  useEffect(() => {
    const saved = getKV(LANG_KEY);
    if (saved === "am" || saved === "en") setLangState(saved);
  }, []);

  const value = useMemo<LocaleCtx>(
    () => ({
      lang,
      setLang: (l) => {
        setLangState(l);
        setKV(LANG_KEY, l);
      },
      t: (key, vars) => i18n.t(key, { locale: lang, ...vars }),
    }),
    [lang],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLocale(): LocaleCtx {
  return useContext(Ctx);
}
