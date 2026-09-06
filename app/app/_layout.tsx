// Root layout: initializes DB + i18n + custom fonts before rendering screens.
// Navigation structure:
//   (tabs)/  -> bottom tabs: Scan | History | Settings
//   result   -> pushed on top of tabs after a capture
//
// Launch failsafe: the splash screen must NEVER trap the user. Fonts that
// stall or a slow DB can no longer block startup forever — after a timeout we
// render with system fonts, and any render error surfaces on screen instead of
// leaving a white screen.

import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import React, { useEffect, useState, useCallback, useRef } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import * as SplashScreen from "expo-splash-screen";
import { Ionicons } from "@expo/vector-icons";
import {
  useFonts,
  NotoSansEthiopic_400Regular,
  NotoSansEthiopic_500Medium,
  NotoSansEthiopic_700Bold,
} from "@expo-google-fonts/noto-sans-ethiopic";
import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_700Bold,
} from "@expo-google-fonts/inter";
import { LocaleProvider, useLocale } from "../src/i18n";
import { initDb } from "../src/db/db";
import { flushQueue } from "../src/upload/uploader";
import { preloadModels } from "../src/ml/inference";
import { colors, radius, spacing, typography } from "../src/theme";

// Keep the splash screen visible while we fetch resources
void SplashScreen.preventAutoHideAsync().catch(() => undefined);

/** Max seconds we wait for fonts before rendering with system fallbacks. */
const FONT_TIMEOUT_MS = 12_000;
/** Absolute upper bound before the splash hides no matter what. */
const SPLASH_HARD_LIMIT_MS = 15_000;

/** Render-error boundary — a crash shows an actionable screen, not a white void. */
class RootErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <View style={[styles.root, styles.boundary]}>
          <View style={styles.boundaryIcon}>
            <Ionicons name="bug-outline" size={38} color={colors.red} />
          </View>
          <Text style={styles.boundaryTitle}>ጥራት</Text>
          <Text style={styles.boundaryBody}>
            {String(this.state.error?.message ?? this.state.error)}
          </Text>
          <TouchableOpacity
            style={styles.boundaryBtn}
            onPress={() => this.setState({ error: null })}
            activeOpacity={0.8}
          >
            <Text style={styles.boundaryBtnText}>Retry</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}

function Shell() {
  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: colors.bg },
        }}
      >
        <Stack.Screen name="(tabs)" />
        {/* Result is presented as a push so back gesture = retake */}
        <Stack.Screen name="result" options={{ animation: "slide_from_right" }} />
      </Stack>
    </>
  );
}

// Separate component so it re-renders when the locale changes at runtime.
function ShellWithLocale() {
  const { t } = useLocale();
  if (!t) return null; // satisfies lint; provider guarantees t exists
  return <Shell />;
}

export default function RootLayout() {
  const [dbReady, setDbReady] = useState(false);
  const [fontsTimedOut, setFontsTimedOut] = useState(false);
  const splashHidden = useRef(false);

  const [fontsLoaded] = useFonts({
    NotoSansEthiopic_400Regular,
    NotoSansEthiopic_500Medium,
    NotoSansEthiopic_700Bold,
    Inter_400Regular,
    Inter_500Medium,
    Inter_700Bold,
  });

  useEffect(() => {
    (async () => {
      try {
        initDb();
        // Opportunistic drain of the opt-in contribution queue (stubbed no-op).
        void flushQueue().catch(() => undefined);
      } catch (e) {
        if (__DEV__) console.warn("[RootLayout] DB init failed:", e);
      } finally {
        setDbReady(true);
      }
    })();
    // Warm the TFLite models so the first scan is instant (safe to fail here).
    void preloadModels().catch(() => undefined);
  }, []);

  // Failsafe 1: never wait for fonts longer than FONT_TIMEOUT_MS.
  useEffect(() => {
    if (fontsLoaded) return;
    const t = setTimeout(() => setFontsTimedOut(true), FONT_TIMEOUT_MS);
    return () => clearTimeout(t);
  }, [fontsLoaded]);

  // Hide the splash exactly once, through any path that gets us to "ready".
  const hideSplash = useCallback(async () => {
    if (splashHidden.current) return;
    splashHidden.current = true;
    await SplashScreen.hideAsync().catch(() => undefined);
  }, []);

  const resourcesReady = (fontsLoaded || fontsTimedOut) && dbReady;

  // Failsafe 2: hide as soon as resources are ready…
  useEffect(() => {
    if (resourcesReady) void hideSplash();
  }, [resourcesReady, hideSplash]);

  // …and failsafe 3: an absolute deadline, whatever happens.
  useEffect(() => {
    const t = setTimeout(() => void hideSplash(), SPLASH_HARD_LIMIT_MS);
    return () => clearTimeout(t);
  }, [hideSplash]);

  if (!resourcesReady) {
    return null;
  }

  return (
    <RootErrorBoundary>
      <View style={styles.root} onLayout={() => void hideSplash()}>
        <LocaleProvider>
          <ShellWithLocale />
        </LocaleProvider>
      </View>
    </RootErrorBoundary>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  boundary: {
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.xl,
    gap: spacing.sm,
  },
  boundaryIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.redSoft,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.xs,
  },
  boundaryTitle: {
    fontSize: typography.fontSize.xl,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.ink,
  },
  boundaryBody: {
    fontSize: typography.fontSize.sm,
    fontFamily: typography.fontFamily.latinRegular,
    color: colors.inkMuted,
    textAlign: "center",
    lineHeight: 20,
  },
  boundaryBtn: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingHorizontal: spacing.xl,
    paddingVertical: 13,
    marginTop: spacing.md,
  },
  boundaryBtnText: {
    color: "#fff",
    fontSize: typography.fontSize.md,
    fontFamily: typography.fontFamily.latinBold,
  },
});

