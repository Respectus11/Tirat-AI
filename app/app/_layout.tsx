// Root layout: initializes DB + i18n + custom fonts before rendering screens.
// Navigation structure:
//   (tabs)/  -> bottom tabs: Scan | History | Settings
//   result   -> pushed on top of tabs after a capture

import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import React, { useEffect, useState, useCallback } from "react";
import { StyleSheet, View } from "react-native";
import * as SplashScreen from "expo-splash-screen";
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
import { colors } from "../src/theme";

// Keep the splash screen visible while we fetch resources
void SplashScreen.preventAutoHideAsync().catch(() => undefined);

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
  }, []);

  const onLayoutRootView = useCallback(async () => {
    if (fontsLoaded && dbReady) {
      await SplashScreen.hideAsync().catch(() => undefined);
    }
  }, [fontsLoaded, dbReady]);

  if (!fontsLoaded || !dbReady) {
    return null;
  }

  return (
    <View style={styles.root} onLayout={onLayoutRootView}>
      <LocaleProvider>
        <ShellWithLocale />
      </LocaleProvider>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
  },
});

