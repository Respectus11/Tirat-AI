// Scan tab: permission gate -> live camera with viewfinder guide & teff tips -> shutter.

import { CameraView, useCameraPermissions } from "expo-camera";
import { router, useFocusEffect } from "expo-router";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  Easing,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { useLocale } from "../../src/i18n";
import { colors, radius, shadows, spacing, typography } from "../../src/theme";

const GUIDE_SIZE = 260;

export default function ScanScreen() {
  const { t, lang } = useLocale();
  const insets = useSafeAreaInsets();
  const isAmharic = lang === "am";
  const cameraRef = useRef<React.ComponentRef<typeof CameraView>>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [flashOn, setFlashOn] = useState(false);
  const [busy, setBusy] = useState(false);
  const [focused, setFocused] = useState(true);
  const [foodType, setFoodType] = useState<"teff" | "redchili">("redchili");

  // Slow breathing pulse on the viewfinder guide (native driver, opacity only).
  const glow = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(glow, {
          toValue: 1,
          duration: 1600,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: true,
        }),
        Animated.timing(glow, {
          toValue: 0,
          duration: 1600,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: true,
        }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [glow]);

  useFocusEffect(
    useCallback(() => {
      setFocused(true);
      return () => setFocused(false);
    }, []),
  );

  const toggleFlash = () => {
    void Haptics.selectionAsync();
    setFlashOn((prev) => !prev);
  };

  const takePhoto = async () => {
    if (!cameraRef.current || busy) return;
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setBusy(true);
    try {
      const pic = await cameraRef.current.takePictureAsync({ quality: 0.9 });
      if (pic?.uri) {
        router.push({ pathname: "/result", params: { photo: pic.uri, foodType } });
      }
    } finally {
      setBusy(false);
    }
  };

  if (!permission) {
    return <View style={styles.fillBlack} />;
  }

  if (!permission.granted) {
    return (
      <View style={[styles.center, { backgroundColor: colors.bg, paddingBottom: insets.bottom + spacing.xl }]}>
        <View style={styles.permIconCircle}>
          <Ionicons name="camera" size={40} color={colors.primary} />
        </View>
        <Text style={[styles.permTitle, isAmharic && styles.fontAmBold]}>
          {t("permission_title")}
        </Text>
        <Text style={[styles.permBody, isAmharic && styles.fontAm]}>
          {t("permission_body")}
        </Text>
        <TouchableOpacity
          style={styles.primaryBtn}
          onPress={() => void requestPermission()}
          activeOpacity={0.8}
        >
          <Text style={[styles.primaryBtnText, isAmharic && styles.fontAmBold]}>
            {t("grant_permission")}
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.fillBlack}>
      {focused ? (
        <>
          {/* CameraView does not support children — overlays are siblings above it */}
          <CameraView
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          facing="back"
          enableTorch={flashOn}
          />
          {/* Top Bar with brand mark & food selector toggle */}
          <View style={[styles.topBar, { paddingTop: insets.top + spacing.sm }]}>
            <View style={styles.brandPill}>
              <Text style={styles.brandPillGlyph}>ጥ</Text>
              <Text style={[styles.topTitle, isAmharic && styles.fontAmBold]}>
                {t("capture_title")}
              </Text>
            </View>
            <View style={styles.foodSelectorRow}>
              <TouchableOpacity
                style={[styles.foodSelectorTab, foodType === "teff" && styles.foodSelectorTabActive]}
                onPress={() => setFoodType("teff")}
                activeOpacity={0.8}
              >
                <Text style={[styles.foodSelectorText, foodType === "teff" && styles.foodSelectorTextActive]}>
                  🌾 Teff Flour
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.foodSelectorTab, foodType === "redchili" && styles.foodSelectorTabActive]}
                onPress={() => setFoodType("redchili")}
                activeOpacity={0.8}
              >
                <Text style={[styles.foodSelectorText, foodType === "redchili" && styles.foodSelectorTextActive, isAmharic && styles.fontAm]}>
                  {isAmharic ? "🌶 ንጹሕ በርበሬ" : "🌶 Pure Chili"}
                </Text>
              </TouchableOpacity>
            </View>
          </View>

          {/* Viewfinder Target & Angle Guidance */}
          <View style={styles.guideWrap} pointerEvents="none">
            <View style={styles.flatnessHintPill}>
              <Ionicons name="scan-outline" size={13} color="#FFD54A" />
              <Text style={[styles.flatnessHintText, isAmharic && styles.fontAm]}>
                {t("instr_hold_flat")}
              </Text>
            </View>

            <View style={styles.guideBox}>
              <Animated.View
                style={[
                  styles.guideGlow,
                  { opacity: glow.interpolate({ inputRange: [0, 1], outputRange: [0, 0.35] }) },
                ]}
              />
              {/* Center reticle leveling target */}
              <View style={styles.reticleCenter}>
                <View style={styles.reticleCircle} />
                <View style={styles.reticleDot} />
              </View>
              {/* Corner accents */}
              <View style={[styles.corner, styles.cornerTL]} />
              <View style={[styles.corner, styles.cornerTR]} />
              <View style={[styles.corner, styles.cornerBL]} />
              <View style={[styles.corner, styles.cornerBR]} />
            </View>

            <Text style={[styles.fillHintText, isAmharic && styles.fontAm]}>
              {t("instr_fill_box")}
            </Text>
          </View>

          {/* Bottom Bar: instruction chips & shutter controls */}
          <View style={[styles.bottomBar, { paddingBottom: insets.bottom + spacing.md }]}>
            <View style={styles.chipRow}>
              <View style={styles.chip}>
                <MaterialCommunityIcons name="grain" size={14} color="#FFD54A" />
                <Text style={[styles.chipText, isAmharic && styles.fontAm]}>{t("instr_flat")}</Text>
              </View>
              <View style={styles.chip}>
                <Ionicons name="square-outline" size={13} color="#FFD54A" />
                <Text style={[styles.chipText, isAmharic && styles.fontAm]}>{t("instr_bg")}</Text>
              </View>
              <View style={styles.chip}>
                <Ionicons name="flash" size={13} color="#FFD54A" />
                <Text style={[styles.chipText, isAmharic && styles.fontAm]}>{t("instr_flash")}</Text>
              </View>
            </View>

            <View style={styles.controlsRow}>
              {/* Flash toggle */}
              <TouchableOpacity
                style={styles.controlBtn}
                accessibilityLabel="Flash toggle"
                onPress={toggleFlash}
                activeOpacity={0.7}
              >
                <Ionicons
                  name={flashOn ? "flash" : "flash-off"}
                  size={24}
                  color={flashOn ? "#FFD54A" : "rgba(255,255,255,0.6)"}
                />
                <Text style={styles.controlBtnText}>{flashOn ? "ON" : "OFF"}</Text>
              </TouchableOpacity>

              {/* Shutter button */}
              <TouchableOpacity
                style={styles.shutterOuter}
                accessibilityLabel={t("shutter")}
                disabled={busy}
                onPress={() => void takePhoto()}
                activeOpacity={0.85}
              >
                {busy ? (
                  <ActivityIndicator color={colors.primary} size="large" />
                ) : (
                  <View style={styles.shutterInner} />
                )}
              </TouchableOpacity>

              {/* Balance spacer */}
              <View style={styles.controlBtnSpacer} />
            </View>
          </View>
        </>
      ) : (
        <View style={[StyleSheet.absoluteFill, styles.fillBlack]}>
          {busy && <ActivityIndicator color={colors.primary} />}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  fillBlack: { flex: 1, backgroundColor: "#000" },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.lg,
  },
  permIconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.primarySoft,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.md,
  },
  permTitle: {
    fontSize: typography.fontSize.xl,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.ink,
    textAlign: "center",
  },
  permBody: {
    fontSize: typography.fontSize.md,
    color: colors.inkMuted,
    textAlign: "center",
    marginTop: spacing.sm,
    marginBottom: spacing.lg,
    lineHeight: 22,
    maxWidth: 300,
  },
  primaryBtn: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingHorizontal: spacing.xl,
    paddingVertical: 14,
    minHeight: 48,
    justifyContent: "center",
    ...shadows.md,
  },
  primaryBtnText: {
    color: "#fff",
    fontSize: typography.fontSize.md,
    fontFamily: typography.fontFamily.latinBold,
  },
  topBar: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    alignItems: "center",
    paddingHorizontal: spacing.lg,
  },
  brandPill: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0, 0, 0, 0.65)",
    paddingHorizontal: spacing.md,
    paddingVertical: 8,
    borderRadius: radius.full,
    gap: spacing.xs,
  },
  brandPillGlyph: {
    color: "#2E9E5B",
    fontSize: 16,
    fontFamily: typography.fontFamily.bold,
  },
  topTitle: {
    color: "#fff",
    fontSize: 14,
    fontFamily: typography.fontFamily.latinMedium,
  },
  guideWrap: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: "center",
    justifyContent: "center",
  },
  flatnessHintPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(0, 0, 0, 0.75)",
    paddingVertical: 5,
    paddingHorizontal: 12,
    borderRadius: radius.full,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: "rgba(255, 213, 74, 0.4)",
  },
  flatnessHintText: {
    color: "#FFFFFF",
    fontSize: 12,
    fontWeight: "600",
  },
  fillHintText: {
    color: "rgba(255, 255, 255, 0.75)",
    fontSize: 11,
    marginTop: spacing.xs,
    textAlign: "center",
  },
  guideBox: {
    width: GUIDE_SIZE,
    height: GUIDE_SIZE,
    borderWidth: 1.5,
    borderColor: "rgba(255, 255, 255, 0.35)",
    borderRadius: radius.md,
    position: "relative",
  },
  guideGlow: {
    ...StyleSheet.absoluteFill,
    borderRadius: radius.md,
    backgroundColor: "rgba(46, 158, 91, 0.5)",
  },
  reticleCenter: {
    ...StyleSheet.absoluteFill,
    alignItems: "center",
    justifyContent: "center",
  },
  reticleCircle: {
    width: 68,
    height: 68,
    borderRadius: 34,
    borderWidth: 1.5,
    borderColor: "rgba(255, 255, 255, 0.3)",
    borderStyle: "dashed",
  },
  reticleDot: {
    position: "absolute",
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "rgba(255, 213, 74, 0.7)",
  },
  corner: {
    position: "absolute",
    width: 24,
    height: 24,
    borderColor: "#FFFFFF",
  },
  cornerTL: {
    top: -2,
    left: -2,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderTopLeftRadius: radius.md,
  },
  cornerTR: {
    top: -2,
    right: -2,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderTopRightRadius: radius.md,
  },
  cornerBL: {
    bottom: -2,
    left: -2,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderBottomLeftRadius: radius.md,
  },
  cornerBR: {
    bottom: -2,
    right: -2,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderBottomRightRadius: radius.md,
  },
  bottomBar: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: spacing.lg,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: spacing.xs,
    marginBottom: spacing.lg,
  },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0, 0, 0, 0.65)",
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: 6,
    borderRadius: radius.full,
    gap: 4,
  },
  chipText: {
    color: "#FFFFFF",
    fontSize: 12,
    fontFamily: typography.fontFamily.latinRegular,
  },
  controlsRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  controlBtn: {
    width: 64,
    alignItems: "center",
    gap: 2,
  },
  controlBtnSpacer: {
    width: 64,
  },
  controlBtnText: {
    color: "#FFFFFF",
    fontSize: 11,
    fontFamily: typography.fontFamily.latinBold,
  },
  shutterOuter: {
    width: 78,
    height: 78,
    borderRadius: 39,
    borderWidth: 4,
    borderColor: "rgba(255, 255, 255, 0.95)",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0, 0, 0, 0.2)",
  },
  shutterInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#FFFFFF",
  },
  fontAm: {
    fontFamily: typography.fontFamily.regular,
  },
  fontAmBold: {
    fontFamily: typography.fontFamily.bold,
  },
  foodSelectorRow: {
    flexDirection: "row",
    gap: spacing.xs,
    marginTop: spacing.xs + 2,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    padding: 3,
    borderRadius: radius.full,
  },
  foodSelectorTab: {
    paddingHorizontal: spacing.md,
    paddingVertical: 5,
    borderRadius: radius.full,
  },
  foodSelectorTabActive: {
    backgroundColor: colors.primary,
  },
  foodSelectorText: {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: 12,
    fontFamily: typography.fontFamily.latinMedium,
  },
  foodSelectorTextActive: {
    color: "#FFFFFF",
    fontFamily: typography.fontFamily.latinBold,
  },
});
