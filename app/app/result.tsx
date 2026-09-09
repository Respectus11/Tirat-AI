// Result screen: runs on-device inference for the captured photo, then shows
// one of four outcomes:
//   analyzing -> pulsing rings while the model forward pass runs
//   verdict   -> Pure (green) or Adulterated + est. % (red), animated reveal
//   inconclusive -> amber card, confidence below threshold (see config.ts)
//   error     -> model load failure vs generic failure, both retryable
//
// Every finished analysis (including inconclusive) is persisted to history;
// the opt-in upload queue pick-up happens inside enqueueForUpload().

import { router, useLocalSearchParams } from "expo-router";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Animated,
  Easing,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { Image } from "expo-image";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import { CONFIDENCE_THRESHOLD } from "../src/config";
import { enqueueForUpload, saveResult, type Verdict } from "../src/db/db";
import { useLocale } from "../src/i18n";
import { analyzePhoto, type VerdictKey, type FoodType } from "../src/ml/inference";
import { colors, radius, shadows, spacing, typography } from "../src/theme";
import { persistPhoto } from "../src/util/fs";

type Phase =
  | { kind: "analyzing" }
  | { kind: "error"; message: string; model: boolean }
  | { kind: "not_food"; isBlur?: boolean }
  | {
      kind: "done";
      verdict: VerdictKey | "inconclusive";
      confidence: number;
      estPct: number | null;
      foodType: FoodType;
    };

interface VerdictTheme {
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  gradFrom: string;
  gradTo: string;
  haptic: Haptics.NotificationFeedbackType;
}

const VERDICT_THEME: Record<Verdict | "adulterated", VerdictTheme> = {
  pure: {
    label: "verdict_pure",
    icon: "shield-checkmark",
    gradFrom: "#2E9E5B",
    gradTo: "#123A21",
    haptic: Haptics.NotificationFeedbackType.Success,
  },
  wood: {
    label: "verdict_wood",
    icon: "warning",
    gradFrom: "#EF4444",
    gradTo: "#7F1D1D",
    haptic: Haptics.NotificationFeedbackType.Error,
  },
  gypsum: {
    label: "verdict_gypsum",
    icon: "warning",
    gradFrom: "#EF4444",
    gradTo: "#7F1D1D",
    haptic: Haptics.NotificationFeedbackType.Error,
  },
  adulterated: {
    label: "Adulterated",
    icon: "warning",
    gradFrom: "#EF4444",
    gradTo: "#7F1D1D",
    haptic: Haptics.NotificationFeedbackType.Error,
  },
  inconclusive: {
    label: "inconclusive_chip",
    icon: "help-circle",
    gradFrom: "#F59E0B",
    gradTo: "#78350F",
    haptic: Haptics.NotificationFeedbackType.Warning,
  },
};

const ADULTERANT_NAME: Partial<Record<VerdictKey | "inconclusive", string>> = {
  wood: "wood flour",
  gypsum: "gypsum",
};

export default function ResultScreen() {
  const { t, lang } = useLocale();
  const insets = useSafeAreaInsets();
  const isAmharic = lang === "am";
  const params = useLocalSearchParams<{ photo?: string; foodType?: FoodType }>();
  const foodType = params.foodType ?? "redchili";
  const [phase, setPhase] = useState<Phase>({ kind: "analyzing" });

  // Guards so tapping Retry never persists the same scan twice.
  const savedRef = useRef(false);

  // Guard against setState on unmounted component when user navigates away during inference.
  const isCancelled = useRef(false);

  // Entrance choreography for the verdict card.
  const entrance = useRef(new Animated.Value(0)).current;

  const fireHaptic = useCallback((type: Haptics.NotificationFeedbackType) => {
    void Haptics.notificationAsync(type).catch(() => undefined);
  }, []);

  const run = useCallback(
    async (photoUri: string) => {
      isCancelled.current = false;
      savedRef.current = false;
      entrance.setValue(0);
      setPhase({ kind: "analyzing" });
      const r = await analyzePhoto(photoUri, foodType);
      if (isCancelled.current) return;
      if (r.status === "not_food" || r.status === "blurry") {
        // Not a powder sample (hand, table, …) or blurry — show guidance, save nothing.
        fireHaptic(Haptics.NotificationFeedbackType.Warning);
        if (isCancelled.current) return;
        setPhase({ kind: "not_food", isBlur: r.status === "blurry" });
        return;
      }
      if (r.status !== "ok") {
        setPhase({ kind: "error", message: r.message, model: r.status === "model_error" });
        return;
      }
      const verdict: VerdictKey | "inconclusive" =
        r.confidence < CONFIDENCE_THRESHOLD ? "inconclusive" : r.verdict;
      try {
        if (!savedRef.current) {
          const savedPath = await persistPhoto(photoUri);
          if (isCancelled.current) return;
          const id = saveResult({
            photoPath: savedPath,
            verdict: verdict as Verdict,
            adulterant: ADULTERANT_NAME[verdict] ?? null,
            confidence: r.confidence,
            estPct: r.estPct,
            foodType,
          });
          enqueueForUpload(id); // no-op unless the user opted in (checked inside)
          savedRef.current = true;
        }
      } catch (err) {
        // Persistence failed but the analysis itself is still valid — show it.
        if (isCancelled.current) return;
        if (__DEV__) console.warn("[result] save failed:", err);
      }
      if (isCancelled.current) return;
      fireHaptic(VERDICT_THEME[verdict].haptic);
      if (isCancelled.current) return;
      setPhase({ kind: "done", verdict, confidence: r.confidence, estPct: r.estPct, foodType });
      Animated.spring(entrance, {
        toValue: 1,
        friction: 7,
        tension: 60,
        useNativeDriver: true,
      }).start();
    },
    [entrance, fireHaptic, foodType],
  );

  useEffect(() => {
    if (typeof params.photo === "string" && params.photo.length > 0) {
      void run(params.photo);
    } else {
      setPhase({ kind: "error", message: "Missing photo parameter", model: false });
    }
    return () => {
      isCancelled.current = true;
    };
  }, [params.photo, run]);

  if (phase.kind === "analyzing") {
    return (
      <Centered bg={colors.bg}>
        <PulseRings />
        <Text style={[styles.analyzing, isAmharic && styles.fontAm]}>
          {t("analyzing")}
        </Text>
      </Centered>
    );
  }

  if (phase.kind === "error") {
    return (
      <Centered bg={colors.bg}>
        <View style={styles.errIconCircle}>
          <Ionicons
            name={phase.model ? "cube-outline" : "cloud-offline-outline"}
            size={34}
            color={colors.red}
          />
        </View>
        <Text style={[styles.errTitle, isAmharic && styles.fontAmBold]}>
          {phase.model ? t("model_missing_title") : t("error_title")}
        </Text>
        <Text style={[styles.errBody, isAmharic && styles.fontAm]}>
          {phase.model ? t("model_missing_body") : phase.message}
        </Text>
        {params.photo && (
          <TouchableOpacity
            style={styles.primaryBtn}
            onPress={() => void run(params.photo!)}
            activeOpacity={0.8}
          >
            <Ionicons name="refresh" size={18} color="#fff" />
            <Text style={[styles.primaryBtnText, isAmharic && styles.fontAmBold]}>
              {t("retry")}
            </Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity style={styles.ghostBtn} onPress={() => router.back()}>
          <Text style={[styles.ghostBtnText, isAmharic && styles.fontAm]}>
            {t("new_scan")}
          </Text>
        </TouchableOpacity>
      </Centered>
    );
  }

  if (phase.kind === "not_food") {
    return (
      <Centered bg={colors.bg}>
        <View style={styles.errIconCircle}>
          <Ionicons
            name={phase.isBlur ? "eye-off-outline" : "hand-left-outline"}
            size={34}
            color={colors.amber}
          />
        </View>
        <Text style={[styles.errTitle, isAmharic && styles.fontAmBold]}>
          {t("not_food_title")}
        </Text>
        <Text style={[styles.errBody, isAmharic && styles.fontAm]}>
          {phase.isBlur
            ? t("not_food_blur_body")
            : foodType === "teff"
            ? t("not_food_teff_body")
            : t("not_food_body")}
        </Text>
        <TouchableOpacity
          style={styles.primaryBtn}
          onPress={() => router.back()}
          activeOpacity={0.8}
        >
          <Ionicons name="camera-reverse-outline" size={18} color="#fff" />
          <Text style={[styles.primaryBtnText, isAmharic && styles.fontAmBold]}>
            {t("retake")}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.ghostBtn} onPress={() => router.back()}>
          <Text style={[styles.ghostBtnText, isAmharic && styles.fontAm]}>
            {t("new_scan")}
          </Text>
        </TouchableOpacity>
      </Centered>
    );
  }

  const theme = VERDICT_THEME[phase.verdict];
  const isInconclusive = phase.verdict === "inconclusive";
  const isAdulterated = !isInconclusive && phase.verdict !== "pure";

  return (
    <ScrollView
      style={{ flex: 1, backgroundColor: colors.bg }}
      contentContainerStyle={[
        styles.container,
        { paddingTop: insets.top + spacing.md, paddingBottom: insets.bottom + spacing.lg },
      ]}
      showsVerticalScrollIndicator={false}
    >
      {/* Capture preview */}
      {params.photo && (
        <View style={styles.photoFrame}>
          <Image source={{ uri: params.photo }} style={styles.photo} contentFit="cover" />
          <LinearGradient
            colors={["transparent", "rgba(0,0,0,0.35)"]}
            style={styles.photoOverlay}
          />
          <View style={styles.photoBadge}>
            <Ionicons name="camera" size={12} color="#fff" />
            <Text style={styles.photoBadgeText}>ጥራት</Text>
          </View>
        </View>
      )}

      {/* Verdict hero card */}
      <Animated.View
        style={{
          opacity: entrance,
          transform: [
            { scale: entrance.interpolate({ inputRange: [0, 1], outputRange: [0.92, 1] }) },
            { translateY: entrance.interpolate({ inputRange: [0, 1], outputRange: [24, 0] }) },
          ],
        }}
      >
        <LinearGradient
          colors={[theme.gradFrom, theme.gradTo]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.heroCard}
        >
          {/* Food type badge */}
          <View style={styles.foodBadge}>
            <Text style={styles.foodBadgeText}>
              {phase.foodType === "redchili" ? t("food_badge_redchili") : t("food_badge_teff")}
            </Text>
          </View>

          {isAdulterated && (
            <View style={styles.kickerPill}>
              <Text style={styles.kickerText}>
                {t("adulterated_generic").toUpperCase()}
              </Text>
            </View>
          )}
          <View style={styles.heroIconCircle}>
            <Ionicons name={theme.icon} size={40} color="#fff" />
          </View>
          <Text style={[styles.verdictTitle, isAmharic && styles.fontAmBold]}>
            {phase.verdict === "adulterated" ? t("verdict_adulterated") : t(theme.label)}
          </Text>

          {isInconclusive && (
            <Text style={[styles.inconclusiveBody, isAmharic && styles.fontAm]}>
              {t("inconclusive_body")}
            </Text>
          )}

          {!isInconclusive && phase.estPct != null && (
            <View style={styles.estRow}>
              <Text style={[styles.estLabel, isAmharic && styles.fontAm]}>
                {t("est_pct")}
              </Text>
              <CountUp target={phase.estPct} suffix="%" style={styles.estValue} />
            </View>
          )}
        </LinearGradient>
      </Animated.View>

      {/* Confidence meter */}
      <View style={styles.statsCard}>
        <View style={styles.statsHeader}>
          <Ionicons name="pulse" size={16} color={colors.primary} />
          <Text style={[styles.statsTitle, isAmharic && styles.fontAmBold]}>
            {t("confidence")}
          </Text>
          <View style={{ flex: 1 }} />
          <CountUp
            target={Math.round(phase.confidence * 100)}
            suffix="%"
            style={styles.statsValue}
          />
        </View>
        <Meter progress={phase.confidence} color={isInconclusive || isAdulterated ? theme.gradFrom : colors.primaryLight} />
        {!isInconclusive && phase.confidence < 0.85 && (
          <Text style={[styles.statsHint, isAmharic && styles.fontAm]}>
            {t("inconclusive_body")}
          </Text>
        )}
      </View>

      {/* Red Chili dataset disclaimer banner */}
      {phase.foodType === "redchili" && (
        <View style={styles.datasetDisclaimerBox}>
          <Ionicons name="information-circle" size={18} color="#D97706" style={{ marginTop: 2 }} />
          <Text style={styles.datasetDisclaimerText}>{t("redchili_disclaimer")}</Text>
        </View>
      )}

      {/* Teff demo banner — the real teff model has not been trained yet */}
      {phase.foodType === "teff" && (
        <View style={styles.datasetDisclaimerBox}>
          <Ionicons name="flask-outline" size={18} color="#D97706" style={{ marginTop: 2 }} />
          <Text style={styles.datasetDisclaimerText}>{t("teff_demo_badge")}</Text>
        </View>
      )}

      <Text style={[styles.savedNote, isAmharic && styles.fontAm]}>{t("saved_note")}</Text>
      <Text style={[styles.disclaimer, isAmharic && styles.fontAm]}>{t("disclaimer")}</Text>

      <View style={styles.actions}>
        <TouchableOpacity style={styles.secondaryBtn} onPress={() => router.back()} activeOpacity={0.8}>
          <Ionicons name="camera-reverse-outline" size={18} color={colors.primaryDark} />
          <Text style={[styles.secondaryBtnText, isAmharic && styles.fontAmBold]}>
            {t("retake")}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.primaryBtnWide} onPress={() => router.back()} activeOpacity={0.8}>
          <Ionicons name="add-circle-outline" size={18} color="#fff" />
          <Text style={[styles.primaryBtnWideText, isAmharic && styles.fontAmBold]}>
            {t("new_scan")}
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

/** Count-up number driven by a single Animated.Value per instance. */
function CountUp({
  target,
  suffix = "",
  style,
}: {
  target: number;
  suffix?: string;
  style?: object;
}) {
  const [display, setDisplay] = useState(0);
  const anim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const id = anim.addListener(({ value }) =>
      setDisplay(Math.round(value * target)),
    );
    Animated.timing(anim, {
      toValue: 1,
      duration: 900,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
    return () => anim.removeListener(id);
  }, [anim, target]);

  return (
    <Text style={style}>
      {display}
      {suffix}
    </Text>
  );
}

/** Horizontal progress meter (width animation; one-shot so JS driver is fine). */
function Meter({ progress, color }: { progress: number; color: string }) {
  const anim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(anim, {
      toValue: Math.max(0.02, Math.min(1, progress)),
      duration: 900,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
  }, [anim, progress]);

  return (
    <View style={styles.meterTrack}>
      <Animated.View
        style={[
          styles.meterFill,
          { backgroundColor: color, width: anim.interpolate({ inputRange: [0, 1], outputRange: ["0%", "100%"] }) },
        ]}
      />
    </View>
  );
}

/** Concentric pulsing rings shown while inference runs. */
function PulseRings() {
  const ring1 = useRef(new Animated.Value(0)).current;
  const ring2 = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const mkLoop = (v: Animated.Value, delay: number) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(v, {
            toValue: 1,
            duration: 1400,
            easing: Easing.out(Easing.quad),
            useNativeDriver: true,
          }),
          Animated.timing(v, { toValue: 0, duration: 0, useNativeDriver: true }),
        ]),
      );
    const l1 = mkLoop(ring1, 0);
    const l2 = mkLoop(ring2, 700);
    l1.start();
    l2.start();
    return () => {
      l1.stop();
      l2.stop();
    };
  }, [ring1, ring2]);

  return (
    <View style={styles.pulseWrap}>
      {[ring1, ring2].map((v, i) => (
        <Animated.View
          key={i}
          style={[
            styles.pulseRing,
            styles.pulseRingAbs,
            {
              opacity: v.interpolate({ inputRange: [0, 1], outputRange: [0.55, 0] }),
              transform: [{ scale: v.interpolate({ inputRange: [0, 1], outputRange: [0.75, 1.45] }) }],
            },
          ]}
        />
      ))}
      <View style={styles.pulseCore}>
        <Ionicons name="sparkles" size={30} color={colors.primary} />
      </View>
    </View>
  );
}

function Centered({ children, bg }: { children: React.ReactNode; bg: string }) {
  return <View style={[styles.center, { backgroundColor: bg }]}>{children}</View>;
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.lg,
  },
  container: { padding: spacing.lg },

  // Analyzing
  pulseWrap: { width: 120, height: 120, alignItems: "center", justifyContent: "center" },
  pulseRing: {
    width: 110,
    height: 110,
    borderRadius: 55,
    borderWidth: 2,
    borderColor: colors.primaryLight,
  },
  pulseRingAbs: { position: "absolute" },
  pulseCore: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: colors.primarySoft,
    alignItems: "center",
    justifyContent: "center",
    ...shadows.md,
  },
  analyzing: {
    marginTop: spacing.lg,
    fontSize: typography.fontSize.md,
    fontFamily: typography.fontFamily.latinMedium,
    color: colors.muted,
  },

  // Error
  errIconCircle: {
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: colors.redSoft,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.md,
  },
  errTitle: {
    fontSize: typography.fontSize.xl,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.ink,
    textAlign: "center",
  },
  errBody: {
    fontSize: typography.fontSize.sm,
    fontFamily: typography.fontFamily.latinRegular,
    color: colors.muted,
    textAlign: "center",
    marginVertical: spacing.sm,
    marginBottom: spacing.lg,
    lineHeight: 20,
    maxWidth: 300,
  },

  // Photo preview
  photoFrame: {
    height: 170,
    borderRadius: radius.lg,
    overflow: "hidden",
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.line,
    ...shadows.md,
  },
  photo: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0 },
  photoOverlay: { position: "absolute", left: 0, right: 0, bottom: 0, height: 70 },
  photoBadge: {
    position: "absolute",
    left: spacing.md,
    bottom: spacing.md,
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: 4,
    borderRadius: radius.full,
  },
  photoBadgeText: { color: "#fff", fontSize: 11, fontFamily: typography.fontFamily.bold },

  // Hero card
  heroCard: {
    borderRadius: radius.xl,
    paddingVertical: spacing.xl,
    paddingHorizontal: spacing.lg,
    alignItems: "center",
    ...shadows.lg,
  },
  kickerPill: {
    backgroundColor: "rgba(255,255,255,0.22)",
    paddingHorizontal: spacing.md,
    paddingVertical: 4,
    borderRadius: radius.full,
    marginBottom: spacing.md,
  },
  kickerText: {
    color: "#fff",
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.latinBold,
    letterSpacing: 1.2,
  },
  heroIconCircle: {
    width: 84,
    height: 84,
    borderRadius: 42,
    backgroundColor: "rgba(255,255,255,0.18)",
    borderWidth: 1.5,
    borderColor: "rgba(255,255,255,0.35)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.md,
  },
  verdictTitle: {
    fontSize: 26,
    fontFamily: typography.fontFamily.latinBold,
    color: "#fff",
    textAlign: "center",
    lineHeight: 32,
  },
  inconclusiveBody: {
    marginTop: spacing.sm,
    fontSize: typography.fontSize.sm,
    fontFamily: typography.fontFamily.regular,
    color: "rgba(255,255,255,0.92)",
    textAlign: "center",
    lineHeight: 20,
    maxWidth: 280,
  },
  estRow: {
    marginTop: spacing.lg,
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.15)",
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    borderRadius: radius.lg,
    minWidth: 180,
  },
  estLabel: {
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.regular,
    color: "rgba(255,255,255,0.85)",
    textTransform: "uppercase",
    letterSpacing: 0.6,
  },
  estValue: {
    fontSize: 40,
    fontFamily: typography.fontFamily.bold,
    color: "#fff",
    lineHeight: 46,
  },

  // Stats card
  statsCard: {
    marginTop: spacing.md,
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.line,
    padding: spacing.md,
    ...shadows.sm,
  },
  statsHeader: { flexDirection: "row", alignItems: "center", gap: spacing.xs + 2 },
  statsTitle: {
    fontSize: typography.fontSize.sm,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.inkMuted,
  },
  statsValue: {
    fontSize: typography.fontSize.lg,
    fontFamily: typography.fontFamily.bold,
    color: colors.primaryDark,
    fontVariant: ["tabular-nums"],
  },
  meterTrack: {
    marginTop: spacing.sm + 2,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.surfaceMuted,
    overflow: "hidden",
  },
  meterFill: { height: "100%", borderRadius: 5 },
  statsHint: {
    marginTop: spacing.sm,
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.regular,
    color: colors.amber,
    lineHeight: 15,
  },

  // Footer notes
  savedNote: {
    textAlign: "center",
    marginTop: spacing.lg,
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.inkSubtle,
  },
  disclaimer: {
    textAlign: "center",
    marginTop: spacing.xs,
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.regular,
    color: colors.inkSubtle,
    lineHeight: 17,
    marginHorizontal: spacing.sm,
  },

  // Actions
  actions: { flexDirection: "row", gap: spacing.md, marginTop: spacing.xl },
  secondaryBtn: {
    flex: 1,
    minHeight: 52,
    borderRadius: radius.md,
    borderWidth: 1.5,
    borderColor: colors.primarySoft,
    backgroundColor: colors.surface,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs,
    ...shadows.sm,
  },
  secondaryBtnText: {
    color: colors.primaryDark,
    fontSize: typography.fontSize.md,
    fontFamily: typography.fontFamily.latinBold,
  },
  primaryBtnWide: {
    flex: 1,
    minHeight: 52,
    borderRadius: radius.md,
    backgroundColor: colors.primary,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs,
    ...shadows.md,
  },
  primaryBtnWideText: { color: "#fff", fontSize: typography.fontSize.md, fontFamily: typography.fontFamily.latinBold },
  primaryBtn: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingHorizontal: spacing.xl,
    paddingVertical: 14,
    minHeight: 48,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs,
    ...shadows.md,
  },
  primaryBtnText: { color: "#fff", fontSize: typography.fontSize.md, fontFamily: typography.fontFamily.latinBold },
  ghostBtn: { marginTop: spacing.md, padding: spacing.sm },
  ghostBtnText: { color: colors.muted, fontSize: typography.fontSize.sm, fontFamily: typography.fontFamily.latinMedium },

  fontAm: { fontFamily: typography.fontFamily.regular },
  fontAmBold: { fontFamily: typography.fontFamily.bold },
  foodBadge: {
    alignSelf: "center",
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    paddingHorizontal: spacing.md,
    paddingVertical: 4,
    borderRadius: radius.full,
    marginBottom: spacing.xs,
  },
  foodBadgeText: {
    color: "#FFFFFF",
    fontSize: 12,
    fontFamily: typography.fontFamily.latinBold,
  },
  datasetDisclaimerBox: {
    flexDirection: "row",
    alignItems: "flex-start",
    backgroundColor: "#FEF3C7",
    borderColor: "#F59E0B",
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.md,
    marginTop: spacing.md,
    gap: spacing.xs + 2,
  },
  datasetDisclaimerText: {
    flex: 1,
    color: "#92400E",
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.latinMedium,
    lineHeight: 17,
  },
});
