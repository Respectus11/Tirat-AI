import React, { useState } from "react";
import {
  Alert,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { Image } from "expo-image";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import Constants from "expo-constants";
import { getKV, pendingUploadCount, setKV } from "../../src/db/db";
import { useLocale, type Language } from "../../src/i18n";
import { flushQueue } from "../../src/upload/uploader";
import { colors, radius, shadows, spacing, typography } from "../../src/theme";

export default function SettingsScreen() {
  const { t, lang, setLang } = useLocale();
  const insets = useSafeAreaInsets();
  const isAmharic = lang === "am";

  const [optIn, setOptIn] = useState(() => getKV("opt_in") === "1");
  const [pendingCount, setPendingCount] = useState(() => pendingUploadCount());
  const [flushing, setFlushing] = useState(false);

  const handleLangSelect = (selectedLang: Language) => {
    void Haptics.selectionAsync();
    setLang(selectedLang);
  };

  const handleOptInToggle = (enabled: boolean) => {
    void Haptics.selectionAsync();
    if (enabled) {
      Alert.alert(t("opt_in_alert_title"), t("opt_in_alert_body"), [
        { text: t("cancel"), style: "cancel", onPress: () => setOptIn(false) },
        { text: "OK", onPress: () => { setOptIn(true); setKV("opt_in", "1"); } },
      ]);
    } else {
      setOptIn(false);
      setKV("opt_in", "0");
    }
  };

  const handleFlushQueue = async () => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setFlushing(true);
    try {
      await flushQueue();
      setPendingCount(pendingUploadCount());
      Alert.alert("ጥራት AI", t("upload_stub_note"));
    } finally {
      setFlushing(false);
    }
  };

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={[
        styles.content,
        { paddingTop: insets.top + spacing.md, paddingBottom: insets.bottom + spacing.xl },
      ]}
      showsVerticalScrollIndicator={false}
    >
      <View style={styles.heroCard}>
        <Image
          source={require("../../assets/images/teff-harvest.jpg")}
          style={styles.fill}
          contentFit="cover"
        />
        <LinearGradient
          colors={["rgba(19, 79, 44, 0.75)", "rgba(31, 122, 69, 0.92)"]}
          style={styles.fill}
        />
        <View style={styles.heroContent}>
          <View style={styles.heroLogoBadge}>
            <Text style={styles.heroLogoText}>ጥ</Text>
          </View>
          <Text style={[styles.heroTitle, isAmharic && styles.fontAmBold]}>
            {t("tab_settings")}
          </Text>
          <Text style={[styles.heroSubtitle, isAmharic && styles.fontAm]}>
            {t("app_tagline")}
          </Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, isAmharic && styles.fontAmBold]}>
          {t("language")}
        </Text>
        <View style={styles.card}>
          <TouchableOpacity
            style={[styles.row, lang === "am" && styles.rowActive]}
            onPress={() => handleLangSelect("am")}
            activeOpacity={0.7}
          >
            <View style={styles.rowLeft}>
              <View style={[styles.iconCircle, lang === "am" && styles.iconCircleActive]}>
                <Text style={styles.amharicGlyph}>አ</Text>
              </View>
              <View>
                <Text style={[styles.rowTitle, styles.fontAmBold]}>{t("lang_amharic")}</Text>
                <Text style={styles.rowSubtitle}>Amharic (Default)</Text>
              </View>
            </View>
            {lang === "am" && <Ionicons name="checkmark-circle" size={24} color={colors.primary} />}
          </TouchableOpacity>

          <View style={styles.divider} />

          <TouchableOpacity
            style={[styles.row, lang === "en" && styles.rowActive]}
            onPress={() => handleLangSelect("en")}
            activeOpacity={0.7}
          >
            <View style={styles.rowLeft}>
              <View style={[styles.iconCircle, lang === "en" && styles.iconCircleActive]}>
                <Ionicons
                  name="globe-outline"
                  size={20}
                  color={lang === "en" ? colors.primary : colors.inkSubtle}
                />
              </View>
              <View>
                <Text style={styles.rowTitle}>{t("lang_english")}</Text>
                <Text style={styles.rowSubtitle}>English</Text>
              </View>
            </View>
            {lang === "en" && <Ionicons name="checkmark-circle" size={24} color={colors.primary} />}
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, isAmharic && styles.fontAmBold]}>
          {t("help_improve")}
        </Text>
        <View style={styles.card}>
          <View style={styles.row}>
            <View style={styles.rowLeftFlex}>
              <View style={styles.iconCircle}>
                <Ionicons name="cloud-upload-outline" size={20} color={colors.primary} />
              </View>
              <View style={styles.textWrap}>
                <Text style={[styles.rowTitle, isAmharic && styles.fontAmBold]}>
                  {t("help_improve")}
                </Text>
                <Text style={[styles.rowDesc, isAmharic && styles.fontAm]}>
                  {t("help_improve_desc")}
                </Text>
              </View>
            </View>
            <Switch
              value={optIn}
              onValueChange={handleOptInToggle}
              trackColor={{ false: colors.line, true: colors.primaryLight }}
              thumbColor={optIn ? colors.primary : "#f4f3f4"}
            />
          </View>
          {optIn && (
            <>
              <View style={styles.divider} />
              <View style={styles.row}>
                <View style={styles.rowLeft}>
                  <View style={styles.iconCircle}>
                    <MaterialCommunityIcons name="folder-clock-outline" size={20} color={colors.inkMuted} />
                  </View>
                  <View>
                    <Text style={[styles.rowTitle, isAmharic && styles.fontAmBold]}>
                      {t("pending_uploads")}
                    </Text>
                    <Text style={styles.rowSubtitle}>{pendingCount} scans</Text>
                  </View>
                </View>
                <TouchableOpacity
                  style={styles.actionPill}
                  onPress={() => void handleFlushQueue()}
                  disabled={flushing}
                >
                  <Text style={[styles.actionPillText, isAmharic && styles.fontAmBold]}>
                    {flushing ? "…" : t("try_upload_now")}
                  </Text>
                </TouchableOpacity>
              </View>
            </>
          )}
        </View>
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, isAmharic && styles.fontAmBold]}>
          {t("about")}
        </Text>
        <View style={styles.card}>
          <View style={styles.aboutHeader}>
            <Image
              source={require("../../assets/images/injera-texture.jpg")}
              style={styles.aboutThumb}
              contentFit="cover"
            />
            <View style={styles.aboutMeta}>
              <Text style={styles.aboutAppName}>ጥራት (Tirat AI)</Text>
              <Text style={styles.aboutVersion}>
                {`v${Constants.expoConfig?.version ?? "0.2.1"} · On-device MobileNetV3`}
              </Text>
              <Text style={styles.aboutDesc}>
                Open-source AI tool to screen food samples (pure red chili powder and teff flour) for adulteration.
              </Text>
            </View>
          </View>
          <View style={styles.divider} />
          <View style={styles.metaRow}>
            <Text style={styles.metaKey}>Model Input</Text>
            <Text style={styles.metaVal}>224 × 224 px · TFLite Float16</Text>
          </View>
          <View style={styles.divider} />
          <View style={styles.metaRow}>
            <Text style={styles.metaKey}>Confidence Gate</Text>
            <Text style={styles.metaVal}>≥ 70% required</Text>
          </View>
        </View>
      </View>

      <View style={styles.disclaimerBox}>
        <Ionicons name="information-circle-outline" size={18} color={colors.inkSubtle} />
        <Text style={[styles.disclaimerText, isAmharic && styles.fontAm]}>
          {t("disclaimer")}
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { paddingHorizontal: spacing.md },
  fill: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0 },
  heroCard: {
    height: 140,
    borderRadius: radius.lg,
    overflow: "hidden",
    marginBottom: spacing.lg,
    justifyContent: "flex-end",
    padding: spacing.md,
    ...shadows.md,
  },
  heroContent: { gap: 2 },
  heroLogoBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 4,
  },
  heroLogoText: { color: colors.white, fontSize: 18, fontFamily: typography.fontFamily.bold },
  heroTitle: { color: colors.white, fontSize: typography.fontSize.xl, fontFamily: typography.fontFamily.latinBold },
  heroSubtitle: { color: "rgba(255, 255, 255, 0.85)", fontSize: typography.fontSize.sm, fontFamily: typography.fontFamily.latinRegular },
  section: { marginBottom: spacing.lg },
  sectionTitle: {
    fontSize: typography.fontSize.sm,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.inkMuted,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: spacing.sm,
    paddingHorizontal: spacing.xs,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.line,
    overflow: "hidden",
    ...shadows.sm,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 14,
    paddingHorizontal: spacing.md,
  },
  rowActive: { backgroundColor: colors.primarySoft },
  rowLeft: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  rowLeftFlex: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: spacing.md,
    flex: 1,
    marginRight: spacing.sm,
  },
  textWrap: { flex: 1 },
  iconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.surfaceMuted,
    alignItems: "center",
    justifyContent: "center",
  },
  iconCircleActive: { backgroundColor: colors.surface },
  amharicGlyph: { fontSize: 18, color: colors.primaryDark, fontFamily: typography.fontFamily.bold },
  rowTitle: { fontSize: typography.fontSize.md, fontFamily: typography.fontFamily.latinMedium, color: colors.ink },
  rowSubtitle: { fontSize: typography.fontSize.xs, color: colors.inkSubtle, marginTop: 2 },
  rowDesc: { fontSize: typography.fontSize.xs, color: colors.inkMuted, marginTop: 4, lineHeight: 16 },
  divider: { height: 1, backgroundColor: colors.line, marginLeft: 56 },
  actionPill: { backgroundColor: colors.primarySoft, paddingHorizontal: spacing.md, paddingVertical: 6, borderRadius: radius.full },
  actionPillText: { color: colors.primaryDark, fontSize: typography.fontSize.xs, fontFamily: typography.fontFamily.latinBold },
  aboutHeader: { flexDirection: "row", padding: spacing.md, gap: spacing.md, alignItems: "center" },
  aboutThumb: { width: 60, height: 60, borderRadius: radius.sm },
  aboutMeta: { flex: 1 },
  aboutAppName: { fontSize: typography.fontSize.md, fontFamily: typography.fontFamily.bold, color: colors.ink },
  aboutVersion: { fontSize: typography.fontSize.xs, color: colors.inkSubtle, marginTop: 2 },
  aboutDesc: { fontSize: typography.fontSize.xs, color: colors.inkMuted, marginTop: 4, lineHeight: 16 },
  metaRow: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 12, paddingHorizontal: spacing.md },
  metaKey: { fontSize: typography.fontSize.sm, color: colors.inkMuted },
  metaVal: { fontSize: typography.fontSize.sm, fontFamily: typography.fontFamily.latinMedium, color: colors.ink },
  disclaimerBox: {
    flexDirection: "row",
    gap: spacing.sm,
    padding: spacing.md,
    backgroundColor: colors.surfaceMuted,
    borderRadius: radius.md,
    alignItems: "center",
  },
  disclaimerText: { flex: 1, fontSize: typography.fontSize.xs, color: colors.inkMuted, lineHeight: 17 },
  fontAm: { fontFamily: typography.fontFamily.regular },
  fontAmBold: { fontFamily: typography.fontFamily.bold },
});
