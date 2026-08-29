// History tab: newest-first list of saved scans with thumbnails, refreshed on
// every focus (a scan may have just been added by the result screen).
// A summary strip shows total scans / pure share / flagged count at a glance.
// Clear-all asks for confirmation because it deletes photos from disk too.

import { router, useFocusEffect } from "expo-router";
import React, { useCallback, useMemo, useState } from "react";
import {
  Alert,
  FlatList,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { Image } from "expo-image";
import * as Haptics from "expo-haptics";
import { clearHistory, getHistory, type HistoryRow, type Verdict } from "../../src/db/db";
import { useLocale } from "../../src/i18n";
import { colors, radius, shadows, spacing, typography } from "../../src/theme";

interface VerdictMeta {
  label: string;
  color: string;
  softBg: string;
  icon: keyof typeof Ionicons.glyphMap;
}

const VERDICT_META: Record<Verdict, VerdictMeta> = {
  pure: {
    label: "verdict_pure",
    color: "#166534",
    softBg: colors.greenSoft,
    icon: "shield-checkmark",
  },
  wood: { label: "verdict_wood", color: "#991B1B", softBg: colors.redSoft, icon: "warning" },
  gypsum: { label: "verdict_gypsum", color: "#991B1B", softBg: colors.redSoft, icon: "warning" },
  inconclusive: {
    label: "inconclusive_chip",
    color: "#92400E",
    softBg: colors.amberSoft,
    icon: "help-circle",
  },
};

export default function HistoryScreen() {
  const { t, lang } = useLocale();
  const insets = useSafeAreaInsets();
  const isAmharic = lang === "am";
  const [rows, setRows] = useState<HistoryRow[]>([]);

  useFocusEffect(
    useCallback(() => {
      setRows(getHistory());
    }, []),
  );

  const stats = useMemo(() => {
    const total = rows.length;
    const pure = rows.filter((r) => r.verdict === "pure").length;
    const flagged = rows.filter((r) => r.verdict === "wood" || r.verdict === "gypsum").length;
    return { total, purePct: total ? Math.round((pure / total) * 100) : null, flagged };
  }, [rows]);

  const confirmClear = () => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    Alert.alert(t("clear_history"), t("clear_history_confirm"), [
      { text: t("cancel"), style: "cancel" },
      {
        text: t("delete"),
        style: "destructive",
        onPress: () => {
          clearHistory();
          setRows([]);
        },
      },
    ]);
  };

  const formatWhen = (iso: string) => {
    const d = new Date(iso);
    try {
      return d.toLocaleString(lang === "am" ? "am-ET" : "en-GB", {
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return iso;
    }
  };

  if (rows.length === 0) {
    return (
      <View style={[styles.center, styles.screen]}>
        <View style={styles.emptyIconCircle}>
          <Ionicons name="file-tray-full-outline" size={38} color={colors.primary} />
        </View>
        <Text style={[styles.emptyTitle, isAmharic && styles.fontAmBold]}>
          {t("history_empty")}
        </Text>
        <Text style={[styles.emptyHint, isAmharic && styles.fontAm]}>
          {t("history_empty_hint")}
        </Text>
        <TouchableOpacity
          style={styles.emptyCta}
          activeOpacity={0.8}
          onPress={() => router.navigate("/")}
        >
          <Ionicons name="camera" size={16} color="#fff" />
          <Text style={[styles.emptyCtaText, isAmharic && styles.fontAmBold]}>
            {t("tab_capture")}
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <FlatList
        contentContainerStyle={{ padding: spacing.md, paddingBottom: insets.bottom + spacing.xl }}
        data={rows}
        keyExtractor={(r) => String(r.id)}
        ItemSeparatorComponent={Separator}
        ListHeaderComponent={
          <>
            <View style={styles.statsRow}>
              <StatCard icon="albums" label={t("tab_history")} value={String(stats.total)} />
              <StatCard
                icon="shield-checkmark"
                label={t("verdict_pure")}
                value={stats.purePct != null ? `${stats.purePct}%` : "—"}
                tone="good"
              />
              <StatCard
                icon="alert-circle"
                label={t("adulterated_generic")}
                value={String(stats.flagged)}
                tone={stats.flagged > 0 ? "bad" : undefined}
              />
            </View>
            <Text style={[styles.listTitle, isAmharic && styles.fontAmBold]}>
              {t("recent_scans")}
            </Text>
          </>
        }
        renderItem={({ item }) => {
          const meta = VERDICT_META[item.verdict];
          return (
            <View style={styles.row}>
              <Image source={{ uri: item.photo_path }} style={styles.thumb} contentFit="cover" />
              <View style={styles.rowMain}>
                <View style={[styles.chip, { backgroundColor: meta.softBg }]}>
                  <Ionicons name={meta.icon} size={12} color={meta.color} />
                  <Text style={[styles.chipText, { color: meta.color }]}>
                    {t(meta.label)}
                  </Text>
                </View>
                <Text style={[styles.sub, isAmharic && styles.fontAm]} numberOfLines={1}>
                  {formatWhen(item.created_at)}
                  {" · "}
                  {t("confidence").toLowerCase()} {Math.round(item.confidence * 100)}%
                  {item.est_pct != null
                    ? ` · ${t("est_pct").toLowerCase()} ${item.est_pct}%`
                    : ""}
                </Text>
              </View>
              {item.queued === 1 && (
                <View style={styles.queuedDotWrap}>
                  <View style={styles.queuedDot} />
                </View>
              )}
            </View>
          );
        }}
      />
      <TouchableOpacity
        style={[styles.clearBtn, { marginBottom: insets.bottom + spacing.sm }]}
        onPress={confirmClear}
        activeOpacity={0.8}
      >
        <Ionicons name="trash-outline" size={16} color={colors.red} />
        <Text style={[styles.clearBtnText, isAmharic && styles.fontAmBold]}>
          {t("clear_history")}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

function StatCard({
  icon,
  label,
  value,
  tone,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  value: string;
  tone?: "good" | "bad";
}) {
  const accent =
    tone === "good" ? colors.green : tone === "bad" ? colors.red : colors.primary;
  return (
    <View style={styles.statCard}>
      <Ionicons name={icon} size={15} color={accent} />
      <Text style={[styles.statValue, { color: accent }]}>{value}</Text>
      <Text style={styles.statLabel} numberOfLines={1}>
        {label}
      </Text>
    </View>
  );
}

function Separator() {
  return <View style={styles.separator} />;
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  center: { alignItems: "center", justifyContent: "center", padding: spacing.lg },

  // Empty state
  emptyIconCircle: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: colors.primarySoft,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.md,
  },
  emptyTitle: {
    fontSize: typography.fontSize.lg,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.ink,
  },
  emptyHint: {
    marginTop: spacing.xs,
    fontSize: typography.fontSize.sm,
    fontFamily: typography.fontFamily.regular,
    color: colors.muted,
  },
  emptyCta: {
    marginTop: spacing.lg,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.xl,
    paddingVertical: 13,
    borderRadius: radius.md,
    ...shadows.md,
  },
  emptyCtaText: { color: "#fff", fontSize: typography.fontSize.sm, fontFamily: typography.fontFamily.latinBold },

  // Stats strip
  statsRow: { flexDirection: "row", gap: spacing.sm, marginBottom: spacing.md },
  statCard: {
    flex: 1,
    alignItems: "center",
    gap: 3,
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.line,
    paddingVertical: spacing.md - 2,
    ...shadows.sm,
  },
  statValue: { fontSize: typography.fontSize.xl, fontFamily: typography.fontFamily.bold },
  statLabel: {
    fontSize: typography.fontSize.xs - 1,
    color: colors.inkSubtle,
    maxWidth: "90%",
  },
  listTitle: {
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.latinBold,
    color: colors.inkSubtle,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: spacing.sm,
    paddingHorizontal: spacing.xs,
  },

  // Rows
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    padding: spacing.sm + 4,
    borderWidth: 1,
    borderColor: colors.line,
    ...shadows.sm,
  },
  thumb: { width: 56, height: 56, borderRadius: radius.sm, backgroundColor: colors.line },
  rowMain: { flex: 1, gap: 5 },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    alignSelf: "flex-start",
    paddingHorizontal: spacing.sm,
    paddingVertical: 3,
    borderRadius: radius.full,
  },
  chipText: { fontSize: typography.fontSize.xs + 1, fontFamily: typography.fontFamily.bold },
  sub: {
    fontSize: typography.fontSize.xs,
    fontFamily: typography.fontFamily.regular,
    color: colors.muted,
  },
  queuedDotWrap: { paddingRight: spacing.xs },
  queuedDot: {
    width: 9,
    height: 9,
    borderRadius: 4.5,
    backgroundColor: colors.green,
  },
  separator: { height: spacing.sm },

  // Clear button
  clearBtn: {
    position: "absolute",
    alignSelf: "center",
    left: 0,
    right: 0,
    bottom: 0,
    marginHorizontal: spacing.xl,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs,
    paddingVertical: 11,
    borderRadius: radius.full,
    borderWidth: 1,
    borderColor: colors.redSoft,
    backgroundColor: colors.redSoft,
  },
  clearBtnText: { color: colors.red, fontSize: typography.fontSize.sm, fontFamily: typography.fontFamily.latinBold },

  fontAm: { fontFamily: typography.fontFamily.regular },
  fontAmBold: { fontFamily: typography.fontFamily.bold },
});
