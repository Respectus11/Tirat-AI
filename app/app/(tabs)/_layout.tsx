// Bottom tabs layout: Scan | History | Settings with Ionicons and styled bar.

import { Tabs } from "expo-router";
import React from "react";
import { Ionicons } from "@expo/vector-icons";
import { useLocale } from "../../src/i18n";
import { colors, typography } from "../../src/theme";

export default function TabsLayout() {
  const { t, lang } = useLocale();
  const isAmharic = lang === "am";

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.inkSubtle,
        tabBarStyle: {
          backgroundColor: colors.surface,
          borderTopColor: colors.line,
          height: 60,
          paddingBottom: 8,
          paddingTop: 6,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontFamily: isAmharic ? typography.fontFamily.medium : typography.fontFamily.latinMedium,
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: t("tab_capture"),
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons
              name={focused ? "camera" : "camera-outline"}
              size={size + 2}
              color={color}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: t("tab_history"),
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons
              name={focused ? "time" : "time-outline"}
              size={size + 2}
              color={color}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: t("tab_settings"),
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons
              name={focused ? "settings" : "settings-outline"}
              size={size + 2}
              color={color}
            />
          ),
        }}
      />
    </Tabs>
  );
}

