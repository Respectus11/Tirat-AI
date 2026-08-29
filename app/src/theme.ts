// Design system tokens for Tirat AI.
// Built for high legibility under harsh Ethiopian sun & outdoor grain markets.

export const colors = {
  // Brand
  primary: "#1F7A45",          // Ethiopian highlands deep forest green
  primaryLight: "#2E9E5B",     // Brighter emerald green for active states
  primaryDark: "#134F2C",      // Deep bottle green
  primarySoft: "#E8F5EC",      // Gentle green tint for badges and highlights
  green: "#1F7A45",            // backward-compat alias
  greenSoft: "#E8F5EC",        // backward-compat alias

  // Secondary & Accents
  amber: "#D97706",            // Warm teff golden grain / warning
  amberLight: "#F59E0B",
  amberSoft: "#FEF3C7",
  red: "#DC2626",              // Adulterated alert crimson
  redSoft: "#FEE2E2",
  gold: "#EAB308",             // Highlight gold

  // Neutrals & Surfaces
  bg: "#F9F8F5",               // Natural warm eggshell / teff-flour cream
  surface: "#FFFFFF",          // Card surface
  surfaceElevated: "#FFFFFF",  // Floating modal / sheet
  surfaceMuted: "#F3F1EC",     // Subdued card fill
  line: "#E7E4DC",             // Card / row borders
  lineStrong: "#D1CBBF",

  // Typography
  ink: "#1E1E1B",              // Near black high contrast
  inkMuted: "#5E5B52",         // Secondary text (meets 4.5:1 AA contrast on bg)
  inkSubtle: "#8A867B",        // De-emphasized metadata
  muted: "#5E5B52",            // backward-compat alias
  white: "#FFFFFF",
  black: "#000000",
} as const;

export const typography = {
  fontFamily: {
    regular: "NotoSansEthiopic_400Regular",
    medium: "NotoSansEthiopic_500Medium",
    bold: "NotoSansEthiopic_700Bold",
    latinRegular: "Inter_400Regular",
    latinMedium: "Inter_500Medium",
    latinBold: "Inter_700Bold",
  },
  fontSize: {
    xs: 11,
    sm: 13,
    md: 15,
    lg: 17,
    xl: 20,
    xxl: 24,
    display: 32,
  },
  lineHeight: {
    tight: 1.2,
    normal: 1.45,
    relaxed: 1.6,
  },
} as const;

export const spacing = {
  xxs: 2,
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export const radius = {
  xs: 4,
  sm: 8,
  md: 14,
  lg: 20,
  xl: 28,
  full: 9999,
} as const;

export const shadows = {
  sm: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 2,
    elevation: 1,
  },
  md: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 3,
  },
  lg: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 12,
    elevation: 6,
  },
} as const;

