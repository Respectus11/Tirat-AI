const { FlatCompat } = require("@eslint/eslintrc");

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

module.exports = [
  ...compat.extends("expo"),
  {
    ignores: ["node_modules/**", "dist/**", ".expo/**", "web-build/**"],
    linterOptions: {
      reportUnusedDisableDirectives: false,
    },
    rules: {
      "react-compiler/react-compiler": "off",
      "react-hooks/refs": "off",
      "react-hooks/exhaustive-deps": "off",
      "@typescript-eslint/no-unused-vars": "off",
    },
  },
];
