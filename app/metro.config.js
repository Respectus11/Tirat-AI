const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// `.tflite` must be treated as an asset (not JS) so `require("./model.tflite")`
// resolves to a bundled file that react-native-fast-tflite can load natively.
config.resolver.assetExts.push("tflite");

module.exports = config;
