#!/bin/zsh
set -euo pipefail

app_path="${1:-dist/AI知识库-AI静静.app}"
signing_identity="${APPLE_SIGNING_IDENTITY:-}"
entitlements_path="${APPLE_ENTITLEMENTS:-packaging/entitlements.plist}"
notary_archive="${app_path%.app}-notarization.zip"

if [[ ! -d "$app_path" ]]; then
  print -u2 "应用不存在：$app_path"
  exit 2
fi
if [[ -z "$signing_identity" ]]; then
  print -u2 "请设置 APPLE_SIGNING_IDENTITY 后再执行正式签名。"
  exit 2
fi
if [[ ! -f "$entitlements_path" ]]; then
  print -u2 "签名权限文件不存在：$entitlements_path"
  exit 2
fi

codesign \
  --force --deep --options runtime --timestamp \
  --entitlements "$entitlements_path" \
  --sign "$signing_identity" "$app_path"
codesign --verify --deep --strict --verbose=2 "$app_path"
codesign --display --verbose=4 "$app_path"

notary_args=()
if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
  notary_args=(--keychain-profile "$APPLE_NOTARY_PROFILE")
elif [[ -n "${APPLE_API_KEY_PATH:-}" && -n "${APPLE_API_KEY_ID:-}" && -n "${APPLE_API_ISSUER_ID:-}" ]]; then
  notary_args=(--key "$APPLE_API_KEY_PATH" --key-id "$APPLE_API_KEY_ID" --issuer "$APPLE_API_ISSUER_ID")
elif [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_PASSWORD:-}" ]]; then
  notary_args=(--apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD")
elif [[ "${SKIP_NOTARIZATION:-0}" != "1" ]]; then
  print -u2 "缺少公证凭据。请配置 APPLE_NOTARY_PROFILE、App Store Connect API Key，或 Apple ID 凭据。"
  print -u2 "仅限本地调试时可显式设置 SKIP_NOTARIZATION=1。"
  exit 2
fi

if (( ${#notary_args[@]} > 0 )); then
  rm -f "$notary_archive"
  ditto -c -k --keepParent "$app_path" "$notary_archive"
  xcrun notarytool submit "$notary_archive" "${notary_args[@]}" --wait
  rm -f "$notary_archive"
  xcrun stapler staple "$app_path"
  xcrun stapler validate "$app_path"
  spctl --assess --type execute --verbose=2 "$app_path"
fi

print "签名验证完成：$app_path"
