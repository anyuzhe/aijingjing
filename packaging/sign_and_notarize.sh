#!/bin/zsh
set -euo pipefail

app_path="${1:-dist/AI知识库-AI静静.app}"
signing_identity="${APPLE_SIGNING_IDENTITY:-}"

if [[ ! -d "$app_path" ]]; then
  print -u2 "应用不存在：$app_path"
  exit 2
fi
if [[ -z "$signing_identity" ]]; then
  print -u2 "请设置 APPLE_SIGNING_IDENTITY 后再执行正式签名。"
  exit 2
fi

codesign --force --deep --options runtime --timestamp --sign "$signing_identity" "$app_path"
codesign --verify --deep --strict --verbose=2 "$app_path"

if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
  archive_path="${app_path%.app}.zip"
  ditto -c -k --keepParent "$app_path" "$archive_path"
  xcrun notarytool submit "$archive_path" --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
  xcrun stapler staple "$app_path"
  spctl --assess --type execute --verbose=2 "$app_path"
fi
