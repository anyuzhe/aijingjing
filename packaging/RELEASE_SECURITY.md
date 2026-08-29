# Release security

Official installers must be produced by the GitHub Actions workflow or an isolated release
machine. CI build artifacts are intentionally unsigned and must not be advertised as a public
release.

## macOS

1. Build the `.app` with `pyinstaller packaging/AI-Jingjing.spec`.
2. Import a Developer ID Application certificate into the build keychain.
3. Set `APPLE_SIGNING_IDENTITY` and one notarization credential method:
   `APPLE_NOTARY_PROFILE`; `APPLE_API_KEY_PATH` + `APPLE_API_KEY_ID` +
   `APPLE_API_ISSUER_ID`; or `APPLE_ID` + `APPLE_TEAM_ID` + `APPLE_APP_PASSWORD`.
4. Run `packaging/sign_and_notarize.sh`. The script signs with hardened runtime, verifies the
   signature, submits to Apple, staples the ticket, validates the staple, and runs Gatekeeper.
5. Create the DMG only after stapling the app. Sign and notarize the final DMG as well.

`SKIP_NOTARIZATION=1` exists only for local diagnostics. It must never be used for an official
release.

## Windows

Sign the final installer with an organization-controlled Authenticode certificate and an RFC
3161 timestamp. Verify with `Get-AuthenticodeSignature` before publishing. EV/private keys must
live in a hardware token or managed signing service and must never be committed or stored in a
plain CI variable.

## Update manifest

Generate SHA-256 from the exact published installer and place it in `update.json`. Publish both
the manifest and installer over HTTPS. AI静静 rejects malformed semantic versions, non-HTTPS
URLs, missing checksums, and any downloaded file whose digest differs from the manifest.

For a production update channel, sign `update.json` with an offline release key and verify that
signature before parsing it. SHA-256 protects the package only after the manifest itself is
authenticated; TLS and repository access controls remain part of the trust boundary.
