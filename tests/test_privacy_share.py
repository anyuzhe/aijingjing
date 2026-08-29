from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

import media_knowledge.desktop.privacy as privacy_module
from media_knowledge.desktop.privacy import (
    PrivacyViolationError,
    ShareCopyOptions,
    create_share_copy,
    scan_privacy,
    verify_share_copy,
)


def _fake_provider_key(label: str) -> str:
    """Create a scanner fixture without committing a key-shaped literal."""

    return "sk-" + hashlib.sha256(label.encode()).hexdigest()


class PrivacyScanTests(unittest.TestCase):
    def test_secret_patterns_and_sensitive_paths_are_reported_without_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scan-root"
            sensitive_directory = root / "owner@example.com"
            sensitive_directory.mkdir(parents=True)
            secret = _fake_provider_key("pattern-test")
            email = "owner@example.com"
            phone = "13800138000"
            absolute_path = "/Users/private-owner/Documents/secret.txt"
            payload = "\n".join(
                [
                    secret,
                    email,
                    phone,
                    absolute_path,
                    "-----BEGIN " + "PRIVATE KEY-----",
                    "password=CorrectHorseBatteryStaple",
                ]
            )
            (sensitive_directory / "credentials.json").write_text(payload, encoding="utf-8")

            report = scan_privacy(root)
            serialized = json.dumps(report.to_dict(), ensure_ascii=False)
            categories = {finding.category for finding in report.findings}

            self.assertEqual(report.status, "blocked")
            self.assertTrue(report.has_blockers)
            self.assertTrue(
                {
                    "sensitive_path",
                    "secret_like_file",
                    "provider_api_key",
                    "email_address",
                    "phone_number",
                    "absolute_user_path",
                    "private_key",
                    "credential_assignment",
                }.issubset(categories)
            )
            self.assertIn("<敏感路径-", serialized)
            for private_value in (
                secret,
                email,
                phone,
                absolute_path,
                "private-owner",
                "CorrectHorseBatteryStaple",
                str(root),
            ):
                self.assertNotIn(private_value, serialized)
                self.assertNotIn(private_value, repr(report))

    def test_clean_text_is_clean_but_report_retains_the_general_limitation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            root.mkdir()
            (root / "public.md").write_text(
                "# Public knowledge\nThis note explains winter weather.", encoding="utf-8"
            )

            report = scan_privacy(root)

            self.assertEqual(report.status, "clean")
            self.assertEqual(report.scanned_files, 1)
            self.assertEqual(report.text_files_scanned, 1)
            self.assertFalse(report.has_blockers)
            self.assertTrue(any("不能证明" in item for item in report.limitations))

    def test_unparsed_pdf_image_office_audio_and_video_have_explicit_limitations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mixed"
            root.mkdir()
            (root / "document.pdf").write_bytes(b"%PDF-1.7\n")
            (root / "broken.png").write_bytes(b"not-a-real-image")
            (root / "lesson.docx").write_bytes(b"not-a-real-office-package")
            (root / "voice.m4a").write_bytes(b"audio")
            (root / "movie.mp4").write_bytes(b"video")

            report = scan_privacy(root)
            categories = {finding.category for finding in report.findings}
            limitations = "\n".join(report.limitations)

            self.assertEqual(report.status, "review")
            self.assertIn("pdf_content_unscanned", categories)
            self.assertIn("image_text_unscanned", categories)
            self.assertIn("office_content_unscanned", categories)
            self.assertIn("media_content_unscanned", categories)
            self.assertIn("PDF", limitations)
            self.assertIn("OCR", limitations)
            self.assertIn("音频和视频", limitations)
            self.assertIn("Office", limitations)

    def test_optional_local_ocr_and_exif_are_scanned_without_leaking_values(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "images"
            root.mkdir()
            image_path = root / "photo.jpg"
            image = Image.new("RGB", (4, 4), "white")
            exif = image.getexif()
            exif[315] = "metadata-owner@example.com"
            image.save(image_path, exif=exif)
            observed: list[Path] = []

            def local_ocr(path: Path) -> dict[str, str]:
                observed.append(path)
                return {"text": "请联系 13900139000 或 ocr-owner@example.com"}

            report = scan_privacy(root, enable_image_ocr=True, ocr_reader=local_ocr)
            serialized = json.dumps(report.to_dict(), ensure_ascii=False)
            categories = {finding.category for finding in report.findings}

            self.assertEqual(observed, [image_path])
            self.assertEqual(report.ocr_images_scanned, 1)
            self.assertIn("image_exif", categories)
            self.assertIn("image_metadata_email_address", categories)
            self.assertIn("image_ocr_phone_number", categories)
            self.assertIn("image_ocr_email_address", categories)
            self.assertNotIn("metadata-owner@example.com", serialized)
            self.assertNotIn("ocr-owner@example.com", serialized)
            self.assertNotIn("13900139000", serialized)

    def test_ocr_failure_message_cannot_escape_into_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "images"
            root.mkdir()
            (root / "image.png").write_bytes(b"invalid image")
            leaked = _fake_provider_key("ocr-exception")

            def failed_ocr(_path: Path) -> str:
                raise RuntimeError(f"OCR failed for {leaked} at /Users/private/path")

            report = scan_privacy(root, enable_image_ocr=True, ocr_reader=failed_ocr)
            serialized = json.dumps(report.to_dict(), ensure_ascii=False)

            self.assertIn("image_ocr_failed", {item.category for item in report.findings})
            self.assertNotIn(leaked, serialized)
            self.assertNotIn("/Users/private/path", serialized)

    def test_empty_or_unavailable_ocr_can_never_make_an_image_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "images"
            root.mkdir()
            (root / "image.png").write_bytes(b"not-a-real-image")

            unavailable = scan_privacy(
                root,
                enable_image_ocr=True,
                ocr_reader=lambda _path: {
                    "engine": "none",
                    "lines": [],
                    "fallback_reasons": ["OCR engine missing"],
                },
            )
            empty = scan_privacy(
                root,
                enable_image_ocr=True,
                ocr_reader=lambda _path: {"engine": "rapidocr", "lines": [], "text": ""},
            )

            self.assertEqual(unavailable.status, "review")
            self.assertIn("image_ocr_unavailable", unavailable.category_counts)
            self.assertEqual(empty.status, "review")
            self.assertIn("image_ocr_empty", empty.category_counts)

    def test_symlinks_are_not_followed(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scan"
            root.mkdir()
            outside = Path(temporary) / "outside.txt"
            outside.write_text("password=OutsideSecretValue", encoding="utf-8")
            try:
                (root / "linked.txt").symlink_to(outside)
            except OSError:
                self.skipTest("symbolic links are not permitted")

            report = scan_privacy(root)
            serialized = json.dumps(report.to_dict(), ensure_ascii=False)

            self.assertEqual(report.status, "blocked")
            self.assertIn("symbolic_link", {item.category for item in report.findings})
            self.assertNotIn("OutsideSecretValue", serialized)


class SecureShareCopyTests(unittest.TestCase):
    def _product_root(self, parent: Path) -> Path:
        root = parent / "product-data"
        for name in ("notes", "archive", "assets", "transcripts", "cache"):
            (root / name).mkdir(parents=True, exist_ok=True)
        return root

    def test_default_share_is_empty_safe_skeleton_and_excludes_private_product_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            secret = _fake_provider_key("default-share")
            (root / "notes" / "private.md").write_text(secret, encoding="utf-8")
            (root / "archive" / "private.mp4").write_bytes(b"private-video")
            (root / "providers.json").write_text(secret, encoding="utf-8")
            (root / "settings.json").write_text('{"private":true}', encoding="utf-8")
            (root / "knowledge.db").write_bytes(b"conversations")
            (root / "cache" / "model.bin").write_bytes(b"cache")
            destination = base / "safe-share"

            report = create_share_copy(root, destination)
            manifest = json.loads((destination / "share_manifest.json").read_text(encoding="utf-8"))
            serialized = json.dumps(manifest, ensure_ascii=False)

            self.assertEqual(report.status, "created")
            self.assertEqual(report.file_count, 0)
            self.assertEqual(report.total_bytes, 0)
            self.assertEqual(
                {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()},
                {"share_manifest.json"},
            )
            self.assertFalse(manifest["external_delivery"])
            self.assertFalse(manifest["contains_conversations"])
            self.assertFalse(manifest["contains_credentials"])
            self.assertFalse(manifest["contains_raw_media"])
            self.assertNotIn(secret, serialized)
            self.assertNotIn(str(root), serialized)
            self.assertEqual(verify_share_copy(destination)["status"], "verified")
            safe_dict = report.to_dict()
            self.assertNotIn(str(destination), json.dumps(safe_dict, ensure_ascii=False))

    def test_opted_in_notes_and_public_sources_have_exact_hash_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            expected = {
                "notes/public.md": b"Public winter knowledge.",
                "archive/public.txt": b"Public source evidence.",
            }
            for relative, payload in expected.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            # Automatic notes selection must silently omit fixed private state.
            (root / "notes" / "cache").mkdir()
            (root / "notes" / "cache" / "private.txt").write_text(
                "private cache", encoding="utf-8"
            )
            (root / "notes" / "settings.json").write_text(
                '{"token":"must-not-copy"}', encoding="utf-8"
            )
            destination = base / "reviewed-share"

            report = create_share_copy(
                root,
                destination,
                options=ShareCopyOptions(
                    include_notes=True,
                    public_sources=("archive/public.txt",),
                ),
            )
            manifest = json.loads((destination / "share_manifest.json").read_text(encoding="utf-8"))
            records = {item["path"]: item for item in manifest["files"]}

            self.assertEqual(set(records), set(expected))
            self.assertEqual(report.file_count, 2)
            for relative, payload in expected.items():
                self.assertEqual((destination / relative).read_bytes(), payload)
                self.assertEqual(records[relative]["size"], len(payload))
                self.assertEqual(records[relative]["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertFalse((destination / "notes" / "cache").exists())
            self.assertFalse((destination / "notes" / "settings.json").exists())
            verified = verify_share_copy(destination)
            self.assertEqual(verified["file_count"], 2)
            self.assertEqual(verified["manifest_sha256"], report.manifest_sha256)

    def test_dirty_selected_note_blocks_atomically_and_exception_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            leaked = _fake_provider_key("dirty-note")
            (root / "notes" / "private.md").write_text(
                f"api_key={leaked}\nowner@example.com\n13800138000", encoding="utf-8"
            )
            destination = base / "blocked-share"

            with self.assertRaises(PrivacyViolationError) as raised:
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(include_notes=True),
                )

            serialized = json.dumps(raised.exception.report.to_dict(), ensure_ascii=False)
            self.assertFalse(destination.exists())
            self.assertFalse(list(base.glob(".ai-jingjing-share-*")))
            self.assertNotIn(leaked, str(raised.exception))
            self.assertNotIn(leaked, serialized)
            self.assertNotIn("owner@example.com", serialized)
            self.assertNotIn("13800138000", serialized)

            # The expert option may acknowledge incomplete media coverage, but
            # must never override a positively detected credential/PII blocker.
            with self.assertRaises(PrivacyViolationError):
                create_share_copy(
                    root,
                    base / "still-blocked-share",
                    options=ShareCopyOptions(include_notes=True, require_clean_scan=False),
                )

    def test_markdown_with_opaque_binary_tail_is_never_copied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            secret = _fake_provider_key("markdown-compressed-tail")
            note = root / "notes" / "public.md"
            note.write_bytes(
                b"# Public winter note\n\nVisible text.\n"
                + zlib.compress(secret.encode("utf-8"))
            )
            destination = base / "blocked-binary-tail"

            with self.assertRaises(PrivacyViolationError) as raised:
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(
                        include_notes=True,
                        require_clean_scan=False,
                    ),
                )

            self.assertIn("binary_content_unscanned", raised.exception.report.category_counts)
            self.assertNotIn(
                secret,
                json.dumps(raised.exception.report.to_dict(), ensure_ascii=False),
            )
            self.assertFalse(destination.exists())

    def test_original_image_with_compressed_tail_is_fail_closed(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional")

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            secret = _fake_provider_key("image-compressed-tail")
            image_path = root / "assets" / "public.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            with image_path.open("ab") as stream:
                stream.write(zlib.compress(secret.encode("utf-8")))
            destination = base / "blocked-original-image"

            with self.assertRaises(PrivacyViolationError) as raised:
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(
                        public_sources=("assets/public.png",),
                        scan_images_with_ocr=True,
                        require_clean_scan=False,
                    ),
                    ocr_reader=lambda _path: {
                        "engine": "rapidocr",
                        "text": "Public white image",
                    },
                )

            self.assertIn(
                "image_original_container_not_shareable",
                raised.exception.report.category_counts,
            )
            self.assertNotIn(
                secret,
                json.dumps(raised.exception.report.to_dict(), ensure_ascii=False),
            )
            self.assertFalse(destination.exists())

    def test_expert_flag_cannot_include_an_unparsed_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            (root / "archive" / "public.pdf").write_bytes(b"%PDF-1.7\npublic placeholder")
            destination = base / "review-required-share"

            with self.assertRaises(PrivacyViolationError):
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(public_sources=("archive/public.pdf",)),
                )

            with self.assertRaises(PrivacyViolationError) as raised:
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(
                        public_sources=("archive/public.pdf",),
                        require_clean_scan=False,
                    ),
                )

            self.assertIn("pdf_content_unscanned", raised.exception.report.category_counts)
            self.assertFalse(destination.exists())

    def test_dynamic_key_in_pdf_or_office_blocks_even_with_expert_flag(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is optional")

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            pdf_secret = _fake_provider_key("pdf-inside-document")
            pdf_path = root / "archive" / "key-in-page.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), pdf_secret)
                document.save(pdf_path)

            office_secret = _fake_provider_key("office-inside-document")
            office_path = root / "archive" / "key-in-runs.docx"
            split = len(office_secret) // 2
            with zipfile.ZipFile(office_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
                package.writestr(
                    "[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
                )
                package.writestr(
                    "word/document.xml",
                    '<?xml version="1.0"?><w:document '
                    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    f"<w:body><w:p><w:r><w:t>{office_secret[:split]}</w:t></w:r>"
                    f"<w:r><w:t>{office_secret[split:]}</w:t></w:r></w:p></w:body></w:document>",
                )

            for index, (relative, secret) in enumerate(
                (
                    ("archive/key-in-page.pdf", pdf_secret),
                    ("archive/key-in-runs.docx", office_secret),
                )
            ):
                with self.subTest(relative=relative):
                    destination = base / f"blocked-document-{index}"
                    with self.assertRaises(PrivacyViolationError) as raised:
                        create_share_copy(
                            root,
                            destination,
                            options=ShareCopyOptions(
                                public_sources=(relative,),
                                require_clean_scan=False,
                            ),
                        )
                    serialized = json.dumps(raised.exception.report.to_dict(), ensure_ascii=False)
                    self.assertIn("provider_api_key", raised.exception.report.category_counts)
                    self.assertNotIn(secret, serialized)
                    self.assertFalse(destination.exists())

    def test_plain_documents_are_scanned_but_original_containers_are_blocked(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is optional")

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            pdf_path = root / "archive" / "public.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Public winter knowledge")
                document.save(pdf_path)
            office_path = root / "archive" / "public.docx"
            with zipfile.ZipFile(office_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
                package.writestr(
                    "[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
                )
                package.writestr(
                    "word/document.xml",
                    '<?xml version="1.0"?><w:document '
                    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    "<w:body><w:p><w:r><w:t>Public knowledge</w:t></w:r></w:p></w:body></w:document>",
                )
                package.writestr(
                    "_rels/.rels",
                    '<?xml version="1.0"?><Relationships '
                    'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="public-document" Target="word/document.xml"/>'
                    "</Relationships>",
                )

            for index, (relative, category) in enumerate(
                (
                    ("archive/public.pdf", "pdf_original_container_not_shareable"),
                    ("archive/public.docx", "office_original_container_not_shareable"),
                )
            ):
                with self.subTest(relative=relative):
                    destination = base / f"blocked-original-document-{index}"
                    with self.assertRaises(PrivacyViolationError) as raised:
                        create_share_copy(
                            root,
                            destination,
                            options=ShareCopyOptions(
                                public_sources=(relative,),
                                require_clean_scan=False,
                            ),
                        )
                    self.assertIn(category, raised.exception.report.category_counts)
                    self.assertEqual(raised.exception.report.text_files_scanned, 1)
                    self.assertFalse(destination.exists())

    def test_unreferenced_flate_pdf_stream_is_fail_closed_without_destination(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is optional")

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            secret = _fake_provider_key("pdf-unreferenced-flate-stream")
            pdf_path = root / "archive" / "hidden-stream.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Public PDF")
                document.save(pdf_path)
            compressed = zlib.compress(secret.encode())
            with pdf_path.open("ab") as stream:
                stream.write(
                    b"\n997 0 obj\n<< /Length "
                    + str(len(compressed)).encode()
                    + b" /Filter /FlateDecode >>\nstream\n"
                    + compressed
                    + b"\nendstream\nendobj\n"
                )

            destination = base / "must-not-copy-original-pdf"
            with self.assertRaises(PrivacyViolationError) as raised:
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(
                        public_sources=("archive/hidden-stream.pdf",),
                        require_clean_scan=False,
                    ),
                )

            serialized = json.dumps(raised.exception.report.to_dict(), ensure_ascii=False)
            self.assertIn(
                "pdf_original_container_not_shareable",
                raised.exception.report.category_counts,
            )
            self.assertNotIn(secret, serialized)
            self.assertFalse(destination.exists())

    def test_pdf_unused_object_and_eof_tail_key_block_share(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is optional")

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            for index, location in enumerate(
                ("unused-object", "eof-comment", "utf16-tail", "hex-object")
            ):
                with self.subTest(location=location):
                    secret = _fake_provider_key(f"pdf-{location}")
                    pdf_path = root / "archive" / f"raw-{index}.pdf"
                    with pymupdf.open() as document:
                        page = document.new_page()
                        page.insert_text((72, 72), "Public PDF")
                        document.save(pdf_path)
                    if location == "unused-object":
                        suffix = f"\n999 0 obj\n({secret})\nendobj\n".encode()
                    elif location == "utf16-tail":
                        suffix = b"\n% utf16 trailing note: " + secret.encode("utf-16-le")
                    elif location == "hex-object":
                        suffix = f"\n998 0 obj\n<{secret.encode().hex()}>\nendobj\n".encode()
                    else:
                        suffix = f"\n% private trailing note: {secret}\n".encode()
                    with pdf_path.open("ab") as stream:
                        stream.write(suffix)

                    destination = base / f"blocked-raw-pdf-{index}"
                    with self.assertRaises(PrivacyViolationError) as raised:
                        create_share_copy(
                            root,
                            destination,
                            options=ShareCopyOptions(
                                public_sources=(f"archive/raw-{index}.pdf",),
                                require_clean_scan=False,
                            ),
                        )
                    serialized = json.dumps(raised.exception.report.to_dict(), ensure_ascii=False)
                    self.assertIn("provider_api_key", raised.exception.report.category_counts)
                    self.assertNotIn(secret, serialized)
                    self.assertFalse(destination.exists())

    def test_office_zip_comment_and_eof_tail_key_block_share(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            for index, location in enumerate(("zip-comment", "eof-tail")):
                with self.subTest(location=location):
                    secret = _fake_provider_key(f"office-{location}")
                    office_path = root / "archive" / f"raw-{index}.docx"
                    with zipfile.ZipFile(
                        office_path, "w", compression=zipfile.ZIP_DEFLATED
                    ) as package:
                        package.writestr(
                            "[Content_Types].xml",
                            '<?xml version="1.0"?><Types '
                            'xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
                        )
                        package.writestr(
                            "word/document.xml",
                            '<?xml version="1.0"?><w:document '
                            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                            "<w:body><w:p><w:r><w:t>Public text</w:t></w:r></w:p></w:body></w:document>",
                        )
                        if location == "zip-comment":
                            package.comment = secret.encode()
                    if location == "eof-tail":
                        with office_path.open("ab") as stream:
                            stream.write(f"\nprivate trailing note: {secret}\n".encode())

                    destination = base / f"blocked-raw-office-{index}"
                    with self.assertRaises(PrivacyViolationError) as raised:
                        create_share_copy(
                            root,
                            destination,
                            options=ShareCopyOptions(
                                public_sources=(f"archive/raw-{index}.docx",),
                                require_clean_scan=False,
                            ),
                        )
                    serialized = json.dumps(raised.exception.report.to_dict(), ensure_ascii=False)
                    self.assertIn("provider_api_key", raised.exception.report.category_counts)
                    self.assertNotIn(secret, serialized)
                    self.assertFalse(destination.exists())

    def test_office_trailing_compressed_payload_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            secret = _fake_provider_key("office-trailing-compressed-payload")
            office_path = root / "archive" / "hidden-compressed.docx"
            with zipfile.ZipFile(office_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
                package.writestr(
                    "[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
                )
                package.writestr(
                    "word/document.xml",
                    '<?xml version="1.0"?><w:document '
                    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    "<w:body><w:p><w:r><w:t>Public text</w:t></w:r></w:p></w:body></w:document>",
                )
            with office_path.open("ab") as stream:
                stream.write(b"\n" + zlib.compress(secret.encode()) + b"\n")

            destination = base / "must-not-copy-original-office"
            with self.assertRaises(PrivacyViolationError) as raised:
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(
                        public_sources=("archive/hidden-compressed.docx",),
                        require_clean_scan=False,
                    ),
                )

            serialized = json.dumps(raised.exception.report.to_dict(), ensure_ascii=False)
            self.assertIn(
                "office_original_container_not_shareable",
                raised.exception.report.category_counts,
            )
            self.assertNotIn(secret, serialized)
            self.assertFalse(destination.exists())

    def test_office_thumbnail_can_be_verified_with_nonempty_local_ocr(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional")

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            thumbnail_path = base / "thumbnail.jpeg"
            Image.new("RGB", (8, 8), "white").save(thumbnail_path)
            office_path = root / "archive" / "thumbnail.docx"
            with zipfile.ZipFile(office_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
                package.writestr(
                    "[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
                )
                package.writestr(
                    "word/document.xml",
                    '<?xml version="1.0"?><w:document '
                    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    "<w:body><w:p><w:r><w:t>Public text</w:t></w:r></w:p></w:body></w:document>",
                )
                package.writestr("docProps/thumbnail.jpeg", thumbnail_path.read_bytes())

            report = scan_privacy(
                office_path,
                enable_image_ocr=True,
                ocr_reader=lambda _path: {"engine": "rapidocr", "text": "Public thumbnail"},
            )

            self.assertEqual(report.status, "review")
            self.assertIn("office_original_container_not_shareable", report.category_counts)
            self.assertEqual(report.ocr_images_scanned, 1)

    def test_expert_override_never_allows_incomplete_or_truncated_text_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            large_text = root / "archive" / "large.txt"
            large_text.write_bytes(
                b"public-padding\n" * (privacy_module.MAX_TEXT_BYTES // 15 + 1)
                + b"password=HiddenBeyondTheScanLimit"
            )
            destination = base / "must-not-exist"

            with self.assertRaises(PrivacyViolationError) as raised:
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(
                        public_sources=("archive/large.txt",),
                        require_clean_scan=False,
                    ),
                )

            self.assertIn("text_too_large", raised.exception.report.category_counts)
            self.assertFalse(destination.exists())
            self.assertNotIn(
                "HiddenBeyondTheScanLimit",
                json.dumps(raised.exception.report.to_dict(), ensure_ascii=False),
            )

    def test_expert_flag_cannot_include_untranscribed_media_or_unknown_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            (root / "archive" / "lesson.mp4").write_bytes(b"uninspected-media")
            (root / "archive" / "opaque.bin").write_bytes(b"\x00\x01\x02\x03")

            for index, (relative, category) in enumerate(
                (
                    ("archive/lesson.mp4", "media_content_unscanned"),
                    ("archive/opaque.bin", "binary_content_unscanned"),
                )
            ):
                with self.subTest(relative=relative):
                    destination = base / f"blocked-opaque-{index}"
                    with self.assertRaises(PrivacyViolationError) as raised:
                        create_share_copy(
                            root,
                            destination,
                            options=ShareCopyOptions(
                                public_sources=(relative,),
                                require_clean_scan=False,
                            ),
                        )
                    self.assertIn(category, raised.exception.report.category_counts)
                    self.assertFalse(destination.exists())

    def test_traversal_absolute_forbidden_and_in_tree_destinations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            (root / "archive" / "public.txt").write_text("public", encoding="utf-8")
            selections = (
                "../outside.txt",
                str((base / "outside.txt").resolve()),
                "providers.json",
                "settings.json",
                "knowledge.db",
                "cache",
                "conversations.json",
            )
            for index, selected in enumerate(selections):
                with self.subTest(selected=selected):
                    destination = base / f"rejected-{index}"
                    with self.assertRaises(ValueError):
                        create_share_copy(
                            root,
                            destination,
                            options=ShareCopyOptions(public_sources=(selected,)),
                        )
                    self.assertFalse(destination.exists())

            with self.assertRaisesRegex(ValueError, "不能位于"):
                create_share_copy(root, root / "share-inside-product")

    def test_selected_symlink_and_destination_symlink_are_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            outside = base / "outside.txt"
            outside.write_text("public-looking", encoding="utf-8")
            try:
                (root / "notes" / "linked.txt").symlink_to(outside)
            except OSError:
                self.skipTest("symbolic links are not permitted")

            destination = base / "symlink-source-share"
            with self.assertRaisesRegex(ValueError, "符号链接"):
                create_share_copy(
                    root,
                    destination,
                    options=ShareCopyOptions(include_notes=True),
                )
            self.assertFalse(destination.exists())

            real_output = base / "real-output"
            real_output.mkdir()
            output_link = base / "output-link"
            output_link.symlink_to(real_output, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "符号链接"):
                create_share_copy(root, output_link / "share")
            self.assertFalse((real_output / "share").exists())

    def test_copy_failure_is_cleaned_and_underlying_secret_error_is_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            (root / "notes" / "public.md").write_text("public", encoding="utf-8")
            destination = base / "failed-share"
            leaked = _fake_provider_key("copy-error")

            with patch.object(
                privacy_module,
                "_copy_file_and_hash",
                side_effect=RuntimeError(f"failure {leaked} /Users/private/path"),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    create_share_copy(
                        root,
                        destination,
                        options=ShareCopyOptions(include_notes=True),
                    )

            self.assertFalse(destination.exists())
            self.assertFalse(list(base.glob(".ai-jingjing-share-*")))
            self.assertNotIn(leaked, str(raised.exception))
            self.assertNotIn("/Users/private/path", str(raised.exception))

    def test_raced_in_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            (root / "notes" / "public.md").write_text("public", encoding="utf-8")
            destination = base / "raced-destination"
            original_verify = privacy_module._verify_staged_files

            def race_after_verification(stage: Path, records: object) -> None:
                original_verify(stage, records)  # type: ignore[arg-type]
                destination.mkdir()
                (destination / "owner-file.txt").write_text("do not replace", encoding="utf-8")

            with patch.object(
                privacy_module,
                "_verify_staged_files",
                side_effect=race_after_verification,
            ):
                with self.assertRaises(ValueError):
                    create_share_copy(
                        root,
                        destination,
                        options=ShareCopyOptions(include_notes=True),
                    )

            self.assertEqual(
                (destination / "owner-file.txt").read_text(encoding="utf-8"),
                "do not replace",
            )
            self.assertFalse((destination / "share_manifest.json").exists())
            self.assertFalse(list(base.glob(".ai-jingjing-share-*")))

    def test_exact_staged_bytes_are_rescanned_before_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            (root / "notes" / "public.md").write_text("initially public", encoding="utf-8")
            destination = base / "changed-during-copy"
            leaked = _fake_provider_key("staged-copy")

            def replace_during_copy(_source: Path, staged: Path) -> tuple[int, str]:
                staged.parent.mkdir(parents=True, exist_ok=True)
                payload = f"api_key={leaked}".encode()
                staged.write_bytes(payload)
                return len(payload), hashlib.sha256(payload).hexdigest()

            with patch.object(privacy_module, "_copy_file_and_hash", side_effect=replace_during_copy):
                with self.assertRaises(PrivacyViolationError) as raised:
                    create_share_copy(
                        root,
                        destination,
                        options=ShareCopyOptions(include_notes=True),
                    )

            self.assertEqual(raised.exception.report.status, "blocked")
            self.assertFalse(destination.exists())
            self.assertFalse(list(base.glob(".ai-jingjing-share-*")))
            self.assertNotIn(
                leaked,
                json.dumps(raised.exception.report.to_dict(), ensure_ascii=False),
            )

    def test_verify_rejects_tampering_manifest_traversal_and_unlisted_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._product_root(base)
            (root / "notes" / "public.md").write_text("public", encoding="utf-8")
            destination = base / "share"
            create_share_copy(
                root,
                destination,
                options=ShareCopyOptions(include_notes=True),
            )
            (destination / "notes" / "public.md").write_text("tampered", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_share_copy(destination)

            malicious = base / "malicious-manifest"
            malicious.mkdir()
            (malicious / "share_manifest.json").write_text(
                json.dumps(
                    {
                        "format": privacy_module.SHARE_FORMAT,
                        "file_count": 1,
                        "total_bytes": 1,
                        "files": [{"path": "../escape", "size": 1, "sha256": "a" * 64}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "路径不安全"):
                verify_share_copy(malicious)

            if hasattr(os, "symlink"):
                clean = base / "clean-share"
                create_share_copy(root, clean)
                outside = base / "outside"
                outside.write_text("outside", encoding="utf-8")
                try:
                    (clean / "unlisted-link").symlink_to(outside)
                except OSError:
                    return
                with self.assertRaisesRegex(RuntimeError, "符号链接"):
                    verify_share_copy(clean)


if __name__ == "__main__":
    unittest.main()
