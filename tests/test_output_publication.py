"""Filesystem-only regression checks: no datasets, statistics, or plots regenerated."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATHS = ["src/data/build_processed_data.py", "src/analysis/model_a_return_diagnostics.py"]


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def windows_acl(path):
    environment = os.environ.copy()
    environment["OUTPUT_ACL_TEST_PATH"] = str(path)
    command = (
        "$a=Get-Acl -LiteralPath $env:OUTPUT_ACL_TEST_PATH; "
        "[pscustomobject]@{Protected=$a.AreAccessRulesProtected;"
        "Rules=@($a.Access | ForEach-Object {[pscustomobject]@{"
        "Identity=$_.IdentityReference.Value;Rights=$_.FileSystemRights.ToString();"
        "Type=$_.AccessControlType.ToString();Inherited=$_.IsInherited;"
        "ObjectInherit=(($_.InheritanceFlags -band "
        "[Security.AccessControl.InheritanceFlags]::ObjectInherit) -ne 0)}})} "
        "| ConvertTo-Json -Depth 4 -Compress"
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                            env=environment, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


class OutputPublicationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = TemporaryDirectory(prefix="publication-tests-")
        self.addCleanup(self.workspace.cleanup)
        self.output_directory = Path(self.workspace.name) / "published"
        self.output_directory.mkdir()

    def check_inheritance(self, destination):
        if os.name != "nt":
            return
        parent, actual = windows_acl(destination.parent), windows_acl(destination)
        self.assertFalse(actual["Protected"])
        self.assertTrue(all(rule["Inherited"] for rule in actual["Rules"]))
        expected = {(rule["Identity"], rule["Rights"], rule["Type"])
                    for rule in parent["Rules"] if rule["ObjectInherit"]}
        received = {(rule["Identity"], rule["Rights"], rule["Type"]) for rule in actual["Rules"]}
        self.assertEqual(expected, received)

    def test_private_staging_create_and_overwrite_inherit_destination_acl(self):
        for path in MODULE_PATHS:
            with self.subTest(module=path):
                module = load_module(path)
                destination = self.output_directory / "publication.bin"
                self.assertTrue(destination.resolve().is_relative_to(self.output_directory.resolve()))
                try:
                    with TemporaryDirectory(prefix="private-staging-", dir=self.workspace.name) as temporary:
                        source = Path(temporary) / "source.bin"
                        for payload in [b"new file\x00\xff", b"replacement\x00\xfe"]:
                            source.write_bytes(payload)
                            module.publish_output(source, destination)
                            self.assertEqual(destination.read_bytes(), payload)
                            self.check_inheritance(destination)
                            self.assertEqual(list(self.output_directory.glob(f".{destination.name}.*.tmp")), [])
                finally:
                    destination.unlink(missing_ok=True)

    def test_failed_copy_preserves_existing_destination_and_removes_temporary(self):
        for path in MODULE_PATHS:
            with self.subTest(module=path):
                module = load_module(path)
                destination = self.output_directory / "publication.bin"
                self.assertTrue(destination.resolve().is_relative_to(self.output_directory.resolve()))
                try:
                    destination.write_bytes(b"original content")
                    with TemporaryDirectory(prefix="private-staging-", dir=self.workspace.name) as temporary:
                        with self.assertRaises(FileNotFoundError):
                            module.publish_output(Path(temporary) / "absent.bin", destination)
                    self.assertEqual(destination.read_bytes(), b"original content")
                    self.assertEqual(list(self.output_directory.glob(f".{destination.name}.*.tmp")), [])
                finally:
                    destination.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
