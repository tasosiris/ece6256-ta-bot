"""Configuration precedence, model discovery, and setup against a local mock."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from endpoint_config import list_models, load_settings, resolve_settings
from mock_endpoint import MODEL_NAME, start_mock_endpoint, stop_mock_endpoint
from scripts import configure_endpoint


class EndpointConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.endpoint, cls.server = start_mock_endpoint()

    @classmethod
    def tearDownClass(cls):
        stop_mock_endpoint(cls.server)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "course_endpoint.local.json"
        clean_environment = patch.dict(os.environ, {}, clear=True)
        clean_environment.start()
        self.addCleanup(clean_environment.stop)

    def test_local_settings_are_loaded(self):
        self.path.write_text(json.dumps({"endpoint": self.endpoint + "/v1/", "model": "saved"}))
        self.assertEqual(load_settings(path=self.path), (self.endpoint + "/v1", "saved"))

    def test_cli_overrides_environment_and_local_file(self):
        self.path.write_text(json.dumps({"endpoint": self.endpoint, "model": "saved"}))
        with patch.dict(os.environ, COURSE_ENDPOINT=self.endpoint + "/env", COURSE_MODEL="env-model"):
            self.assertEqual(load_settings(path=self.path), (self.endpoint + "/env", "env-model"))
            self.assertEqual(load_settings(self.endpoint + "/cli", "cli-model", self.path),
                             (self.endpoint + "/cli", "cli-model"))

    def test_missing_endpoint_reports_required_staff_url(self):
        with self.assertRaisesRegex(ValueError, "staff-provided URL"):
            load_settings(path=self.path)

    def test_invalid_config_and_urls_fail_before_network(self):
        for content in ("{", "[]"):
            self.path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_settings(path=self.path)
        self.path.unlink()
        for endpoint in ("localhost:8000", "file:///tmp/server", "https://host/v1?key=value"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                load_settings(endpoint, path=self.path)

    def test_discovery_accepts_both_url_forms(self):
        for suffix in ("", "/v1", "/v1/"):
            with self.subTest(suffix=suffix):
                self.assertEqual(list_models(self.endpoint + suffix), [MODEL_NAME])

    def test_discovery_errors_are_clear(self):
        with patch("endpoint_config.requests.get", side_effect=requests.ConnectionError()):
            with self.assertRaisesRegex(TimeoutError, "course server"):
                list_models(self.endpoint)
        for payload in ({}, {"data": []}, {"data": [None]}, {"data": [{"id": None}]}):
            response = Mock()
            response.json.return_value = payload
            with self.subTest(payload=payload), patch("endpoint_config.requests.get", return_value=response):
                with self.assertRaises(TimeoutError):
                    list_models(self.endpoint)

    def test_single_model_is_automatically_selected(self):
        with patch("endpoint_config.load_settings", return_value=(self.endpoint, None)):
            self.assertEqual(resolve_settings(), (self.endpoint, MODEL_NAME))

    def test_multiple_models_require_explicit_selection(self):
        with patch("endpoint_config.load_settings", return_value=(self.endpoint, None)), \
                patch("endpoint_config.list_models", return_value=["first", "second"]):
            with self.assertRaisesRegex(ValueError, "multiple models"):
                resolve_settings()

    def test_explicit_model_does_not_require_discovery(self):
        with patch("endpoint_config.load_settings", return_value=(self.endpoint, "selected")), \
                patch("endpoint_config.list_models") as discover:
            self.assertEqual(resolve_settings(), (self.endpoint, "selected"))
            discover.assert_not_called()

    def test_setup_saves_only_verified_settings(self):
        with patch.object(configure_endpoint, "LOCAL_CONFIG", self.path), \
                patch("sys.argv", ["configure_endpoint.py", "--endpoint", self.endpoint]), \
                contextlib.redirect_stdout(io.StringIO()):
            configure_endpoint.main()
        self.assertEqual(json.loads(self.path.read_text()), {"endpoint": self.endpoint, "model": MODEL_NAME})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_setup_does_not_save_a_model_not_served(self):
        with patch.object(configure_endpoint, "LOCAL_CONFIG", self.path), \
                patch("sys.argv", ["configure_endpoint.py", "--endpoint", self.endpoint, "--model", "wrong"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                configure_endpoint.main()
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
