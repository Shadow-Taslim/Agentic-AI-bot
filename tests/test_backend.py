"""
Unit tests for the Agentic AI Chatbot backend.

These tests run WITHOUT any API keys or network access:
    python -m unittest discover -s tests -v

They cover the pure/helper logic only (content normalization and
Supabase environment-variable resolution), not live LLM calls.
"""

import os
import sys
import unittest
from unittest import mock

# Make the repository root importable when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agentic_chatbot_backend as backend


class ContentToTextTests(unittest.TestCase):
    """Gemini/OpenAI chunks can be strings OR lists of content blocks."""

    def test_plain_string_passthrough(self):
        self.assertEqual(backend._content_to_text("hello"), "hello")

    def test_empty_string(self):
        self.assertEqual(backend._content_to_text(""), "")

    def test_list_of_text_blocks(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
        self.assertEqual(backend._content_to_text(content), "Hello world")

    def test_list_of_plain_strings(self):
        self.assertEqual(backend._content_to_text(["a", "b", "c"]), "abc")

    def test_mixed_blocks_and_non_text_entries(self):
        content = ["Hi ", {"type": "text", "text": "there"}, {"type": "image"}]
        self.assertEqual(backend._content_to_text(content), "Hi there")

    def test_empty_list(self):
        self.assertEqual(backend._content_to_text([]), "")

    def test_non_string_fallback(self):
        self.assertEqual(backend._content_to_text(42), "42")


class SupabaseEnvResolutionTests(unittest.TestCase):
    """Both SUPABASE_* and NEXT_PUBLIC_* variable names must work."""

    def _fake_create_client(self, url, key):
        return {"url": url, "key": key}

    def test_next_public_names_accepted(self):
        env = {
            "NEXT_PUBLIC_SUPABASE_URL": "https://demo.supabase.co",
            "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY": "sb_publishable_demo",
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(backend, "create_client", self._fake_create_client):
            client = backend.get_supabase_client()
        self.assertEqual(client["url"], "https://demo.supabase.co")
        self.assertEqual(client["key"], "sb_publishable_demo")

    def test_standard_names_accepted(self):
        env = {
            "SUPABASE_URL": "https://demo.supabase.co",
            "SUPABASE_KEY": "service-role-demo",
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(backend, "create_client", self._fake_create_client):
            client = backend.get_supabase_client()
        self.assertEqual(client["key"], "service-role-demo")

    def test_no_credentials_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(backend.get_supabase_client())


class LazyLlmInitTests(unittest.TestCase):
    """Importing without keys must not crash; chatting without keys must."""

    def test_missing_key_raises_on_chat_not_on_import(self):
        with mock.patch.object(backend, "llm", None):
            with self.assertRaises(ValueError):
                backend.chat_node({"messages": []})


if __name__ == "__main__":
    unittest.main()
