from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from storage import SqliteStorage
from sql_expert_agent import SqlAgentError, run_sql_expert_agent, validate_readonly_sql


class SqlExpertAgentTests(unittest.TestCase):
    def test_only_single_readonly_statement_is_allowed(self) -> None:
        for query in (
            "DELETE FROM participants",
            "SELECT * FROM participants; DROP TABLE participants",
            "SELECT * FROM private_secrets",
            "SELECT * FROM participants -- bypass",
            "UPDATE participants SET full_name = 'x'",
            "SELECT * FROM participants",
            "SELECT answer FROM answers",
        ):
            with self.subTest(query=query):
                with self.assertRaises(SqlAgentError):
                    validate_readonly_sql(query, {"participants"})

    def test_limit_is_added_and_capped(self) -> None:
        self.assertTrue(validate_readonly_sql("SELECT participant_id FROM participants", {"participants"}).endswith("LIMIT 200"))
        self.assertIn("LIMIT 200", validate_readonly_sql("SELECT participant_id FROM participants LIMIT 9999", {"participants"}))

    def test_agent_generates_and_executes_query_through_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = SqliteStorage(Path(directory) / "agent.sqlite3")
            storage.append("participants", ["participant_id", "full_name"], {"participant_id": "P-1", "full_name": "Ada"})

            def fake_llm(_messages):
                return ('{"sql":"SELECT participant_id FROM participants","explanation":"Lists participant IDs."}', "Test", "test-model")

            result = run_sql_expert_agent("List participant IDs", storage, fake_llm, "Groq", "admin")
            self.assertEqual(result["agent"], "sql_expert_agent")
            self.assertEqual(result["row_count"], 1)
            self.assertEqual(result["rows"][0]["participant_id"], "P-1")
            self.assertIn("LIMIT 200", result["sql"])

    def test_participant_role_cannot_use_dashboard_agent(self) -> None:
        def fake_llm(_messages):
            return ('{"sql":"SELECT 1","explanation":"No-op."}', "Test", "test-model")

        with self.assertRaises(SqlAgentError):
            run_sql_expert_agent("Show data", object(), fake_llm, "Groq", "participant")


if __name__ == "__main__":
    unittest.main()
