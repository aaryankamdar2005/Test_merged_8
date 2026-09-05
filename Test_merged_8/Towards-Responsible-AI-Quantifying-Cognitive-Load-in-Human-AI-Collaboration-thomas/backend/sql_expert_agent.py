"""Strict, schema-aware SQL Expert Agent helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from sqlglot import exp, parse
from sqlglot.errors import ParseError

MAX_ROWS = 200
STATEMENT_TIMEOUT_MS = 5000
PUBLIC_TABLES = {
    "participants", "sessions", "answers", "chat_logs", "eye_tracking",
    "facial_expression", "keyboard_tracking", "mouse_tracking",
    "tracking_data", "participant_monitoring", "nasa_tlx", "question_ratings",
    "session_cli_summary", "rag_chunks", "rag_prompt_evaluations", "rag_pipeline",
    "final_dataset",
}
SENSITIVE_COLUMNS = {
    "password", "password_hash", "password_digest", "token", "access_token",
    "participant_session_token", "secret", "api_key", "answer", "message",
    "prompt", "prompt_text", "question_text", "chat_history", "full_name",
}
BLOCKED_FUNCTIONS = {"pg_sleep", "pg_terminate_backend", "pg_cancel_backend", "current_setting", "set_config", "dblink_connect", "dblink_exec", "lo_import", "lo_export"}


class SqlAgentError(ValueError):
    pass


def schema_description(storage: Any) -> str:
    parts: list[str] = []
    for table in sorted(PUBLIC_TABLES):
        columns = storage.table_columns(table)
        if columns:
            parts.append(f"{table}({', '.join(columns)})")
    return "\n".join(parts) or "No application tables are currently available."


def _schema_maps(schema: str) -> tuple[set[str], dict[str, set[str]]]:
    tables: set[str] = set()
    columns: dict[str, set[str]] = {}
    for line in schema.splitlines():
        if "(" not in line or not line.endswith(")"):
            continue
        table, raw_columns = line.split("(", 1)
        table = table.strip().lower()
        tables.add(table)
        columns[table] = {item.strip().lower() for item in raw_columns[:-1].split(",") if item.strip()}
    return tables, columns


def _normalise_sql(sql: str) -> str:
    query = sql.strip()
    fenced = re.fullmatch(r"```(?:sql)?\s*(.*?)```", query, re.IGNORECASE | re.DOTALL)
    if fenced:
        query = fenced.group(1).strip()
    if query.lower().startswith("sql\n"):
        query = query.split("\n", 1)[1].strip()
    return query.rstrip(";").strip()


def validate_readonly_sql(sql: str, available_tables: set[str] | None = None, available_columns: dict[str, set[str]] | None = None) -> str:
    query = _normalise_sql(sql)
    if not query:
        raise SqlAgentError("The SQL Expert Agent returned an empty query.")
    if ";" in query or "--" in query or "/*" in query or "*/" in query:
        raise SqlAgentError("Only one comment-free SQL statement is allowed.")
    try:
        statements = parse(query, read="postgres")
    except ParseError as exc:
        raise SqlAgentError("The SQL Expert Agent returned invalid SQL syntax.") from exc
    if len(statements) != 1:
        raise SqlAgentError("Only one SQL statement is allowed.")
    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.With)):
        raise SqlAgentError("Only read-only SELECT or WITH queries are allowed.")
    allowed_tables = {item.lower() for item in (available_tables or PUBLIC_TABLES)}
    column_map = {table.lower(): {column.lower() for column in values} for table, values in (available_columns or {}).items()}
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    for table_node in statement.find_all(exp.Table):
        table_name = (table_node.name or "").lower()
        catalog = (table_node.catalog or "").lower()
        db = (table_node.db or "").lower()
        if catalog or (db and db != "public") or (table_name not in allowed_tables and table_name not in cte_names):
            raise SqlAgentError(f"Access to table '{table_name}' is not allowed.")
    for node in statement.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Transaction)):
            raise SqlAgentError("The query contains a prohibited SQL operation.")
        if isinstance(node, exp.Anonymous) and node.name.lower() in BLOCKED_FUNCTIONS:
            raise SqlAgentError(f"Function '{node.name}' is not allowed.")
        if isinstance(node, exp.Column):
            column_name = node.name.lower()
            if column_name in SENSITIVE_COLUMNS:
                raise SqlAgentError(f"Sensitive column '{column_name}' is not available to the SQL agent.")
            if column_name == "*" or not column_map or (node.table or "").lower() in cte_names:
                continue
            qualifiers = [(node.table or "").lower()] if node.table else list(column_map)
            if not any(column_name in column_map.get(table, set()) for table in qualifiers):
                raise SqlAgentError(f"Column '{column_name}' is not available in the approved schema.")
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            raise SqlAgentError("Wildcard column selection is not allowed; name approved columns explicitly.")
    limit = statement.args.get("limit")
    if limit is None:
        query = f"{query} LIMIT {MAX_ROWS}"
    else:
        limit_expression = limit.expression
        if not isinstance(limit_expression, exp.Literal) or not limit_expression.is_int:
            raise SqlAgentError("LIMIT must be a positive integer.")
        limit_expression.replace(exp.Literal.number(min(max(int(limit_expression.this), 1), MAX_ROWS)))
        query = statement.sql(dialect="postgres")
    return query


def parse_agent_response(raw: str) -> tuple[str, str]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SqlAgentError("The SQL Expert Agent returned an invalid response.") from exc
    if not isinstance(value, dict) or not isinstance(value.get("sql"), str):
        raise SqlAgentError("The SQL Expert Agent response did not contain SQL.")
    return value["sql"], str(value.get("explanation") or "Read-only query generated by the SQL Expert Agent.")[:2000]


def run_sql_expert_agent(question: str, storage: Any, llm: Callable[[list[dict[str, str]]], tuple[str, str, str]], provider: str, role: str) -> dict[str, Any]:
    if role not in {"admin", "host", "dashboard"}:
        raise SqlAgentError("The SQL Expert Agent is available only to dashboard roles.")
    schema = schema_description(storage)
    tables, columns = _schema_maps(schema)
    prompt = ("You are CogniTrack's SQL Expert Agent. Convert the user's analytics question into one "
        "read-only PostgreSQL query over the approved schema. Never use answer text, passwords, tokens, "
        "authentication data, or sensitive columns. Return JSON only with keys sql and explanation. "
        f"Use only SELECT/WITH and a numeric LIMIT no greater than {MAX_ROWS}.\n\nApproved schema:\n{schema}\n\nUser question: {question}")
    raw, model_provider, model = llm([{"role": "system", "content": prompt}])
    proposed_sql, explanation = parse_agent_response(raw)
    sql = validate_readonly_sql(proposed_sql, tables or PUBLIC_TABLES, columns)
    rows = storage.execute_readonly(sql, statement_timeout_ms=STATEMENT_TIMEOUT_MS)
    return {"success": True, "agent": "sql_expert_agent", "provider": model_provider, "model": model, "question": question, "sql": sql, "explanation": explanation, "row_count": len(rows), "rows": rows}
