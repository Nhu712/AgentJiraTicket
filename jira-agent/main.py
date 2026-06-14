import json
import logging
import os
from datetime import datetime
from typing import Annotated, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from greennode_agentbase import GreenNodeAgentBaseApp, PingStatus, RequestContext

load_dotenv()

class _SingleLineFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return msg.replace("\n", " ↵ ").replace("\r", "")

_handler = logging.StreamHandler()
_handler.setFormatter(_SingleLineFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger(__name__)

LLM_MODEL = os.environ["LLM_MODEL"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
# Corp Jira (legacy internal, kept for reference)
CORP_JIRA_BASE_URL = os.environ.get("CORP_JIRA_BASE_URL", "https://10.30.94.60").rstrip("/")
CORP_JIRA_HOST = os.environ.get("CORP_JIRA_HOST", "jira.zalopay.vn")
CORP_JIRA_API_TOKEN = os.environ.get("CORP_JIRA_API_TOKEN", "MTY2MTQwNzY4MDE4OrmpBoNIz1iKoCCM63xJNA/THfTK")
CORP_JIRA_PROJECT_KEY = os.environ.get("CORP_JIRA_PROJECT_KEY", "PCPOP")

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://10.30.94.60").rstrip("/")
JIRA_HOST = os.environ.get("JIRA_HOST", "jira.zalopay.vn")
JIRA_AUTH_TYPE = os.environ.get("JIRA_AUTH_TYPE", "Basic")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "MTY2MTQwNzY4MDE4OrmpBoNIz1iKoCCM63xJNA/THfTK")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "vothihuynhnhu2310@gmail.com")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "PCPOP")

def _build_headers() -> dict:
    import base64
    is_cloud = "atlassian.com" in JIRA_BASE_URL
    if is_cloud and JIRA_EMAIL:
        # Atlassian Cloud: Basic auth requires base64(email:api_token)
        token = base64.b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode()
        auth = f"Basic {token}"
    else:
        auth = f"{JIRA_AUTH_TYPE} {JIRA_API_TOKEN}"

    headers = {
        "Authorization": auth,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Host header only needed for on-prem Jira (IP-based with virtual host)
    if not is_cloud:
        headers["Host"] = JIRA_HOST
    return headers

_headers = _build_headers()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _fetch_project_schema() -> str:
    """Call createmeta to discover required/optional fields per issue type."""
    try:
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/2/issue/createmeta",
            params={
                "projectKeys": JIRA_PROJECT_KEY,
                "expand": "projects.issuetypes.fields",
            },
            headers=_headers,
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()
        projects = resp.json().get("projects", [])
    except Exception as exc:
        return f"(Schema fetch failed: {exc})"

    if not projects:
        return f"(No project found for key {JIRA_PROJECT_KEY!r})"

    lines = []
    for itype in projects[0].get("issuetypes", []):
        required, optional = [], []
        for key, field in itype.get("fields", {}).items():
            if key in ("issuetype", "project"):
                continue
            name = field.get("name", key)
            allowed = field.get("allowedValues", [])
            suffix = ""
            if allowed:
                vals = [v.get("name", v.get("value", "")) for v in allowed[:8]]
                suffix = f" [options: {', '.join(vals)}]"
            entry = f"    - {name} (key={key}){suffix}"
            (required if field.get("required") else optional).append(entry)

        lines.append(f"Issue type: {itype['name']}")
        if required:
            lines.append("  Required:")
            lines.extend(required)
        if optional:
            lines.append("  Optional (common):")
            lines.extend(optional[:5])
        lines.append("")

    return "\n".join(lines)


log.info("Starting Jira agent | model=%s base_url=%s project=%s", LLM_MODEL, LLM_BASE_URL, JIRA_PROJECT_KEY)

# Fetched once at startup; project schema rarely changes between requests.
_PROJECT_SCHEMA = _fetch_project_schema()
log.info("Project schema loaded (%d chars)", len(_PROJECT_SCHEMA))

# SYSTEM_PROMPT = f"""You are a Jira project management assistant for project {JIRA_PROJECT_KEY}.
# Instance: {JIRA_BASE_URL}

# === Project field schema ===
# {_PROJECT_SCHEMA}
# ===========================

# Workflow for creating a ticket:
# 1. Parse the user message and extract values for every REQUIRED field in the schema above.
# 2. If any required field cannot be determined, ask the user for it. Do not guess.
# 3. Only call create_jira_ticket once ALL required fields are confirmed.
# 4. After creation, reply with the ticket key and direct URL.

# You can also use get_jira_ticket to look up a ticket, or search_jira_tickets with JQL.
# Be concise and professional.
# """

SYSTEM_PROMPT = f"""
## Role
You are a Jira project management assistant, expert at extracting structured
information from unstructured messages and creating well-formed Jira tickets.
You operate on project {JIRA_PROJECT_KEY} at {JIRA_BASE_URL}.

## Objective
Help users go from a raw message or description → a confirmed, correctly
structured Jira ticket — with zero manual field-hunting and no silent guessing.

## Skills
- Extract ticket fields (summary, description, type, priority, assignee, labels,
  components, etc.) from free-form text
- Map extracted values to valid field options from the live project schema
- Identify missing required fields and ask for them clearly
- Confirm all data with the user before writing anything to Jira

## Steps

### Step 1 — Load Schema
Call get_project_schema to fetch the live field definitions for project
{JIRA_PROJECT_KEY}. Use the returned schema as the source of truth for:
- Which fields are required vs optional
- Valid options for every dropdown / select field (issue type, priority,
  components, labels, assignee, custom fields, etc.)

### Step 2 — Extract & Map
Parse the user's message and extract values for every field present.
Map each extracted value to the closest valid option in the schema.
Do NOT invent or guess values for fields that are unclear — flag them.

### Step 3 — Resolve Missing Required Fields
Compare extracted fields against the schema's required fields.
For each required field that is missing or ambiguous, ask the user ONE
consolidated follow-up (group all missing fields into a single message,
never ask one field per message).

### Step 4 — Confirm Before Creating
Before calling create_jira_ticket, present a structured summary of all
field values and ask the user to confirm or correct:

    📋 *Ticket Summary — please confirm:*
    - Type       : Bug
    - Summary    : Login fails on SSO with Chrome 124
    - Priority   : High
    - Assignee   : @nguyen.van.a
    - Component  : Authentication
    - Description: <first 100 chars>...
    ➡ Reply *"confirm"* to create, or tell me what to change.

Only call create_jira_ticket after the user explicitly confirms.

### Step 5 — Create & Return Link
Call create_jira_ticket with the confirmed payload.
Return the ticket key and direct URL to the user.

## Constraints
- NEVER call create_jira_ticket without explicit user confirmation in Step 4
- NEVER guess or default a required field silently — always ask
- NEVER present options not returned by get_project_schema
- Ask all missing fields in ONE message, not one-by-one
- Keep all messages concise and professional — no filler text
- get_project_schema must be called once at the start of each ticket
  creation flow to ensure schema is current

## Output Format
After successful creation, always reply with exactly:

    ✅ Ticket created: {JIRA_BASE_URL}/browse/{{TICKET_KEY}}

## Language Rule
Detect the language of the user's message and reply in the same language
throughout the entire conversation (English → English, Vietnamese → Vietnamese).
"""


@tool
def create_jira_ticket(
    summary: str,
    issue_type: str,
    description: str = "",
    priority: str = "",
    custom_fields: str = "",
) -> dict:
    """
    Create a Jira ticket. Only call when all required fields are confirmed by the user.

    custom_fields: JSON string of extra field key-value pairs,
    e.g. '{"customfield_10200": "value"}'.
    """
    parsed_cf: dict = {}
    if custom_fields:
        try:
            result = json.loads(custom_fields.strip())
            if isinstance(result, dict):
                parsed_cf = result
        except (json.JSONDecodeError, AttributeError):
            pass

    fields: dict = {
        "project": {"key": JIRA_PROJECT_KEY},
        "summary": summary,
        "issuetype": {"name": issue_type},
    }
    if description:
        fields["description"] = description
    if priority:
        fields["priority"] = {"name": priority}
    if parsed_cf:
        fields.update(parsed_cf)
    log.info("Start to create ticket: %s", json.dumps(fields))

    payload = {"fields": fields}
    safe_headers = {k: (v[:20] + "...") if k == "Authorization" else v for k, v in _headers.items()}
    log.info("[jira] POST %s/rest/api/2/issue", JIRA_BASE_URL)
    log.info("[jira] headers=%s", json.dumps(safe_headers))
    log.info("[jira] body=%s", json.dumps(payload))
    try:
        resp = requests.post(
            f"{JIRA_BASE_URL}/rest/api/2/issue",
            json=payload,
            headers=_headers,
            timeout=15,
            verify=False,
        )
    except Exception as exc:
        log.error("Jira API error: %s", exc, exc_info=True)
        raise ValueError(f"Failed to connect to Jira API: {str(exc)}")

    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        log.error("Jira non-JSON (status=%d): %r", resp.status_code, resp.text[:300])
        raise ValueError(f"Jira returned non-JSON (HTTP {resp.status_code}). Response: {resp.text[:200]}")

    if not resp.ok:
        errors = body.get("errors", {})
        msgs = body.get("errorMessages", [])
        detail = "; ".join([f"{k}: {v}" for k, v in errors.items()] + msgs)
        log.warning("[jira] Validation error (HTTP %d): %s", resp.status_code, detail)
        raise ValueError(f"Jira validation error — {detail}. Please provide the missing fields.")

    return {
        "key": body["key"],
        "id": body["id"],
        "url": f"{JIRA_BASE_URL}/browse/{body['key']}",
    }


@tool
def get_jira_ticket(ticket_key: str) -> dict:
    """Get details of a Jira issue by its key, e.g. PROJ-123."""
    resp = requests.get(
        f"{JIRA_BASE_URL}/rest/api/2/issue/{ticket_key}",
        headers=_headers,
        timeout=15,
    )
    resp.raise_for_status()
    f = resp.json()["fields"]
    return {
        "key": ticket_key,
        "summary": f.get("summary"),
        "status": f["status"]["name"],
        "priority": f["priority"]["name"] if f.get("priority") else None,
        "issue_type": f["issuetype"]["name"],
        "assignee": f["assignee"]["displayName"] if f.get("assignee") else None,
        "url": f"{JIRA_BASE_URL}/browse/{ticket_key}",
    }


@tool
def search_jira_tickets(jql: str, max_results: int = 10) -> list:
    """Search Jira issues with JQL. Example: 'project=PROJ AND status=Open ORDER BY created DESC'."""
    try:
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/2/issue/search",
            params={
                "jql": jql,
                "maxResults": max_results,
                "fields": "summary,status,priority,issuetype",
            },
            headers=_headers,
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()
        return [
            {
                "key": i["key"],
                "summary": i["fields"]["summary"],
                "status": i["fields"]["status"]["name"],
                "url": f"{JIRA_BASE_URL}/browse/{i['key']}",
            }
            for i in resp.json().get("issues", [])
        ]
    except Exception as exc:
        log.warning("[jira] search failed (jql=%r): %s", jql, exc)
        return []


_tools = [create_jira_ticket, get_jira_ticket, search_jira_tickets]

llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
llm_with_tools = llm.bind_tools(_tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _chatbot(state: State) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


_g = StateGraph(State)
_g.add_node("chatbot", _chatbot)
_g.add_node("tools", ToolNode(_tools))
_g.add_edge(START, "chatbot")
_g.add_conditional_edges("chatbot", tools_condition)
_g.add_edge("tools", "chatbot")
_g.add_edge("chatbot", END)
graph = _g.compile()

app = GreenNodeAgentBaseApp()


@app.entrypoint
def handler(payload: dict, context: RequestContext) -> dict:
    message = payload.get("message", "")
    if not message:
        return {"status": "error", "response": "Missing 'message' in payload"}

    log.info("→ User: %s", message)
    try:
        result = graph.invoke({"messages": [("user", message)]})
        messages = result.get("messages", [])
        if not messages:
            log.warning("Agent returned no messages")
            return {"status": "error", "response": "Agent returned no messages"}

        reply = messages[-1].content
        log.info("← Agent: %s", reply[:200] + ("..." if len(reply) > 200 else ""))

        tool_calls = [
            m for m in messages
            if hasattr(m, "tool_calls") and m.tool_calls
        ]
        for m in tool_calls:
            for tc in m.tool_calls:
                log.info("  [tool] %s(%s)", tc["name"], ", ".join(f"{k}={v!r}" for k, v in tc["args"].items()))

        return {
            "status": "success",
            "response": reply,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        log.error("Handler error: %s", exc, exc_info=True)
        return {
            "status": "error",
            "response": f"Agent error: {str(exc)}",
        }


@app.ping
def health_check() -> PingStatus:
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
