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
# User-facing browse URL (may differ from API base URL for Atlassian Cloud)
JIRA_BROWSE_URL = os.environ.get("JIRA_BROWSE_URL", "https://vothihuynhnhu2310.atlassian.net").rstrip("/")

_is_cloud = "atlassian.com" in JIRA_BASE_URL
_headers = {
    "Authorization": f"{JIRA_AUTH_TYPE} {JIRA_API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}
# Host header only needed for on-prem Jira (IP-based virtual host routing)
if not _is_cloud:
    _headers["Host"] = JIRA_HOST

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

### Step 2b — Gather Bug Details (Bug type only)
If the issue type is Bug (or likely a Bug) AND any of the following are missing
from the conversation, ask them ALL in ONE message before proceeding:
- **Reproduce steps**: exact steps to reproduce the bug
- **Browser / device**: browser name+version, OS, device type
- **Frequency & environment**: how often does it happen (% or always/sometimes)?
  Which environment (staging / production / both)?
- **Error message or log**: any console error, stack trace, or log snippet

Only skip this step if all four points are already clearly answered in the
conversation. Never ask them one-by-one across multiple messages — always batch
into a single message. Do NOT proceed to Step 3 until these are answered.

### Step 3 — Resolve Missing Required Fields
Compare extracted fields against the schema's required fields.
For each required field that is missing or ambiguous (EXCEPT customfield_10042
and customfield_10043 — handle those automatically in Step 3a), ask the user ONE
consolidated follow-up (group all missing fields into a single message,
never ask one field per message).

### Step 3a — Auto-resolve Product Domain & Sub Domain
customfield_10042 (Product Domain) and customfield_10043 (Sub Domain) are ALWAYS
required. You MUST set them automatically — do NOT ask the user.

Before creating the ticket:
1. Call get_jira_custom_field_options("customfield_10042") and
   get_jira_custom_field_options("customfield_10043") to get valid option IDs.
2. Based on the ticket content and conversation context, pick the most appropriate
   option for each field (use semantic reasoning — e.g. a finance internal tool maps
   to Internal Tools; a customer-facing feature maps to Customer Experience).
3. Include both fields in the create_jira_ticket call as custom_fields JSON:
   '{{"customfield_10042": {{"id": "<chosen_id>"}}, "customfield_10043": {{"id": "<chosen_id>"}}}}'
4. If truly ambiguous and you cannot determine a reasonable value even after
   reading the options, ask the user with this exact format (do NOT expose raw
   field IDs or Jira error text):

   "Mình cần bạn xác nhận thêm 1-2 thông tin để tạo ticket:

   **Product Domain** — lĩnh vực sản phẩm liên quan:
   • [list each option name from get_jira_custom_field_options]

   **Sub Domain** — mảng cụ thể hơn trong domain đó:
   • [list each option name from get_jira_custom_field_options]

   Bạn chọn phương án nào phù hợp nhất nhé?"

   After the user answers, map their answer to the correct option ID and proceed.

### Step 4 — Confirm Before Creating
Before calling create_jira_ticket, present a structured summary of all
field values and ask the user to confirm or correct:

    📋 *Ticket Summary — please confirm:*
    - Type          : Bug
    - Summary       : Login fails on SSO with Chrome 124
    - Priority      : High
    - Assignee      : @nguyen.van.a
    - Component     : Authentication
    - Product Domain: Customer Experience
    - Sub Domain    : Internal Tools
    - Description   : <first 100 chars>...
    ➡ Reply *"confirm"* to create, or tell me what to change.

Only call create_jira_ticket after the user explicitly confirms.

### Step 5 — Create & Return Link
Call create_jira_ticket with the confirmed payload.
Return the ticket key and direct URL to the user.

## Handling Tool Errors
When any tool raises an error, DO NOT forward the raw error text to the user.
Instead, evaluate the error and decide the best action:

1. **Error contains "Mình cần bạn xác nhận"** (Product Domain / Sub Domain missing):
   Show the error text verbatim to the user — it already contains the question
   in Vietnamese with available options. Wait for the user's answer, then map
   their choice to the option id and retry create_jira_ticket.

2. **Error mentions a required field the user can provide** (e.g. missing summary,
   priority, component): Ask the user for that value in Vietnamese. Retry once
   the user answers.

3. **Error is a technical/system error the user cannot fix** (connection timeout,
   auth failure, server 5xx): Tell the user briefly in Vietnamese that there was
   a system error and suggest retrying, e.g.:
   "Có lỗi kỹ thuật khi tạo ticket, bạn thử lại sau nhé."

NEVER show raw field IDs (customfield_XXXXX), HTTP status codes, or stack traces
to the user.

## Constraints
- NEVER call create_jira_ticket without explicit user confirmation in Step 4
- NEVER ask the user about customfield_10042 or customfield_10043 — always infer them
- NEVER guess or default other required fields silently — always ask
- NEVER present options not returned by get_project_schema
- Ask all missing fields in ONE message, not one-by-one
- Keep all messages concise and professional — no filler text
- get_project_schema must be called once at the start of each ticket
  creation flow to ensure schema is current

## Output Format
After successful creation, always reply with exactly:

    ✅ Ticket created: [{{TICKET_KEY}}]({JIRA_BROWSE_URL}/browse/{{TICKET_KEY}}) — {{TICKET_SUMMARY}}

## Language Rule
Detect the language of the user's message and reply in the same language
throughout the entire conversation (English → English, Vietnamese → Vietnamese).
STRICT: NEVER output Chinese characters (中文/漢字) under any circumstances,
even partially. If you are about to write a Chinese word, replace it with the
equivalent Vietnamese or English word instead.
"""


def _get_custom_field_options_raw(field_id: str) -> list:
    """Internal helper — same logic as get_jira_custom_field_options but callable from within tools."""
    try:
        ctx_resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/field/{field_id}/context",
            headers=_headers, timeout=10, verify=False,
        )
        if ctx_resp.ok:
            contexts = ctx_resp.json().get("values", [])
            if contexts:
                ctx_id = contexts[0]["id"]
                opt_resp = requests.get(
                    f"{JIRA_BASE_URL}/rest/api/3/field/{field_id}/context/{ctx_id}/option",
                    headers=_headers, timeout=10, verify=False,
                )
                if opt_resp.ok:
                    return [{"id": v["id"], "value": v["value"]} for v in opt_resp.json().get("values", [])]
    except Exception:
        pass
    try:
        meta = requests.get(
            f"{JIRA_BASE_URL}/rest/api/2/issue/createmeta",
            params={"projectKeys": JIRA_PROJECT_KEY, "expand": "projects.issuetypes.fields"},
            headers=_headers, timeout=15, verify=False,
        )
        if meta.ok:
            projects = meta.json().get("projects", [])
            if projects:
                for itype in projects[0].get("issuetypes", []):
                    field = itype.get("fields", {}).get(field_id, {})
                    allowed = field.get("allowedValues", [])
                    if allowed:
                        return [{"id": v.get("id", ""), "value": v.get("value", v.get("name", ""))} for v in allowed]
    except Exception:
        pass
    return []


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

        # Auto-strip fields that "cannot be set" on this screen and retry once
        unset_fields = [
            k for k, v in errors.items()
            if "cannot be set" in str(v) or "not on the appropriate screen" in str(v)
        ]
        if unset_fields:
            log.warning("[jira] Auto-stripping unsettable fields and retrying: %s", unset_fields)
            for f in unset_fields:
                payload["fields"].pop(f, None)
            retry = requests.post(
                f"{JIRA_BASE_URL}/rest/api/2/issue",
                json=payload,
                headers=_headers,
                timeout=15,
                verify=False,
            )
            if retry.ok:
                body = retry.json()
                key = body["key"]
                return {"key": key, "id": body["id"], "url": f"{JIRA_BROWSE_URL}/browse/{key}"}
            errors = retry.json().get("errors", {})
            msgs = retry.json().get("errorMessages", [])

        # If Product Domain or Sub Domain is among the errors, ask user in Vietnamese
        DOMAIN_KEYS = {"customfield_10042", "customfield_10043", "Product Domain", "Sub Domain"}
        if set(errors.keys()) & DOMAIN_KEYS:
            opts_domain = _get_custom_field_options_raw("customfield_10042")
            opts_subdomain = _get_custom_field_options_raw("customfield_10043")
            domain_list = "\n".join(f"  • {o['value']} (id: {o['id']})" for o in opts_domain) if opts_domain else "  (không lấy được danh sách)"
            subdomain_list = "\n".join(f"  • {o['value']} (id: {o['id']})" for o in opts_subdomain) if opts_subdomain else "  (không lấy được danh sách)"
            raise ValueError(
                "Mình cần bạn xác nhận thêm 2 thông tin để tạo ticket:\n\n"
                f"Product Domain — lĩnh vực sản phẩm liên quan:\n{domain_list}\n\n"
                f"Sub Domain — mảng cụ thể hơn trong domain đó:\n{subdomain_list}\n\n"
                "Bạn chọn phương án nào phù hợp nhất nhé?"
            )

        detail = "; ".join([f"{k}: {v}" for k, v in errors.items()] + msgs)
        log.warning("[jira] Validation error (HTTP %d): %s", resp.status_code, detail)
        raise ValueError(f"Lỗi tạo ticket: {detail}")

    key = body["key"]
    return {
        "key": key,
        "id": body["id"],
        "url": f"{JIRA_BROWSE_URL}/browse/{key}",
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


@tool
def get_jira_custom_field_options(field_id: str) -> list:
    """
    Fetch allowed options for a custom select/dropdown field.
    Call this when create_jira_ticket returns a validation error about a required
    custom field — pass the field id (e.g. 'customfield_10042') to get valid options.
    Returns a list of {id, value} dicts, or [] if the options cannot be retrieved.
    """
    try:
        # Try field context options (API v3)
        ctx_resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/field/{field_id}/context",
            headers=_headers,
            timeout=10,
            verify=False,
        )
        if ctx_resp.ok:
            contexts = ctx_resp.json().get("values", [])
            if contexts:
                ctx_id = contexts[0]["id"]
                opt_resp = requests.get(
                    f"{JIRA_BASE_URL}/rest/api/3/field/{field_id}/context/{ctx_id}/option",
                    headers=_headers,
                    timeout=10,
                    verify=False,
                )
                if opt_resp.ok:
                    return [{"id": v["id"], "value": v["value"]} for v in opt_resp.json().get("values", [])]
    except Exception as exc:
        log.warning("[jira] get_field_options(%s) failed: %s", field_id, exc)

    # Fallback: extract from createmeta
    try:
        meta = requests.get(
            f"{JIRA_BASE_URL}/rest/api/2/issue/createmeta",
            params={"projectKeys": JIRA_PROJECT_KEY, "expand": "projects.issuetypes.fields"},
            headers=_headers,
            timeout=15,
            verify=False,
        )
        if meta.ok:
            projects = meta.json().get("projects", [])
            if projects:
                for itype in projects[0].get("issuetypes", []):
                    field = itype.get("fields", {}).get(field_id, {})
                    allowed = field.get("allowedValues", [])
                    if allowed:
                        return [{"id": v.get("id", ""), "value": v.get("value", v.get("name", ""))} for v in allowed]
    except Exception as exc:
        log.warning("[jira] get_field_options createmeta fallback failed: %s", exc)

    return []


_tools = [create_jira_ticket, get_jira_ticket, search_jira_tickets, get_jira_custom_field_options]

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
