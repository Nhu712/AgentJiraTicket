import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Annotated, Dict, List, Optional, TypedDict

import requests
from botbuilder.core import (
    ActivityHandler,
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import Activity, ActivityTypes
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from starlette.middleware import Middleware as _StarletteMiddleware
from starlette.requests import Request as _StarletteRequest
from starlette.responses import JSONResponse as _StarletteJSONResponse
from starlette.responses import Response as _StarletteResponse

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
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "PCPOP")
JIRA_BROWSE_URL = os.environ.get("JIRA_BROWSE_URL", "https://vothihuynhnhu2310.atlassian.net").rstrip("/")

_is_cloud = "atlassian.com" in JIRA_BASE_URL

_headers = {
    "Authorization": f"{JIRA_AUTH_TYPE} {JIRA_API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}
if not _is_cloud:
    _headers["Host"] = JIRA_HOST

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _fetch_project_schema_for_key(project_key: str) -> str:
    """Call createmeta to discover required/optional fields per issue type for a given project."""
    try:
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/2/issue/createmeta",
            params={
                "projectKeys": project_key,
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
        return f"(No project found for key {project_key!r})"

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


log.info("Starting Jira agent | model=%s base_url=%s jira=%s", LLM_MODEL, LLM_BASE_URL, JIRA_BASE_URL)

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

_SAM1_SCHEMA = """\
Issue type: Epic
  Required:
    - Summary (key=summary)
    - Reporter (key=reporter)
  Optional (common):
    - Parent (key=parent)
    - Description (key=description)

Issue type: Subtask
  Required:
    - Summary (key=summary)
    - Parent (key=parent)
    - Reporter (key=reporter)
  Optional (common):
    - Description (key=description)

Issue type: Task
  Required:
    - Summary (key=summary)
    - Reporter (key=reporter)
  Optional (common):
    - Parent (key=parent)
    - Description (key=description)

Issue type: Story
  Required:
    - Summary (key=summary)
    - Product Domain (key=customfield_10042) [options: Customer Experience, Payment, User]
    - Reporter (key=reporter)
    - Sub Domain (key=customfield_10043) [options: Help Center, Internal Tools, Config Tools]
  Optional (common):
    - Parent (key=parent)
    - Description (key=description)

Issue type: Bug
  Required:
    - Summary (key=summary)
    - Reporter (key=reporter)
  Optional (common):
    - Parent (key=parent)
    - Description (key=description)
    - Product Domain (key=customfield_10042) [options: Customer Experience, Payment, User]
"""

SYSTEM_PROMPT = f"""
## Role
You are a Jira project management assistant connected to {JIRA_BASE_URL}.
You help users create and manage Jira tickets for project **{JIRA_PROJECT_KEY}**.

## Objective
Help users go from a raw message or description → a confirmed, correctly
structured Jira ticket — with zero manual field-hunting and no silent guessing.

## Skills
- Extract ticket fields (summary, description, type, priority, assignee, labels,
  components, etc.) from free-form text
- Map extracted values to valid field options from the schema below
- Identify missing required fields and ask for them clearly
- Confirm all data with the user before writing anything to Jira

## Project Schema ({JIRA_PROJECT_KEY})
Always use project_key = **{JIRA_PROJECT_KEY}**. Do NOT call any tool to list or check projects.
Use the schema below as the source of truth for required/optional fields.

```
{_SAM1_SCHEMA}
```

## Steps

### Step 1 — Extract & Map
Parse the user's message and extract values for every field present.
Map each extracted value to the closest valid option in the schema.
Do NOT invent or guess values for fields that are unclear — flag them.

### Step 2 — Gather Bug Details (Bug type only)
If the issue type is Bug AND any of the following are missing, ask them ALL
in ONE message before proceeding:
- **Reproduce steps**: exact steps to reproduce the bug
- **Browser / device**: browser name+version, OS, device type
- **Frequency & environment**: how often / which environment?
- **Error message or log**: any console error or stack trace

Never ask one-by-one — always batch into a single message.
Do NOT proceed to Step 3 until these are answered.

### Step 3 — Resolve Missing Required Fields
For each required field that is missing or ambiguous, ask the user ONE
consolidated follow-up (group all missing fields in a single message).

### Step 4 — Confirm Before Creating
Before calling create_jira_ticket, present a structured summary:

    📋 *Ticket Summary — please confirm:*
    - Project        : <KEY>
    - Type           : Bug
    - Summary        : Login fails on SSO
    - Priority       : High
    - Description    : <first 100 chars>...
    ➡ Reply *"confirm"* to create, or tell me what to change.

Only call create_jira_ticket after the user explicitly confirms.

### Step 5 — Create & Return Link
Call create_jira_ticket with the confirmed payload.
Return the ticket key and direct URL to the user.

## Handling Tool Errors
When any tool raises an error, DO NOT forward the raw error text to the user.

1. **Required field missing**: Ask the user for that value. Retry once answered.
2. **Technical/system error** (timeout, auth, 5xx): Tell the user briefly:
   "Có lỗi kỹ thuật khi tạo ticket, bạn thử lại sau nhé."

NEVER show raw field IDs (customfield_XXXXX), HTTP status codes, or stack traces.

## Constraints
- NEVER call create_jira_ticket without explicit user confirmation in Step 4
- NEVER guess or default required fields silently — always ask
- Ask all missing fields in ONE message, not one-by-one
- Keep all messages concise and professional — no filler text

## Output Format
After successful creation, always reply with exactly:

    ✅ Ticket created: [{{TICKET_KEY}}]({JIRA_BROWSE_URL}/browse/{{TICKET_KEY}}) — {{TICKET_SUMMARY}}

## Language Rule
Detect the language of the user's message and reply in the same language
(English → English, Vietnamese → Vietnamese).
STRICT: NEVER output Chinese characters under any circumstances.
"""


@tool
def list_jira_projects() -> list:
    """List all Jira projects the user has access to. Call this when the user has not specified a project."""
    try:
        resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/2/project",
            headers=_headers,
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()
        return [
            {"key": p["key"], "name": p["name"], "type": p.get("projectTypeKey", "")}
            for p in resp.json()
        ]
    except Exception as exc:
        log.warning("[jira] list_projects failed: %s", exc)
        return []


@tool
def get_project_schema(project_key: str) -> str:
    """
    Fetch the live field definitions for a Jira project.
    Call this at the start of each ticket creation flow with the chosen project_key.
    Returns required and optional fields per issue type.
    """
    if not project_key:
        return "Error: project_key is required. Ask the user which project to use."
    return _fetch_project_schema_for_key(project_key)


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
    project_key: str = "",
    description: str = "",
    priority: str = "",
    custom_fields: str = "",
) -> dict:
    """
    Create a Jira ticket. Only call when all required fields are confirmed by the user.

    project_key: the Jira project key (e.g. 'SCRUM'). Uses env default if not provided.
    custom_fields: JSON string of extra field key-value pairs,
    e.g. '{"customfield_10200": "value"}'.
    """
    key = project_key or JIRA_PROJECT_KEY
    if not key:
        raise ValueError("project_key is required. Ask the user which project to use.")

    parsed_cf: dict = {}
    if custom_fields:
        try:
            result = json.loads(custom_fields.strip())
            if isinstance(result, dict):
                parsed_cf = result
        except (json.JSONDecodeError, AttributeError):
            pass

    # Jira API v2 requires select/option fields as {"value": "..."} not plain strings.
    # Convert string values for known select fields so the LLM doesn't need to know the format.
    _SELECT_FIELDS = {"customfield_10042", "customfield_10043"}
    for _sf in _SELECT_FIELDS:
        if _sf in parsed_cf and isinstance(parsed_cf[_sf], str):
            parsed_cf[_sf] = {"value": parsed_cf[_sf]}

    fields: dict = {
        "project": {"key": key},
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

        # If Product Domain or Sub Domain is among the errors, ask user in Vietnamese.
        # Return as a string (not raise) so LangGraph passes it to the LLM as a tool result.
        DOMAIN_KEYS = {"customfield_10042", "customfield_10043", "Product Domain", "Sub Domain"}
        if set(errors.keys()) & DOMAIN_KEYS:
            opts_domain = _get_custom_field_options_raw("customfield_10042")
            opts_subdomain = _get_custom_field_options_raw("customfield_10043")
            domain_list = "\n".join(f"  • {o['value']} (id: {o['id']})" for o in opts_domain) if opts_domain else "  (không lấy được danh sách)"
            subdomain_list = "\n".join(f"  • {o['value']} (id: {o['id']})" for o in opts_subdomain) if opts_subdomain else "  (không lấy được danh sách)"
            return (
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

# ---------------------------------------------------------------------------
# Microsoft Teams Bot Integration
# ---------------------------------------------------------------------------

_TEAMS_APP_ID = os.environ.get("MicrosoftAppId", "")
_TEAMS_APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
_TEAMS_TENANT_ID = os.environ.get("MicrosoftAppTenantId", "")

_teams_histories: Dict[str, List[dict]] = {}
_MAX_HISTORY_TURNS = 20

_bot_adapter = BotFrameworkAdapter(
    BotFrameworkAdapterSettings(
        _TEAMS_APP_ID,
        _TEAMS_APP_PASSWORD,
        channel_auth_tenant=_TEAMS_TENANT_ID or None,
    )
)


def _build_teams_message(history: List[dict]) -> str:
    if len(history) <= 1:
        return history[-1]["content"]
    prior = history[:-1]
    lines = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in prior
    )
    return f"[Previous conversation]\n{lines}\n[End of history]\n\nUser: {history[-1]['content']}"


def _invoke_agent_sync(message: str) -> str:
    try:
        result = graph.invoke({"messages": [("user", message)]})
        msgs = result.get("messages", [])
        if not msgs:
            return "Sorry, the agent returned no response."
        return msgs[-1].content
    except Exception as exc:
        log.error("Teams agent error: %s", exc, exc_info=True)
        return f"Sorry, something went wrong: {str(exc)}"


async def _process_and_reply_async(
    conv_ref,
    message: str,
    history: List[dict],
    conv_id: str,
) -> None:
    """Run agent in background and send reply proactively so Teams gets 201 quickly."""
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, _invoke_agent_sync, message)

    history.append({"role": "assistant", "content": reply})
    max_entries = _MAX_HISTORY_TURNS * 2
    if len(history) > max_entries:
        _teams_histories[conv_id] = history[-max_entries:]

    async def _send_reply(ctx: TurnContext):
        await ctx.send_activity(reply)

    try:
        await _bot_adapter.continue_conversation(conv_ref, _send_reply, _TEAMS_APP_ID)
    except Exception as exc:
        log.error("Proactive reply failed (conv=%s): %s", conv_id, exc, exc_info=True)


async def _on_adapter_error(context: TurnContext, error: Exception):
    log.error("Bot adapter turn error: %s", error, exc_info=True)

_bot_adapter.on_turn_error = _on_adapter_error


class _TeamsBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        # Use official Bot Framework helper to strip @bot mention (handles all Teams formats)
        text = TurnContext.remove_recipient_mention(turn_context.activity)
        if not text:
            text = (turn_context.activity.text or "").strip()
        else:
            text = text.strip()
        if not text:
            return

        conv_id = turn_context.activity.conversation.id
        channel = turn_context.activity.channel_id or "unknown"
        log.info("Teams message received (channel=%s conv=%s): %s", channel, conv_id[:16], text[:100])

        history = _teams_histories.setdefault(conv_id, [])
        history.append({"role": "user", "content": text})

        # Send typing indicator before yielding (outbound to Teams service URL)
        await turn_context.send_activity(Activity(type=ActivityTypes.typing))

        # Save conversation reference before on_turn returns
        conv_ref = TurnContext.get_conversation_reference(turn_context.activity)
        message = _build_teams_message(history)

        # Background the LLM call — Teams requires 201 within ~5 s; don't block here
        asyncio.create_task(
            _process_and_reply_async(conv_ref, message, history, conv_id)
        )


_teams_bot = _TeamsBot()


async def _teams_messages_endpoint(request: _StarletteRequest) -> _StarletteResponse:
    if "application/json" not in request.headers.get("content-type", ""):
        return _StarletteResponse(status_code=415)
    try:
        body = await request.json()
        activity = Activity().deserialize(body)
        auth_header = request.headers.get("Authorization", "")
        log.info(
            "Teams /api/messages: type=%s channel=%s serviceUrl=%s has_auth=%s",
            activity.type,
            activity.channel_id,
            (activity.service_url or "")[:60],
            bool(auth_header),
        )
        invoke_response = await _bot_adapter.process_activity(
            activity, auth_header, _teams_bot.on_turn
        )
        if invoke_response:
            return _StarletteJSONResponse(
                content=invoke_response.body, status_code=invoke_response.status
            )
        return _StarletteResponse(status_code=201)
    except PermissionError as exc:
        log.error("Teams JWT auth rejected (401): %s", exc)
        return _StarletteResponse(status_code=401)
    except Exception as exc:
        log.error("Teams messages endpoint error: %s", exc, exc_info=True)
        return _StarletteResponse(status_code=500)


# ---------------------------------------------------------------------------
# ASGI middleware: intercepts /api/messages before Starlette router sees it.
# More reliable than router.routes.append() which can miss in some Starlette
# versions when routes are compiled at init time.

_CHAT_HTML_PATH = os.path.join(os.path.dirname(__file__), "chat.html")


async def _chat_ui_endpoint(_request: _StarletteRequest) -> _StarletteResponse:
    try:
        with open(_CHAT_HTML_PATH, "rb") as f:
            content = f.read()
        return _StarletteResponse(
            content=content,
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
        )
    except FileNotFoundError:
        return _StarletteResponse(content=b"chat.html not found", status_code=404)


async def _chat_api_endpoint(request: _StarletteRequest) -> _StarletteResponse:
    if "application/json" not in request.headers.get("content-type", ""):
        return _StarletteJSONResponse(
            content={"status": "error", "response": "Content-Type must be application/json"},
            status_code=415,
        )
    try:
        body = await request.json()
    except Exception:
        return _StarletteJSONResponse(
            content={"status": "error", "response": "Invalid JSON"}, status_code=400
        )
    message = (body.get("message") or "").strip()
    if not message:
        return _StarletteJSONResponse(
            content={"status": "error", "response": "Missing 'message'"}, status_code=400
        )
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, _invoke_agent_sync, message)
    return _StarletteJSONResponse(
        content={"status": "success", "response": reply, "timestamp": datetime.now().isoformat()}
    )


class _TeamsASGIMiddleware:
    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")
        request = _StarletteRequest(scope, receive, send)

        if path == "/api/messages" and method == "POST":
            response = await _teams_messages_endpoint(request)
        elif path in ("/", "/chat") and method == "GET":
            response = await _chat_ui_endpoint(request)
        elif path == "/api/chat" and method == "POST":
            response = await _chat_api_endpoint(request)
        else:
            # Forward to inner app; preserve X-Accel-Buffering for streaming
            async def _send_with_accel(message):
                if message.get("type") == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-accel-buffering", b"no"))
                    message = {**message, "headers": headers}
                await send(message)
            await self._app(scope, receive, _send_with_accel)
            return

        await response(scope, receive, send)


_teams_middleware = (
    [_StarletteMiddleware(_TeamsASGIMiddleware)]
    if (_TEAMS_APP_ID and _TEAMS_APP_PASSWORD)
    else None
)

app = GreenNodeAgentBaseApp(middleware=_teams_middleware)

if _TEAMS_APP_ID and _TEAMS_APP_PASSWORD:
    log.info("Teams bot /api/messages ready via ASGI middleware (app_id=%s...)", _TEAMS_APP_ID[:8])
else:
    log.warning("MicrosoftAppId/Password not set — Teams /api/messages disabled")


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
