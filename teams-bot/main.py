"""Microsoft Teams bot that proxies messages to the GreenNode agent endpoint."""

import os
from typing import Dict, List

import aiohttp
from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
    ActivityHandler,
)
from botbuilder.schema import Activity, ActivityTypes
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.environ["MicrosoftAppId"]
APP_PASSWORD = os.environ["MicrosoftAppPassword"]
APP_TENANT_ID = os.environ.get("MicrosoftAppTenantId", "")
GREENNODE_ENDPOINT = os.environ["GREENNODE_ENDPOINT"]
PORT = int(os.environ.get("PORT", 3978))
MAX_HISTORY_TURNS = 20  # pairs of user/assistant

SETTINGS = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD, channel_auth_tenant=APP_TENANT_ID or None)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# In-memory history per Teams conversation ID: [{role, content}]
_histories: Dict[str, List[dict]] = {}


async def _on_error(context: TurnContext, error: Exception):
    print(f"[ERROR] {error}", flush=True)
    await context.send_activity("Sorry, something went wrong. Please try again.")


ADAPTER.on_turn_error = _on_error


class TeamsBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        text = (turn_context.activity.text or "").strip()

        # Strip @mention text added by Teams
        if turn_context.activity.entities:
            for entity in turn_context.activity.entities:
                if entity.type == "mention":
                    mention_text = entity.additional_properties.get("text", "")
                    if mention_text:
                        text = text.replace(mention_text, "").strip()

        if not text:
            return

        conv_id = turn_context.activity.conversation.id
        history = _histories.setdefault(conv_id, [])
        history.append({"role": "user", "content": text})

        await turn_context.send_activity(Activity(type=ActivityTypes.typing))

        payload = _build_payload(history)
        reply = await _call_greennode(payload)

        history.append({"role": "assistant", "content": reply})

        # Trim old turns
        max_entries = MAX_HISTORY_TURNS * 2
        if len(history) > max_entries:
            _histories[conv_id] = history[-max_entries:]

        await turn_context.send_activity(reply)


def _build_payload(history: List[dict]) -> str:
    """Build context-aware message string matching the chat.html format."""
    if len(history) <= 1:
        return history[-1]["content"]

    prior = history[:-1]
    lines = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in prior
    )
    current = history[-1]["content"]
    return f"[Previous conversation]\n{lines}\n[End of history]\n\nUser: {current}"


async def _call_greennode(message: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GREENNODE_ENDPOINT,
                json={"message": message},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json(content_type=None)
                if data.get("status") == "success":
                    return data.get("response") or "(empty response)"
                return f"Agent error: {data.get('response') or data.get('error') or 'unknown'}"
    except Exception as exc:
        return f"Failed to reach the agent: {exc}"


BOT = TeamsBot()


async def messages(req: web.Request) -> web.Response:
    if "application/json" not in req.content_type:
        return web.Response(status=415, text="Expected application/json")

    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    invoke_response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
    if invoke_response:
        return web.json_response(data=invoke_response.body, status=invoke_response.status)
    return web.Response(status=201)


async def health(req: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/health", health)

if __name__ == "__main__":
    print(f"Teams bot starting on port {PORT}", flush=True)
    web.run_app(app, host="0.0.0.0", port=PORT)
