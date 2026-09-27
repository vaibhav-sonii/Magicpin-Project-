"""
magicpin AI Challenge - simple Vera-style merchant bot.

Idea: ONE prompt. We hand the LLM the 4 contexts (category, merchant, trigger,
customer) as JSON plus a short rulebook, and ask for a WhatsApp message back as
JSON. A few cheap checks run after the LLM (empty? more than one CTA? repeat?).
If a check fails we retry once; if the LLM is down we use a plain template.

Run locally:  uvicorn bot:app --host 0.0.0.0 --port 8080
"""
import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL = "claude-sonnet-4-6"

try:
    import anthropic
    _client = anthropic.Anthropic() if os.getenv("ANTHROPIC_API_KEY") else None
except ImportError:
    _client = None


# ----------------------------------------------------------------------------
# 1. The prompt
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp assistant for local merchants in India
(dentists, salons, restaurants, gyms, pharmacies). You write ONE short WhatsApp
message using only the data you are given.

Rules:
1. Why now: the first line must make clear what happened (the trigger).
2. Be specific: use at least one real number, date, price or source from the data
   (e.g. "2,410 views", "Haircut @ Rs 99", "JIDA Oct 2026"). Never "10% off" style offers
   when a service+price offer exists.
3. Never invent anything: no numbers, offers, studies, competitor names or slots
   that are not in the data. If something is missing, just leave it out.
4. Voice: follow category.voice (tone, allowed words). Never use its taboo words.
   Dentists/pharmacies = calm, clinical, peer-like. Salons/gyms/restaurants = warm, practical.
   No hype, no ALL CAPS, no long greetings, no "I hope you are doing well".
5. Language: if the merchant's languages include "hi" (or the customer's language_pref
   mentions hi), write natural Hindi-English mix in Roman script. Otherwise English.
6. Exactly ONE call to action, as the last sentence. Prefer a simple yes/no
   ("Reply YES and I'll draft it"). For a pure information update, no CTA is fine.
   For customer booking reminders you may offer the given slots as 1 / 2.
7. Keep it under ~70 words. Do not repeat any message from conversation_history.
8. If a customer is given, you are writing ON BEHALF OF THE MERCHANT to that customer:
   use the customer's first name, sign as the business, no internal metrics.

Return ONLY a JSON object, no markdown:
{"body": "<message>", "cta": "binary_yes_no" | "open_ended" | "slot_choice" | "none",
 "rationale": "<one sentence: which fact you used and why now>"}"""

REPLY_PROMPT = """You are Vera, magicpin's WhatsApp assistant, in the middle of a chat with a
merchant (or with a merchant's customer). Write the next reply.

Rules:
- If they said yes / go ahead / let's do it: stop pitching. Confirm and say the exact
  next step you are doing now. Do NOT ask another qualifying question.
- If they asked a question: answer it briefly using only the data given; if you don't
  know, say so honestly and offer what you can do.
- If it is off-topic (e.g. GST filing, personal stuff) or rude: stay polite, say it's
  outside what you help with, and bring it back to their magicpin/Google profile in one line.
- Match the language they just used (English, Hindi or Hinglish).
- No re-introduction, max ~50 words, at most one question/CTA, never repeat an earlier message.

Return ONLY JSON: {"body": "<reply>", "cta": "binary_yes_no" | "open_ended" | "none",
"rationale": "<one sentence>"}"""


def _pick_digest_item(category: dict, trigger: dict) -> dict | None:
    """If the trigger points at a digest item, pull that one item out (keeps prompt small)."""
    item_id = (trigger.get("payload") or {}).get("top_item_id")
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def build_prompt(category: dict, merchant: dict, trigger: dict, customer: dict | None,
                 previous_bodies: list[str]) -> str:
    cat = {k: category.get(k) for k in
           ("slug", "voice", "offer_catalog", "peer_stats", "seasonal_beats", "trend_signals")}
    item = _pick_digest_item(category, trigger)
    if item:
        cat["digest_item_for_this_trigger"] = item
    data = {
        "category": cat,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
        "messages_already_sent_do_not_repeat": previous_bodies[-5:],
    }
    return "Here is the data:\n" + json.dumps(data, ensure_ascii=False, indent=1) + \
           "\n\nWrite the message now."


# ----------------------------------------------------------------------------
# 2. LLM call + checks
# ----------------------------------------------------------------------------

def call_llm(system: str, user: str) -> dict | None:
    if _client is None:
        return None
    try:
        resp = _client.messages.create(
            model=MODEL, max_tokens=600, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
            timeout=20,
        )
        text = resp.content[0].text.strip()
        match = re.search(r"\{.*\}", text, re.S)   # tolerate stray text around the JSON
        return json.loads(match.group(0)) if match else None
    except Exception as e:
        print("LLM error:", repr(e)[:200])
        return None


CTA_PATTERN = re.compile(r"\breply\b|\bjawab\b|\bbataiye\b|\blet me know\b", re.I)


def check_message(body: str, previous_bodies: list[str]) -> str | None:
    """Returns a problem description, or None if the message is fine."""
    if not body or not body.strip():
        return "the message was empty"
    if len(CTA_PATTERN.findall(body)) > 1 or body.count("?") > 2:
        return "it had more than one call to action - keep only one, at the end"
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    if any(norm(body) == norm(p) for p in previous_bodies):
        return "it repeated an earlier message word for word - say it differently"
    return None


def fallback_message(merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """Used only if the LLM is unavailable. Plain but safe (no invented facts)."""
    name = (merchant.get("identity") or {}).get("owner_first_name") or \
           (merchant.get("identity") or {}).get("name", "there")
    kind = str(trigger.get("kind", "update")).replace("_", " ")
    if customer:
        cname = (customer.get("identity") or {}).get("name", "")
        biz = (merchant.get("identity") or {}).get("name", "our clinic")
        body = f"Hi {cname}, {biz} here. Quick reminder about your {kind}. Reply YES and we'll book a slot for you."
    else:
        views = (merchant.get("performance") or {}).get("views")
        extra = f" Your profile got {views} views in the last 30 days." if views else ""
        body = f"Hi {name}, a quick {kind} update for you.{extra} Want me to prepare the next step? Reply YES."
    return {"body": body, "cta": "binary_yes_no", "rationale": f"Template fallback for {kind} trigger (LLM unavailable)."}


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None,
            previous_bodies: list[str] | None = None) -> dict:
    """The required compose() function: 4 contexts in, one message out."""
    previous = list(previous_bodies or [])
    previous += [t.get("body", "") for t in merchant.get("conversation_history", []) if t.get("from") == "vera"]

    prompt = build_prompt(category, merchant, trigger, customer, previous)
    out = call_llm(SYSTEM_PROMPT, prompt)
    problem = check_message(out.get("body", ""), previous) if out else "no answer"
    if out and problem:                                     # one retry with feedback
        out = call_llm(SYSTEM_PROMPT, prompt + f"\n\nYour last attempt was rejected because {problem}. Try again.")
        problem = check_message(out.get("body", ""), previous) if out else "no answer"
    if not out or problem:
        out = fallback_message(merchant, trigger, customer)

    return {
        "body": out["body"].strip(),
        "cta": out.get("cta", "open_ended"),
        "send_as": "merchant_on_behalf" if customer else "vera",
        "suppression_key": trigger.get("suppression_key", trigger.get("id", "")),
        "rationale": out.get("rationale", ""),
    }


# ----------------------------------------------------------------------------
# 3. Multi-turn replies (small rules first, LLM for everything else)
# ----------------------------------------------------------------------------

STOP_WORDS = re.compile(r"\b(stop|unsubscribe|not interested|don'?t message|nahi chahiye|"
                        r"band karo|mat bhejo|no thanks)\b", re.I)
LATER_WORDS = re.compile(r"\b(later|busy|baad mein|kal baat|call later)\b", re.I)
AUTO_REPLY_HINTS = re.compile(r"(thank you for contacting|thanks for reaching out|automated|"
                              r"auto[- ]?reply|will get back to you|hamari team tak|out of office)", re.I)


def respond(conv: dict, message: str) -> dict:
    """Decide the next move in an ongoing conversation."""
    turns = conv["turns"]
    their_msgs = [t["msg"].strip().lower() for t in turns if t["from"] != "bot"]
    bot_msgs = [t["msg"] for t in turns if t["from"] == "bot"]

    if STOP_WORDS.search(message):
        return {"action": "end", "rationale": "They opted out - ending politely, no more messages."}

    repeated = their_msgs.count(message.strip().lower()) >= 2
    if repeated or AUTO_REPLY_HINTS.search(message):
        if conv.get("auto_reply_nudged") or repeated:
            return {"action": "end", "rationale": "Looks like a WhatsApp auto-reply again - exiting instead of wasting turns."}
        conv["auto_reply_nudged"] = True
        return {"action": "send", "cta": "binary_yes_no",
                "body": "Lagta hai yeh auto-reply hai. Jab owner free hon, bas YES reply kar dijiye - main baaki kaam kar dungi.",
                "rationale": "Auto-reply detected; one short nudge for the owner, then stop."}

    if LATER_WORDS.search(message) and len(message) < 60:
        return {"action": "wait", "wait_seconds": 1800, "rationale": "They asked for time - backing off 30 min."}

    if len(bot_msgs) >= 5:
        return {"action": "end", "rationale": "Conversation has run long enough - closing gracefully."}

    history = "\n".join(f"{'VERA' if t['from'] == 'bot' else t['from'].upper()}: {t['msg']}" for t in turns)
    user = (f"Merchant data:\n{json.dumps(conv.get('merchant') or {}, ensure_ascii=False)[:3000]}\n\n"
            f"Conversation so far:\n{history}\n\nWrite the next reply.")
    out = call_llm(REPLY_PROMPT, user)
    if out and not check_message(out.get("body", ""), bot_msgs):
        return {"action": "send", "body": out["body"].strip(), "cta": out.get("cta", "open_ended"),
                "rationale": out.get("rationale", "")}
    return {"action": "send", "cta": "open_ended",
            "body": "Samajh gayi. Main isko note kar rahi hoon - aapko next step 10 min mein share karti hoon.",
            "rationale": "LLM unavailable; safe acknowledgement."}


# ----------------------------------------------------------------------------
# 4. HTTP API (adapted from the challenge skeleton), in-memory state
# ----------------------------------------------------------------------------

app = FastAPI(title="Vera bot (simple)")
START = time.time()
contexts: dict[tuple[str, str], dict] = {}      # (scope, id) -> {"version", "payload"}
conversations: dict[str, dict] = {}             # conversation_id -> {"turns", "merchant", ...}
sent_keys: set[str] = set()                     # suppression keys already used

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


def get_ctx(scope: str, cid: str | None) -> dict | None:
    return (contexts.get((scope, cid)) or {}).get("payload") if cid else None


@app.get("/v1/healthz")
async def healthz():
    counts = {s: 0 for s in VALID_SCOPES}
    for scope, _ in contexts:
        counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.getenv("TEAM_NAME", "Team Simple Vera"),
        "team_members": [m.strip() for m in os.getenv("TEAM_MEMBERS", "Your Name").split(",")],
        "model": MODEL,
        "approach": "single-prompt composer (4 contexts as JSON -> Claude, temp 0) + light post-checks + rule-based reply routing",
        "contact_email": os.getenv("CONTACT_EMAIL", "you@example.com"),
        "version": "1.0.0",
        "submitted_at": "2026-09-27T00:00:00Z",
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str | None = None


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope",
                                                      "details": f"scope must be one of {sorted(VALID_SCOPES)}"})
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] > body.version:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version",
                                                      "current_version": cur["version"]})
    if not (cur and cur["version"] == body.version):   # same version again = no-op
        contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat()}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _action_for(trg_id: str) -> dict | None:
    trg = get_ctx("trigger", trg_id)
    if not trg:
        return None
    key = trg.get("suppression_key") or trg_id
    if key in sent_keys:
        return None
    merchant = get_ctx("merchant", trg.get("merchant_id"))
    category = get_ctx("category", (merchant or {}).get("category_slug"))
    if not (merchant and category):
        return None
    customer = get_ctx("customer", trg.get("customer_id"))
    msg = compose(category, merchant, trg, customer)
    sent_keys.add(key)
    conv_id = f"conv_{trg['merchant_id']}_{trg_id}"
    conversations[conv_id] = {"turns": [{"from": "bot", "msg": msg["body"]}],
                              "merchant": merchant, "customer": customer}
    name = (customer or {}).get("identity", {}).get("name") or merchant["identity"].get("name", "")
    return {
        "conversation_id": conv_id,
        "merchant_id": trg["merchant_id"],
        "customer_id": trg.get("customer_id"),
        "send_as": msg["send_as"],
        "trigger_id": trg_id,
        "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
        "template_params": [name, msg["body"][:120]],
        "body": msg["body"],
        "cta": msg["cta"],
        "suppression_key": msg["suppression_key"],
        "rationale": msg["rationale"],
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    todo = body.available_triggers[:8]          # stay well inside the 30s budget
    tasks = [asyncio.to_thread(_action_for, t) for t in todo]
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=25)
    except asyncio.TimeoutError:
        results = []
    return {"actions": [r for r in results if isinstance(r, dict)]}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str = "merchant"
    message: str
    received_at: str | None = None
    turn_number: int | None = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversations.setdefault(body.conversation_id, {
        "turns": [], "merchant": get_ctx("merchant", body.merchant_id),
        "customer": get_ctx("customer", body.customer_id)})
    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation already closed."}
    conv["turns"].append({"from": body.from_role, "msg": body.message})
    try:
        result = await asyncio.wait_for(asyncio.to_thread(respond, conv, body.message), timeout=25)
    except Exception:
        result = {"action": "wait", "wait_seconds": 600, "rationale": "Took too long; backing off briefly."}
    if result["action"] == "send":
        conv["turns"].append({"from": "bot", "msg": result["body"]})
    elif result["action"] == "end":
        conv["ended"] = True
    return result


@app.post("/v1/teardown")
async def teardown():
    contexts.clear(); conversations.clear(); sent_keys.clear()
    return {"ok": True}
