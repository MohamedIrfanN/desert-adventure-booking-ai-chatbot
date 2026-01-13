import os
import json
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

# ----------------------------
# Env
# ----------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "jetset_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TZ = os.getenv("TZ", "Asia/Dubai")
JETSET_LOCATION_REFERENCE = "Jetset Desert Camp, Dubai"

# ----------------------------
# Vector DB / Retriever
# ----------------------------
_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
_db = Chroma(
    collection_name=CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
)

def _search(query: str, k: int = 6) -> Dict[str, Any]:
    docs = _db.similarity_search(query, k=k)
    matches = [{"text": d.page_content, "meta": d.metadata} for d in docs]
    return {"query": query, "matches": matches}

# ----------------------------
# Booking Store (replace later with DB/Redis)
# ----------------------------
BOOKINGS: Dict[str, Dict[str, Any]] = {}

def _get_or_create_booking(user_id: str) -> Dict[str, Any]:
    if user_id not in BOOKINGS:
        BOOKINGS[user_id] = {
            "status": "collecting",          # collecting | ready_to_confirm | confirmed
            "customer_name": None,
            "activity": None,                # buggy | quad | safari
            "package": None,                 # safari shared/private etc (optional)
            "vehicle_model": None,           # e.g., "Buggy Polaris 4 Seats 1000cc Turbo"
            "quantity": None,                # number of vehicles (buggy/quad) OR pax/cars for safari
            "duration_min": None,            # 30/60/90/120 etc.
            "date_time_iso": None,           # ISO datetime in Dubai TZ (stored internally)
            "pickup_required": None,         # True/False
            "payment_method": None,          # cash/card
            "price_aed": None,
            "notes": []
        }
    return BOOKINGS[user_id]

def _normalize_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    return None

def _ensure_dubai_tz(dt: datetime) -> datetime:
    """Force datetime to Dubai TZ if naive or other tz is provided."""
    dubai = ZoneInfo(TZ)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=dubai)
    return dt.astimezone(dubai)

def _within_open_hours_start_end(start_dt: datetime, duration_min: int) -> bool:
    """
    Tours must START and FINISH within 9:00am–7:00pm Dubai time.
    """
    start_dt = _ensure_dubai_tz(start_dt)
    dubai = ZoneInfo(TZ)

    open_start_dt = datetime.combine(start_dt.date(), time(9, 0), tzinfo=dubai)
    open_end_dt = datetime.combine(start_dt.date(), time(19, 0), tzinfo=dubai)

    if not (open_start_dt <= start_dt <= open_end_dt):
        return False

    end_dt = start_dt + timedelta(minutes=duration_min)

    # must finish by 7pm (inclusive) on the same day
    return end_dt <= open_end_dt

# ----------------------------
# Tools
# ----------------------------
@tool
def current_datetime_tool(tz: str = "Asia/Dubai") -> str:
    """Return the current datetime (ISO string) in the given timezone. Used to resolve 'tomorrow', etc."""
    now = datetime.now(ZoneInfo(tz))
    return json.dumps({"tz": tz, "now_iso": now.isoformat()})


@tool
def retrieval_tool(query: str, k: int = 6) -> str:
    """Search the Jetset knowledge base (Chroma) for factual answers and return top matching chunks."""
    return json.dumps(_search(query=query, k=k))


@tool
def about_tool() -> str:
    """Return 'About Jetset Dubai' information from the knowledge base."""
    return json.dumps(_search("About Jetset Dubai desert tours company overview", k=6))


@tool
def location_tool() -> str:
    """Return Jetset Dubai location/contact/map details from the knowledge base."""
    return json.dumps(_search("Jetset Dubai location address contact phone map", k=6))


@tool
def packages_tool(activity: str = "all") -> str:
    """Return packages/prices/durations from the knowledge base for activity: buggy, quad, safari, or all."""
    activity = (activity or "all").lower().strip()
    if activity not in {"buggy", "quad", "safari", "all"}:
        activity = "all"
    q = f"Packages prices durations for {activity} tours Jetset Dubai"
    return json.dumps(_search(q, k=8))


@tool
def faq_tool(question: str = "") -> str:
    """Return policy/FAQ answers from the knowledge base (age limits, rules, refunds, safety, etc.)."""
    q = "Jetset Dubai FAQ policies rules"
    if question.strip():
        q = f"{q}. User question: {question}"
    return json.dumps(_search(q, k=8))


@tool
def booking_get_or_create(user_id: str) -> str:
    """Get or create a booking draft object for a given Telegram user_id."""
    draft = _get_or_create_booking(user_id)
    return json.dumps(draft)

@tool
def booking_update(
    user_id: str,
    patch: Optional[dict] = None,
    customer_name: Optional[str] = None,
    activity: Optional[str] = None,
    package: Optional[str] = None,
    vehicle_model: Optional[str] = None,
    quantity: Optional[int] = None,
    duration_min: Optional[int] = None,
    date_time_iso: Optional[str] = None,
    pickup_required: Optional[Any] = None,
    payment_method: Optional[str] = None,
    notes: Optional[Any] = None,
) -> str:
    """
    Update booking draft fields. Supports two calling styles:
    1) booking_update(user_id="..", patch={...})
    2) booking_update(user_id="..", date_time_iso="..", quantity=2, ...)
    """

    draft = _get_or_create_booking(user_id)

    merged: Dict[str, Any] = {}
    if isinstance(patch, dict):
        merged.update(patch)

    direct = {
        "customer_name": customer_name,
        "activity": activity,
        "package": package,
        "vehicle_model": vehicle_model,
        "quantity": quantity,
        "duration_min": duration_min,
        "date_time_iso": date_time_iso,
        "pickup_required": pickup_required,
        "payment_method": payment_method,
        "notes": notes,
    }
    for k, v in direct.items():
        if v is not None:
            merged[k] = v

    # normalize
    if "pickup_required" in merged:
        merged["pickup_required"] = _normalize_bool(merged["pickup_required"])

    if "activity" in merged and merged["activity"]:
        merged["activity"] = str(merged["activity"]).strip().lower()

    if "payment_method" in merged and merged["payment_method"]:
        merged["payment_method"] = str(merged["payment_method"]).strip().lower()

    for key in ["quantity", "duration_min"]:
        if key in merged and merged[key] is not None:
            try:
                merged[key] = int(merged[key])
            except Exception:
                pass

    # apply
    for k, v in merged.items():
        if k in draft:
            draft[k] = v
        elif k == "notes":
            if isinstance(v, list):
                draft["notes"].extend(v)
            else:
                draft["notes"].append(str(v))

    # readiness (NO seats requirement)
    required = ["customer_name", "activity", "quantity", "duration_min", "date_time_iso", "payment_method", "pickup_required"]
    is_ready = all(draft.get(r) not in [None, ""] for r in required)

    draft["status"] = "ready_to_confirm" if is_ready else "collecting"
    if any(k in merged for k in ["quantity", "duration_min", "pickup_required", "payment_method"]):
        draft["price_aed"] = None
    BOOKINGS[user_id] = draft

    return json.dumps(draft)

# ----------------------------
# Pricing rules (placeholder until we map full doc tables)
# ----------------------------
# Buggy pricing depends on vehicle_model (2-seat vs 4-seat), NOT on seats field.
BUGGY_2_SEAT = {30: 400, 60: 750, 90: 1150, 120: 1500}
BUGGY_4_SEAT = {30: 600, 60: 1150, 90: 1750, 120: 2300}

PICKUP_FEE = 350
CARD_VAT_MULTIPLIER = 1.05

def _is_buggy_4_seat(vehicle_model: Optional[str]) -> bool:
    if not vehicle_model:
        return False
    s = vehicle_model.lower()
    return "4" in s and ("seat" in s or "seater" in s)

@tool
def booking_compute_price(user_id: str) -> str:
    """
    Compute total price deterministically:
    - Buggy/Quad: price per vehicle
    - Pickup +350 AED if pickup_required
    - Card adds 5% VAT
    - Time must fit inside 9am–7pm INCLUDING duration
    """
    draft = _get_or_create_booking(user_id)

    dt_iso = draft.get("date_time_iso")
    dur = draft.get("duration_min")
    qty = draft.get("quantity")

    if dt_iso:
        try:
            dt = datetime.fromisoformat(dt_iso)
            dt = _ensure_dubai_tz(dt)
        except Exception:
            return json.dumps({"error": "I couldn't understand the time. Please tell me the time in Dubai time (e.g., 'tomorrow 5pm').", "draft": draft})
    else:
        return json.dumps({"error": "Missing booking time.", "draft": draft})

    if dur is None:
        return json.dumps({"error": "Missing duration.", "draft": draft})

    if not _within_open_hours_start_end(dt, int(dur)):
        return json.dumps({"error": "That start time plus the selected duration goes beyond our closing time (7pm). Please choose an earlier start time.", "draft": draft})

    base = 0.0

    if draft.get("activity") == "buggy":
        if not qty or not dur:
            return json.dumps({"error": "Missing quantity or duration for buggy booking.", "draft": draft})

        table = BUGGY_4_SEAT if _is_buggy_4_seat(draft.get("vehicle_model")) else BUGGY_2_SEAT

        if int(dur) not in table:
            return json.dumps({"error": "Unsupported buggy duration. Please choose 30, 60, 90, or 120 minutes.", "draft": draft})

        base = table[int(dur)] * int(qty)

    elif draft.get("activity") == "quad":
        # Fallback: pricing must be fetched from KB
        return json.dumps({
            "needs_pricing_from_kb": True,
            "message": "Quad pricing should be retrieved from the knowledge base.",
            "draft": draft
        })


    else:
        return json.dumps({"error": "Select an activity first (buggy/quad/safari).", "draft": draft})

    if draft.get("pickup_required") is True:
        base += PICKUP_FEE

    if (draft.get("payment_method") or "").lower() == "card":
        base *= CARD_VAT_MULTIPLIER

    draft["price_aed"] = round(base, 2)
    BOOKINGS[user_id] = draft

    return json.dumps({"price_aed": draft["price_aed"], "draft": draft})

@tool
def booking_confirm(user_id: str) -> str:
    """Confirm a booking if the draft is complete and a price has been computed."""
    draft = _get_or_create_booking(user_id)

    if draft.get("status") != "ready_to_confirm":
        return json.dumps({"error": "Booking is not complete yet.", "draft": draft})

    if draft.get("price_aed") is None:
        price_out = json.loads(booking_compute_price(user_id))
        if "error" in price_out:
            return json.dumps({"error": price_out["error"], "draft": draft})

    draft["status"] = "confirmed"
    BOOKINGS[user_id] = draft
    return json.dumps({
        "confirmed": True,
        "location": JETSET_LOCATION_REFERENCE,
        "draft": draft
    })

def all_tools():
    return [
        current_datetime_tool,
        retrieval_tool,
        about_tool,
        location_tool,
        packages_tool,
        faq_tool,
        booking_get_or_create,
        booking_update,
        booking_compute_price,
        booking_confirm,
    ]
