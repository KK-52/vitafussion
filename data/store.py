"""
VitaFusion AI — User & Booking persistence (flat JSON files).
"""

import json, pathlib
from datetime import datetime

DATA_DIR    = pathlib.Path(__file__).parent
USERS_FILE  = DATA_DIR / "users.json"
BOOKINGS_FILE = DATA_DIR / "bookings.json"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write(path: pathlib.Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ── User management ───────────────────────────────────────────────────────────

def save_user(user: dict) -> None:
    """Create or update a user record (keyed by phone)."""
    users = _read(USERS_FILE)
    phone = user["phone"]
    if phone in users:
        # Preserve existing history
        user["analyses"] = users[phone].get("analyses", [])
        user["registered_at"] = users[phone].get("registered_at", datetime.now().isoformat())
    else:
        user.setdefault("analyses", [])
        user["registered_at"] = datetime.now().isoformat()
    users[phone] = user
    _write(USERS_FILE, users)


def get_user(phone: str) -> dict | None:
    return _read(USERS_FILE).get(phone)


def add_analysis_to_user(phone: str, analysis: dict) -> None:
    """Append an analysis snapshot to the user's history."""
    users = _read(USERS_FILE)
    if phone in users:
        users[phone].setdefault("analyses", []).append(analysis)
        _write(USERS_FILE, users)


# ── Booking management ────────────────────────────────────────────────────────

def save_booking(booking: dict) -> None:
    bookings = _read(BOOKINGS_FILE)
    bookings[booking["token"]] = booking
    _write(BOOKINGS_FILE, bookings)


def get_booking(token: str) -> dict | None:
    return _read(BOOKINGS_FILE).get(token)


def get_user_bookings(phone: str) -> list[dict]:
    bookings = _read(BOOKINGS_FILE)
    return [b for b in bookings.values() if b.get("user_phone") == phone]
