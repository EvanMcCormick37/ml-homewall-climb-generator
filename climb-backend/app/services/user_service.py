from datetime import datetime, timezone
from app.database import get_db

def ensure_user_exists(user_id: str, email: str | None, name: str | None = None):
    """
    Upsert a user record on first authenticated API call.
    This avoids needing a Clerk webhook for user creation.
    """
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE id = ?", (user_id,)
        ).fetchone()

        if existing:
            # Update last-seen info
            conn.execute(
                "UPDATE users SET email = COALESCE(?, email), "
                "display_name = COALESCE(?, display_name), "
                "updated_at = ? WHERE id = ?",
                (email, name, datetime.now(timezone.utc).isoformat(), user_id),
            )
        else:
            conn.execute(
                "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
                (user_id, email or "", name),
            )