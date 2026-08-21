import json
import logging
from typing import Any, Dict, Optional

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

logger = logging.getLogger('9router.settings_store')

TABLE = 'router9_runtime_settings'
ROW_KEY = 'default'


class SettingsStore:
    """Persists admin-edited overrides (API keys, aliases, provider
    on/off) across restarts. Same table-per-blob shape as
    ChatGPTProvider's credential store; separate table so the two never
    contend for rows. Encrypted at rest when an encryption key is set,
    plaintext otherwise (still requires DATABASE_URL, which on Render is
    not publicly reachable).
    """

    def __init__(self, database_url: str, encryption_key: str = ''):
        self.database_url = (database_url or '').strip()
        self.encryption_key = (encryption_key or '').strip()
        self._ready = False
        if self.database_url:
            self._init_table()

    def _fernet(self):
        if not self.encryption_key or Fernet is None:
            return None
        return Fernet(self.encryption_key.encode())

    def _init_table(self) -> None:
        try:
            import psycopg
            with psycopg.connect(self.database_url, connect_timeout=10) as conn:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE} (
                        settings_key TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                conn.commit()
            self._ready = True
        except Exception as exc:
            logger.error('Could not initialize %s: %s', TABLE, exc)

    @property
    def enabled(self) -> bool:
        return self._ready

    def load(self) -> Dict[str, Any]:
        if not self._ready:
            return {}
        try:
            import psycopg
            with psycopg.connect(self.database_url, connect_timeout=10) as conn:
                row = conn.execute(
                    f'SELECT payload FROM {TABLE} WHERE settings_key = %s', (ROW_KEY,)
                ).fetchone()
            if not row:
                return {}
            raw = str(row[0])
            fernet = self._fernet()
            raw = fernet.decrypt(raw.encode()).decode() if fernet else raw
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning('Could not load runtime settings overrides: %s', exc)
            return {}

    def save(self, overrides: Dict[str, Any]) -> None:
        if not self._ready:
            raise RuntimeError('DATABASE_URL is not configured; admin changes cannot be persisted.')
        blob = json.dumps(overrides, separators=(',', ':'))
        fernet = self._fernet()
        payload = fernet.encrypt(blob.encode()).decode() if fernet else blob
        import psycopg
        with psycopg.connect(self.database_url, connect_timeout=10) as conn:
            conn.execute(f"""
                INSERT INTO {TABLE} (settings_key, payload, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (settings_key)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
            """, (ROW_KEY, payload))
            conn.commit()

    def clear(self) -> None:
        self.save({})
