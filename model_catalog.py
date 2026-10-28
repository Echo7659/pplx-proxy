"""Authenticated website catalog synchronization with restart-safe fallback."""

import asyncio
import hashlib
import html
import json
import re
import time
import urllib.request

from official_models import derive_account_type, parse_catalog


CATALOG_URL="https://www.perplexity.ai/rest/models/config/v2"
SESSION_URL="https://www.perplexity.ai/api/auth/session"


def cookie_owner(token):
    return hashlib.sha256(token.encode()).hexdigest() if token else ""


def fetch_browser_catalog(cookies, solver_url):
    """Use the existing deployment's browser reader when REST is challenged."""
    token=cookies.get("__Secure-next-auth.session-token")
    if not token:
        raise ValueError("No authenticated session for model synchronization")
    payload={
        "cmd": "request.get", "url": CATALOG_URL, "maxTimeout": 20000,
        "cookies": [{"name": "__Secure-next-auth.session-token", "value": token,
                     "domain": ".perplexity.ai", "path": "/", "secure": True, "httpOnly": True}],
    }
    request=urllib.request.Request(solver_url+"/v1", data=json.dumps(payload).encode(),
                                   headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=25) as response:
        result=json.loads(response.read(2_000_001))
    solution=result.get("solution", {})
    if result.get("status") != "ok" or solution.get("status") != 200:
        raise ValueError("Browser reader did not return an authenticated model catalog")
    body=solution.get("response", "")
    if not isinstance(body, str) or len(body) > 1_000_000:
        raise ValueError("Invalid website model catalog body")
    pre=re.search(r"<pre[^>]*>(.*?)</pre>", body, re.DOTALL)
    return json.loads(html.unescape(pre.group(1) if pre else body))


class ModelCatalog:
    def __init__(self, path, owner, writer):
        self.path=path
        self.owner=owner
        self.writer=writer
        self.catalog=None
        self.account_type=None
        self.updated_at=None
        self.last_error=None
        self.lock=asyncio.Lock()
        self.last_attempt=0
        self.last_attempt_owner=None
        try:
            snapshot=json.loads(path.read_text())
            if not isinstance(snapshot, dict):
                raise ValueError("Catalog cache must be an object")
            if snapshot.get("version") == 1 and owner and snapshot.get("owner") == owner:
                self.catalog=parse_catalog(snapshot["config"], snapshot["account_type"])
                self.account_type=snapshot["account_type"]
                self.updated_at=snapshot["updated_at"]
        except FileNotFoundError:
            pass
        except (ValueError, KeyError, TypeError, OSError):
            self.last_error="Invalid catalog cache; waiting for website synchronization"

    def status(self):
        state="pending"
        if self.last_error:
            state="error"
        elif self.catalog:
            state="synced"
        return {
            "source": CATALOG_URL, "status": state,
            "updated_at": self.updated_at, "account_type": self.account_type,
            "model_count": len(self.catalog["models"]) if self.catalog else 0,
            "last_error": self.last_error,
        }

    def aliases(self):
        if not self.catalog:
            return {}
        aliases={}
        families={"gpt": r"^GPT", "gemini": r"^Gemini", "gemini-flash": r"^Gemini.*Flash",
                  "sonnet": r"Sonnet", "opus": r"Opus", "grok": r"^Grok", "nemotron": r"^Nemotron"}
        for model_id, entry in self.catalog["models"].items():
            if model_id == "auto":
                continue
            slug=re.sub(r"[^a-z0-9.-]+", "-", entry["label"].lower()).strip("-")
            if slug and slug not in self.catalog["models"]:
                aliases.setdefault(slug, model_id)
            for alias, pattern in families.items():
                if alias not in self.catalog["models"] and re.search(pattern, entry["label"], re.IGNORECASE):
                    aliases.setdefault(alias, model_id)
        return aliases

    def metadata(self, model_id):
        if not self.catalog:
            return None
        canonical=self.aliases().get(model_id, model_id)
        return self.catalog["models"].get(canonical)

    def model_map(self):
        if not self.catalog:
            return None
        result={model_id: (entry["mode"], entry["pref"]) for model_id, entry in self.catalog["models"].items()}
        for alias, canonical in self.aliases().items():
            result.setdefault(alias, result[canonical])
        return result

    async def refresh(self, client, solver_url):
        requested_at=time.time()
        requested_owner=self.owner
        async with self.lock:
            if self.last_attempt_owner == requested_owner and self.last_attempt >= requested_at:
                return self.status()
            try:
                if self.owner != requested_owner:
                    raise ValueError("Account changed before model synchronization")
                await client.init()
                response=await client.session.get(SESSION_URL, timeout=20)
                if response.status_code != 200:
                    raise ValueError(f"Account lookup returned HTTP {response.status_code}")
                user=response.json().get("user")
                if not user:
                    raise ValueError("Website session is not authenticated")
                account_type=derive_account_type(user)
                response=await client.session.get(CATALOG_URL, timeout=20)
                if response.status_code == 403:
                    payload=await asyncio.to_thread(fetch_browser_catalog, client.cookies, solver_url)
                elif response.status_code == 200:
                    payload=response.json()
                else:
                    raise ValueError(f"Website model catalog returned HTTP {response.status_code}")
                catalog=parse_catalog(payload, account_type)
                updated_at=int(time.time())
                snapshot={"version": 1, "owner": requested_owner, "config": payload,
                          "account_type": account_type, "updated_at": updated_at}
                if self.owner != requested_owner:
                    raise ValueError("Account changed during model synchronization")
                self.writer(self.path, snapshot)
                self.catalog=catalog
                self.account_type=account_type
                self.updated_at=updated_at
                self.last_error=None
            except Exception as error:
                # Provider exceptions may contain request details; expose only the type.
                if self.owner == requested_owner:
                    self.last_error=f"Website model synchronization failed ({type(error).__name__}); previous catalog retained"
            finally:
                if self.owner == requested_owner:
                    self.last_attempt=time.time()
                    self.last_attempt_owner=requested_owner
            return self.status()
