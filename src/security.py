"""
security.py
-----------
MILESTONE 12: Security Hardening

Provides public functions/objects used by app.py and tests:

    validate_input(text)        → (is_safe: bool, reason: str)
    sanitize_input(text)        → cleaned string
    filter_output(text)         → (filtered_text: str, findings: list[str])
    check_api_key_safety(key)   → (ok: bool, message: str)
    rate_limiter                → RateLimiter instance  (.check, .remaining)
    get_session_id(state)       → str
    audit_log(event, detail)    → logs to security_audit.log
    check_env_exposure()        → (ok: bool, issues: list[str])
    DATA_PRIVACY_POLICY         → dict describing what data goes where
"""

import html
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 0. AUDIT LOGGING — every security event gets recorded
# ═══════════════════════════════════════════════════════════════════════════════

_log = logging.getLogger("kpi_engine.security")
_log.setLevel(logging.INFO)

# File handler — writes to project root
_log_path = Path(__file__).parent.parent / "security_audit.log"
if not _log.handlers:
    _fh = logging.FileHandler(str(_log_path))
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _log.addHandler(_fh)


def audit_log(event: str, detail: str, level: str = "INFO"):
    """
    Write a security event to security_audit.log.

    Events: INPUT_BLOCKED, OUTPUT_FILTERED, RATE_LIMITED, KEY_REJECTED,
            KEY_ACCEPTED, ENV_CHECK, RED_TEAM
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    _log.log(lvl, f"[{event}] {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INPUT VALIDATION  —  prompt injection / jailbreak detection
# ═══════════════════════════════════════════════════════════════════════════════

# High-confidence attack patterns  (block the request)
_BLOCK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Instruction override
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|context)", re.I),
     "Instruction override attempt"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|context)", re.I),
     "Instruction override attempt"),
    (re.compile(r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|context)", re.I),
     "Instruction override attempt"),

    # System prompt extraction
    (re.compile(r"(print|show|display|reveal|output|repeat|give|tell)\s+.{0,15}(your|the|system)\s+(system\s+)?(prompt|instructions|rules)", re.I),
     "System prompt extraction attempt"),
    (re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules|directives)", re.I),
     "System prompt extraction attempt"),
    (re.compile(r"(show|give|tell|display|output|reveal)\s+.{0,30}(system\s+)?prompt", re.I),
     "System prompt extraction attempt"),
    (re.compile(r"(your|the)\s+(prompt|instructions|rules)\b", re.I),
     "System prompt extraction attempt"),

    # Role hijacking
    (re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
     "Role hijacking attempt"),
    (re.compile(r"act\s+as\s+(a|an|if)\s+", re.I),
     "Role hijacking attempt"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I),
     "Role hijacking attempt"),
    (re.compile(r"from\s+now\s+on\s+you\s+(are|will|should|must)", re.I),
     "Role hijacking attempt"),

    # Data exfiltration
    (re.compile(r"(send|post|transmit|exfiltrate|leak|extract|steal|grab|dump)\s+.*(data|key|secret|password|token|credential)", re.I),
     "Data exfiltration attempt"),
    (re.compile(r"(api|openai|secret)\s*key", re.I),
     "API key reference detected"),
    (re.compile(r"(curl|wget|fetch|http|https)://", re.I),
     "URL / data exfiltration attempt"),

    # Delimiter / token injection
    (re.compile(r"```\s*(system|assistant)\s*\n", re.I),
     "Delimiter injection attempt"),
    (re.compile(r"<\|?(system|im_start|im_end|endoftext)\|?>", re.I),
     "Token injection attempt"),

    # Encoding tricks
    (re.compile(r"base64\s*(decode|encode)", re.I),
     "Encoding obfuscation attempt"),
    (re.compile(r"\\x[0-9a-fA-F]{2}"),
     "Hex escape obfuscation"),
    (re.compile(r"\\u[0-9a-fA-F]{4}"),
     "Unicode escape obfuscation"),

    # Jailbreak keywords
    (re.compile(r"DAN\s+(mode|prompt)", re.I),
     "DAN jailbreak attempt"),
    (re.compile(r"developer\s+mode", re.I),
     "Developer-mode jailbreak attempt"),
    (re.compile(r"bypass\s+(filter|safety|restriction|guardrail)", re.I),
     "Safety bypass attempt"),
    (re.compile(r"(no|without|remove)\s+(restriction|guardrail|filter|safety|limitation)s?", re.I),
     "Safety bypass attempt"),
    (re.compile(r"jailbreak", re.I),
     "Jailbreak keyword detected"),
]

MAX_INPUT_LENGTH = 5000


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Check user input for prompt injection / jailbreak patterns.

    Returns:
        (True,  "ok")                   if safe
        (False, "<human-readable reason>")  if blocked
    """
    if not text or not text.strip():
        return False, "Input is empty."

    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Input too long ({len(text):,} chars, max {MAX_INPUT_LENGTH:,})."

    for pattern, label in _BLOCK_PATTERNS:
        if pattern.search(text):
            audit_log("INPUT_BLOCKED", f"{label} | input={text[:80]}", "WARNING")
            return False, f"{label} — your query matched a known adversarial pattern."

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INPUT SANITIZATION  —  clean text before it reaches the LLM
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_input(text: str) -> str:
    """
    Light cleaning of user input before sending to the LLM.
    Strips control characters, normalizes whitespace.
    """
    if not text:
        return ""
    # Strip control chars except newline/tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse excessive whitespace
    text = re.sub(r'[ \t]{4,}', '   ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUT FILTERING  —  strip dangerous content from LLM responses
# ═══════════════════════════════════════════════════════════════════════════════

# Regex for things that should never appear in displayed output
_OUTPUT_FILTERS: List[Tuple[re.Pattern, str, str]] = [
    # Dangerous HTML tags
    (re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.I), '', "script tag removed"),
    (re.compile(r'<iframe[^>]*>.*?</iframe>', re.DOTALL | re.I), '', "iframe removed"),
    (re.compile(r'<object[^>]*>.*?</object>', re.DOTALL | re.I), '', "object tag removed"),
    (re.compile(r'<embed[^>]*/?>', re.I), '', "embed tag removed"),
    (re.compile(r'<link[^>]*/?>', re.I), '', "link tag removed"),
    # Residual tag fragments (handles nested injection like <scr<script>ipt>)
    (re.compile(r'<\s*(?:script|iframe|object|embed|link)[^>]*>?', re.I), '', "tag fragment removed"),
    (re.compile(r'</\s*(?:script|iframe|object|embed|link)\s*>', re.I), '', "tag fragment removed"),
    # Event handlers
    (re.compile(r'\bon\w+\s*=\s*["\'][^"\']*["\']', re.I), '', "event handler removed"),
    # Data URIs
    (re.compile(r'data:text/html[^"\'>\\s]*', re.I), '[removed]', "data URI removed"),
    # System prompt leak markers
    (re.compile(r'<\|?(system|im_start|im_end|endoftext)\|?>', re.I), '', "system marker removed"),
    # API key patterns (sk-proj-... or sk-...)
    (re.compile(r'sk-[a-zA-Z0-9_-]{20,}'), '[REDACTED_KEY]', "API key redacted"),
]


def filter_output(text: str) -> Tuple[str, List[str]]:
    """
    Scan LLM output and remove dangerous / sensitive content.

    Returns:
        (filtered_text, findings)
        findings is a list of human-readable labels for what was removed.
        Empty list means nothing was changed.
    """
    if not text:
        return "", []

    findings = []
    for pattern, replacement, label in _OUTPUT_FILTERS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            if label not in findings:
                findings.append(label)

    # Final pass: HTML-escape the whole thing so no raw tags survive
    text = html.escape(text, quote=False)

    if findings:
        audit_log("OUTPUT_FILTERED", f"removed: {findings}", "WARNING")

    return text, findings


# ═══════════════════════════════════════════════════════════════════════════════
# 4. API KEY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def check_api_key_safety(key: str) -> Tuple[bool, str]:
    """
    Quick format check on an OpenAI API key (does NOT call OpenAI).

    Returns:
        (True,  "ok")         if format looks valid
        (False, "<reason>")   if something is wrong
    """
    if not key or not key.strip():
        return False, "Key is empty."

    key = key.strip()

    if not key.startswith("sk-"):
        audit_log("KEY_REJECTED", "bad prefix", "WARNING")
        return False, "Key should start with 'sk-'."

    if len(key) < 20:
        audit_log("KEY_REJECTED", "too short", "WARNING")
        return False, "Key is too short."

    if re.search(r'\s', key):
        audit_log("KEY_REJECTED", "contains whitespace", "WARNING")
        return False, "Key contains whitespace."

    audit_log("KEY_ACCEPTED", f"key={key[:7]}...{key[-4:]}")
    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Sliding-window rate limiter.  Per-session, in-memory.

    Usage:
        ok, msg = rate_limiter.check(session_id)
        n       = rate_limiter.remaining(session_id)
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: Dict[str, List[float]] = {}

    def _clean(self, session_id: str) -> List[float]:
        now = time.time()
        cutoff = now - self.window_seconds
        window = self._windows.get(session_id, [])
        window = [t for t in window if t > cutoff]
        self._windows[session_id] = window
        return window

    def check(self, session_id: str) -> Tuple[bool, str]:
        """Return (allowed, message)."""
        window = self._clean(session_id)
        if len(window) >= self.max_requests:
            wait = int(self.window_seconds - (time.time() - window[0])) + 1
            audit_log("RATE_LIMITED", f"session={session_id}, limit={self.max_requests}/min", "WARNING")
            return False, f"Rate limit reached ({self.max_requests}/min). Try again in ~{wait}s."
        self._windows.setdefault(session_id, []).append(time.time())
        return True, "ok"

    def remaining(self, session_id: str) -> int:
        """How many requests are left in the current window."""
        window = self._clean(session_id)
        return max(0, self.max_requests - len(window))


# Global instance — 10 analysis runs per minute
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SESSION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def get_session_id(session_state) -> str:
    """
    Get or create a stable session ID from Streamlit session_state.
    """
    if "security_session_id" not in session_state:
        session_state["security_session_id"] = uuid.uuid4().hex[:12]
    return session_state["security_session_id"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ENVIRONMENT / API KEY EXPOSURE CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_env_exposure() -> Tuple[bool, List[str]]:
    """
    Review the project for common API key management mistakes.

    Returns:
        (ok: bool, issues: list[str])
        ok=True means no issues found.
    """
    issues = []
    project_root = Path(__file__).parent.parent

    # 1. Check .env is in .gitignore
    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".env" not in content:
            issues.append(".env is NOT listed in .gitignore — keys could be committed")
    else:
        issues.append("No .gitignore file found")

    # 2. Check .env doesn't contain a real key (if it exists)
    env_file = project_root / ".env"
    if env_file.exists():
        env_content = env_file.read_text()
        for line in env_content.splitlines():
            if line.startswith("OPENAI_API_KEY=") and "sk-your-key-here" not in line:
                val = line.split("=", 1)[1].strip()
                if val.startswith("sk-") and len(val) > 30:
                    issues.append(".env contains what appears to be a real API key — rotate it if it was ever committed")

    # 3. Check security_audit.log is in .gitignore
    if gitignore.exists():
        content = gitignore.read_text()
        if "security_audit.log" not in content:
            issues.append("security_audit.log not in .gitignore — audit logs could be committed")

    ok = len(issues) == 0
    audit_log("ENV_CHECK", f"ok={ok}, issues={issues}")
    return ok, issues


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DATA PRIVACY POLICY
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PRIVACY_POLICY = {
    "what_is_sent_to_openai": [
        "User queries (after validation and sanitization)",
        "Anomaly summary text (aggregated stats, no raw PII)",
        "Chart images (base64-encoded for vision analysis)",
        "RAG context chunks (from uploaded PDFs)",
    ],
    "what_is_NOT_sent_to_openai": [
        "API keys (used only in HTTP headers, never in prompt content)",
        "Raw CSV file contents (only aggregated summaries)",
        "Other users' session data (session isolation)",
        "Security audit logs",
    ],
    "local_data_storage": [
        "Uploaded CSVs: in-memory only (Streamlit session), never written to disk",
        "Uploaded PDFs: written to OS temp directory, auto-cleaned",
        "ChromaDB vectors: temp directory per session, not shared",
        "Charts: temp directory, auto-cleaned on server restart",
        "Audit log: security_audit.log in project root (excluded from git)",
    ],
    "api_key_practices": [
        "Stored only in environment variable for session duration",
        "Format-validated before acceptance (prefix, length, whitespace)",
        "Masked in audit logs (sk-proj...xxxx)",
        "Never displayed in UI (password-masked input field)",
        "Never written to disk by the application",
        ".env file excluded from git via .gitignore",
    ],
    "recommendations_for_production": [
        "Rotate API keys regularly via OpenAI dashboard",
        "Set spending limits on OpenAI account",
        "Deploy behind authentication (OAuth, SSO)",
        "Use a secrets manager (AWS Secrets Manager, Vault) instead of .env",
        "Add Redis-backed rate limiting for multi-process deployments",
        "Enable OpenAI data retention opt-out if handling sensitive data",
    ],
}
