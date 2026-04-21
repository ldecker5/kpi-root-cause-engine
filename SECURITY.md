# Milestone 12: Security Hardening

## Overview

This document covers the security measures, red teaming results, and known limitations for the KPI Root Cause Analysis Engine.

---

## 1. Red Teaming Exercises

### 1.1 Test Suite

All security tests are in `tests/test_security.py`. Run them with:

```bash
python tests/test_security.py
```

### 1.2 Categories Tested (60 tests total)

| Category | Tests | Description |
|----------|-------|-------------|
| Prompt Injection Detection | 18 | Tests 13 attack patterns + 5 benign queries to verify no false positives |
| Input Validation | 5 | Empty, whitespace, oversized, valid, and injection-warning prompts |
| API Key Validation | 7 | Format checks, prefix, length, whitespace, masking |
| Output Sanitization (XSS) | 8 | Script tags, event handlers, iframes, data URIs, system markers |
| Rate Limiting | 5 | Normal usage, excess blocking, session isolation, daily limits |
| Data Leakage Prevention | 2 | API key masking verification |
| File Upload Validation | 10 | CSV/PDF/image size limits, extension checks, DataFrame validation |
| Jailbreak Scenarios | 5 | DAN mode, developer mode, nested injection, gradual escalation, indirect extraction |

### 1.3 Red Team Attack Vectors Tested

**Prompt Injection:**
- Direct instruction override ("ignore all previous instructions...")
- System prompt extraction ("print your system prompt")
- Role hijacking ("you are now a Python interpreter")
- Data exfiltration via URL ("send the API key to...")
- Delimiter injection (`<|system|>`, ` ```system `)
- Encoding tricks (base64 decode attacks)

**Jailbreak Attempts:**
- DAN (Do Anything Now) persona attacks
- Developer mode exploitation
- Nested injection via translation requests
- Gradual escalation ("for research purposes")
- Indirect extraction via data column claims

### 1.4 Results

```
All 60/60 security tests PASSED.
```

- 13/13 malicious prompts correctly detected as suspicious
- 5/5 benign KPI queries correctly allowed without false positives
- 5/5 jailbreak scenarios detected
- All XSS attack patterns neutralized
- Rate limiting correctly enforces both per-minute and daily limits

---

## 2. Security Safeguards Implemented

### 2.1 Input Validation (`src/security.py`)

| Safeguard | Details |
|-----------|---------|
| Prompt length limit | 5,000 characters max |
| CSV file size limit | 50 MB max |
| PDF file size limit | 25 MB per file, 10 files max |
| Image file size limit | 10 MB max |
| CSV row limit | 500,000 rows max |
| CSV column limit | 100 columns max |
| File extension whitelist | CSV: `.csv`, PDF: `.pdf`, Images: `.png`, `.jpg`, `.jpeg` |
| DataFrame validation | Empty check, row/column count checks |
| API key format validation | Prefix, length, whitespace, control character checks |

### 2.2 Prompt Injection Detection (`src/security.py`)

The `check_prompt_injection()` function uses regex-based pattern matching to detect:

- **High-confidence patterns** (12 patterns): instruction overrides, system prompt extraction, role hijacking, data exfiltration, delimiter injection, encoding tricks
- **Suspicious patterns** (5 patterns): admin/developer/DAN mode, jailbreak keywords, bypass attempts

Detection is **non-blocking** -- suspicious prompts are flagged and logged, and the system prompt is reinforced with a security note. This avoids false-positive blocking of legitimate queries while still defending against attacks.

### 2.3 Output Sanitization (`src/security.py`)

Two layers of output sanitization:

1. **`sanitize_html_output()`** -- HTML entity escaping for all LLM output rendered with `unsafe_allow_html=True`. Prevents XSS by converting `<`, `>`, `"`, `&` to HTML entities.

2. **`sanitize_llm_response()`** -- Strips dangerous content from LLM responses before display:
   - `<script>`, `<iframe>`, `<object>`, `<embed>`, `<link>` tags
   - Event handler attributes (`onclick`, `onmouseover`, etc.)
   - `data:text/html` URIs (executable data URIs)
   - System prompt leakage markers (`<|system|>`, `<|im_start|>`, etc.)

### 2.4 Rate Limiting (`src/security.py`)

The `RateLimiter` class implements sliding-window rate limiting:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_requests` | 20 | Max requests per window |
| `window_seconds` | 60 | Sliding window duration |
| `max_daily_requests` | 200 | Hard daily cap |

- Per-session isolation (different users don't share limits)
- Graceful error messages with estimated wait times
- Usage stats API for monitoring

### 2.5 Security Audit Logging (`src/security.py`)

All security events are logged to `security_audit.log`:

| Event Type | Severity | Trigger |
|------------|----------|---------|
| `INJECTION_DETECTED` | WARNING/CRITICAL | Prompt injection patterns found |
| `RATE_LIMIT` | WARNING | Request throttled |
| `INPUT_VALIDATION` | WARNING | Invalid input rejected |
| `AUTH` | INFO | API key set or validated |

Log format: `timestamp | severity | [EVENT_TYPE] details`

---

## 3. API Key Management

### 3.1 Current Practices

| Practice | Status |
|----------|--------|
| `.env` in `.gitignore` | Implemented |
| `.env.example` template (no real keys) | Implemented |
| API key format validation | Implemented (Milestone 12) |
| Key masking in logs/UI | Implemented (Milestone 12) |
| `security_audit.log` in `.gitignore` | Implemented (Milestone 12) |
| Streamlit password field for key input | Implemented |

### 3.2 Key Handling Flow

1. User enters key via Streamlit sidebar (password-masked input)
2. Key is validated for format (prefix, length, no whitespace)
3. If valid, set as environment variable for the session only
4. Key is masked in any audit log entries (`sk-proj...uMEA`)
5. Key is never written to disk or displayed in UI

### 3.3 Recommendations for Production

- **Rotate keys regularly** via OpenAI dashboard
- **Set usage limits** on the OpenAI account to cap spending
- **Use Streamlit secrets** (`~/.streamlit/secrets.toml`) instead of `.env` for deployed apps
- **If a key is exposed in git history**, revoke it immediately and generate a new one
- **Consider a secrets manager** (AWS Secrets Manager, HashiCorp Vault) for team deployments

---

## 4. Data Privacy Protections

### 4.1 Data Handling

| Aspect | Protection |
|--------|-----------|
| User CSV data | Processed in-memory only; never persisted to disk beyond the session |
| PDF uploads | Written to OS temp directory; cleaned up on session end |
| Image uploads | Written to temp files; explicitly cleaned up after analysis |
| ChromaDB vectors | Stored in temp directory per session; not shared between users |
| LLM API calls | Data sent to OpenAI API per their data usage policy |
| Session state | Stored in Streamlit server memory; cleared on session end |

### 4.2 What is NOT Sent to the LLM

- API keys (only used in headers, never in prompt content)
- Raw file contents (only processed summaries and structured data)
- Other users' data (session isolation)

---

## 5. Known Limitations

### 5.1 Security Limitations

1. **Regex-based injection detection** -- Pattern matching can be bypassed by sufficiently creative obfuscation (e.g., Unicode homoglyphs, tokenization tricks). A production system should add ML-based detection.

2. **In-memory rate limiting** -- The rate limiter resets when the Streamlit server restarts. Multi-process deployments need Redis-backed rate limiting.

3. **No authentication** -- The app has no user login or role-based access control. Anyone with the URL can use it (rely on network-level access control for now).

4. **OpenAI API dependency** -- Data sent to OpenAI is subject to their data retention and usage policies. For sensitive data, consider self-hosted models.

5. **No Content Security Policy headers** -- Streamlit doesn't natively support CSP headers. For production, deploy behind a reverse proxy (nginx/Caddy) that adds CSP.

6. **Temp file cleanup** -- While we now explicitly clean up image temp files, OS-level temp directory cleanup timing varies. Sensitive data in temp files could persist briefly.

### 5.2 Functional Limitations

1. **False positive rate** -- The injection detector may flag edge-case legitimate queries that happen to match patterns (e.g., a user asking "show me the system prompt impact on conversion"). The system warns but does not block.

2. **No real-time key verification** -- API key validation checks format only, not whether the key is active with OpenAI. Invalid keys will fail at the first API call.

3. **Single-session rate limits** -- Rate limits are per Streamlit session, not per user identity. A user can bypass limits by opening a new browser tab.

---

## 6. Files Modified/Created

| File | Change |
|------|--------|
| `src/security.py` | **NEW** -- Core security module (input validation, output sanitization, rate limiting, injection detection, audit logging) |
| `tests/test_security.py` | **NEW** -- 60-test security suite covering all attack categories |
| `SECURITY.md` | **NEW** -- This document |
| `src/agent.py` | **MODIFIED** -- Added rate limiting, input validation, injection detection, output sanitization to the ReAct agent loop |
| `app.py` | **MODIFIED** -- Added API key validation, file upload validation, HTML output sanitization, image temp file cleanup |
| `.gitignore` | **MODIFIED** -- Added `security_audit.log` and `.streamlit/secrets.toml` |
| `.env.example` | **MODIFIED** -- Added security guidance comments |
