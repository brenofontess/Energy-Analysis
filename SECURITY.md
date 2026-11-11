# Security and Data Protection Policy

This project was designed under strict principles of **Security by Design**, privacy protection, and **Zero Credential Exposure**, ensuring protection against sensitive data leaks and unauthorized remote access vectors.

---

## 1. Repository Protection Principles

1. **100% Local and Isolated Execution:**
   - The project runs autonomously on the local machine via local Python scripts.
   - There are no persistent listening services, remote sockets, or external access vectors (no backdoors or telemetry).
   - The optional visual interface server ([`scripts/canvas_server.py`](scripts/canvas_server.py)) is only started manually by the user, binds strictly to `127.0.0.1:8765` (localhost), and blocks hidden files such as `.env`.
2. **Zero Personally Identifiable Information (PII):**
   - All file paths are normalized to paths relative to the project root (`.`, `data/`, `models/`, `logs/`).
   - No local usernames, operating system directories (e.g., `C:\Users\...` or `D:\...`), IP addresses, or personal data appear in code, metadata, or notebook outputs.
3. **Secure Secrets and API Management:**
   - All sensitive keys and credentials must reside exclusively in the `.env` file (automatically ignored by `.gitignore`).
   - The provided `.env.example` file contains only empty generic placeholders (`your_api_key_here`).
   - The data used comes exclusively from open public APIs (OMIE, Open-Meteo, Yahoo Finance).
4. **Cryptographic Integrity Verification:**
   - All data files and processed partitions have SHA-256 cryptographic checksums recorded in `data/processed/manifest.json`.
   - Any inadvertent tampering or file corruption is detected immediately when the pipelines start.

---

## 2. Pre-Publication Security Check

Before submitting code or creating new commits, run the local security scanner:

```bash
python scripts/security_check.py
```

The script automatically checks for the absence of:
- `.env` files tracked by Git;
- API keys, tokens, or private credentials in source code and `.ipynb` files;
- Absolute system paths or local usernames;
- Listening ports or remote access modules.

---

## 3. Reporting Vulnerabilities

If you identify a potential vulnerability or security flaw in this repository's code, please do not open a public Issue. Contact the maintainer responsibly through a private channel on GitHub or create a private security Advisory.
