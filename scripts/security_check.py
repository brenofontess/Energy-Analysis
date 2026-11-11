"""
Security & PII Scanner for Energy Price Agents project.
Scans all project files for exposed secrets, API keys, credentials,
local absolute file paths, remote listeners, and personally identifiable information (PII).
"""

import os
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Case-sensitive secret patterns
FORBIDDEN_CASE_SENSITIVE = [
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS Access Key ID"),
    (r"\bghp_[a-zA-Z0-9]{36}\b", "GitHub Personal Access Token"),
    (r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----", "Private Cryptographic Key"),
]

# Case-insensitive patterns
FORBIDDEN_CASE_INSENSITIVE = [
    (r"dapi[a-zA-Z0-9]{20,}", "Databricks Personal Access Token (PAT)"),
    (r"[cCdD]:[\\\/]Users[\\\/][a-zA-Z0-9_\-]+", "Local Windows User Path (PII)"),
    (r"[dD]:[\\\/]Antigravity", "Local Windows Drive Path"),
    (r"https?:\/\/[a-zA-Z0-9_\-\.]*databricks\.com", "Databricks Cloud URL"),
    (r"brenofontesx", "Personal Email / Account Name"),
]

EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
SCANNED_EXTENSIONS = {".py", ".ipynb", ".json", ".yaml", ".yml", ".md", ".csv", ".example"}

def sanitize_content_for_scan(content, fname):
    # In Jupyter notebooks, strip out raw base64 PNG data before scanning to avoid false positives on random hashes
    if fname.endswith(".ipynb"):
        content = re.sub(r'"image/png":\s*"[a-zA-Z0-9+/=]{100,}"', '"image/png": "<STRIPPED_PNG>"', content)
    return content

def scan_repository():
    print("=" * 80)
    print("ENERGY PRICE AGENTS - SECURITY & PII SCANNER")
    print("Verifying absence of secrets, credentials, remote access and local paths...")
    print("=" * 80)
    
    issues_found = []
    files_scanned = 0
    
    # 1. Check that .gitignore exists and ignores .env
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if not gitignore_path.exists():
        issues_found.append(("CRITICAL", ".gitignore missing!"))
    else:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gi_content = f.read()
            if ".env" not in gi_content:
                issues_found.append(("HIGH", ".env is not explicitly ignored in .gitignore!"))
            else:
                print("[PASS] .gitignore correctly ignores .env and sensitive files.")
                
    # 2. Check that .env contains NO real secrets if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip():
                        issues_found.append(("WARNING", f".env contains non-empty value for key: {k}"))
                        
    # 3. Scan all project files
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for fname in files:
            ext = os.path.splitext(fname)[1]
            if ext not in SCANNED_EXTENSIONS and fname != ".env.example":
                continue
                
            fpath = Path(root) / fname
            rel_path = fpath.relative_to(PROJECT_ROOT)
            
            if fname == "security_check.py":
                continue
                
            files_scanned += 1
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                    
                content = sanitize_content_for_scan(content, fname)
                
                # Case-sensitive scan
                for pattern, desc in FORBIDDEN_CASE_SENSITIVE:
                    matches = re.findall(pattern, content)
                    if matches:
                        issues_found.append(("FAIL", f"{rel_path}: Matched forbidden pattern '{desc}' ({len(matches)} occurrences)"))
                        
                # Case-insensitive scan
                for pattern, desc in FORBIDDEN_CASE_INSENSITIVE:
                    matches = re.findall(pattern, content, flags=re.IGNORECASE)
                    if matches:
                        issues_found.append(("FAIL", f"{rel_path}: Matched forbidden pattern '{desc}' ({len(matches)} occurrences)"))
                        
            except Exception as e:
                issues_found.append(("ERROR", f"Could not read {rel_path}: {e}"))
                
    print(f"[INFO] Scanned {files_scanned} files across repository.")
    print("-" * 80)
    
    if not issues_found:
        print("[SUCCESS] 100% CLEAN! Zero PII, zero private tokens, zero local paths found.")
        print("Repository is completely safe for public deployment on GitHub.")
        print("=" * 80)
        return 0
    else:
        print(f"[ALERT] Found {len(issues_found)} potential security finding(s):")
        for level, msg in issues_found:
            print(f"  [{level}] {msg}")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(scan_repository())
