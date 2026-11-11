"""
================================================================================
CANVAS BACKEND SERVER (100% LOCALHOST & ZERO EXTERNAL DEPENDENCIES)
================================================================================
Provides a lightweight local HTTP API connecting workflow_canvas.html
directly to real Python execution of all agent notebooks.

Features:
- Binds strictly to 127.0.0.1 (Localhost only)
- Zero third-party dependencies (uses Python standard library only)
- Real-time cell-by-cell notebook execution and stdout streaming
- Serves workflow_canvas.html directly at http://127.0.0.1:8765/
================================================================================
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.parse

BASE_DIR = Path(__file__).resolve().parent.parent

AGENTS_PIPELINE = [
    {"key": "ag1", "name": "Agente 1: Pesquisa & Relevância", "notebook": "reports/01_pesquisa.ipynb"},
    {"key": "ag2", "name": "Agente 2: Construtor Dataset Base", "notebook": "reports/02_dataset.ipynb"},
    {"key": "ag3", "name": "Agente 3: Validação & EDA", "notebook": "reports/03_validacao_eda.ipynb"},
    {"key": "ag4", "name": "Agente 4: Correção & Higienização", "notebook": "reports/04_correcao.ipynb"},
    {"key": "ag5", "name": "Agente 5: Engenharia de Features", "notebook": "reports/05_features.ipynb"},
    {"key": "ag6", "name": "Agente 6: Modelação XGBoost (100 Treinos)", "notebook": "reports/06_ml.ipynb"},
    {"key": "ag7", "name": "Agente 7: Avaliação, SHAP & Recomendações", "notebook": "reports/07_analise_resultados.ipynb"},
    {"key": "coord", "name": "Coordenador: Relatório Final & Contratos", "notebook": "reports/99_relatorio_final.ipynb"},
]

ALLOWED_STATIC_PREFIXES = ("reports/", "logs/", "models/", "src/", "data/")
ALLOWED_STATIC_FILES = {"README.md", "SECURITY.md", "config.yaml"}

state_lock = threading.Lock()
pipeline_state = {
    "online": True,
    "running": False,
    "aborted": False,
    "current_agent": None,
    "current_notebook": None,
    "completed_count": 0,
    "global_progress_pct": 0,
    "agent_states": {
        a["key"]: {
            "status": "standby",
            "pct": 0,
            "action": "A aguardar execução...",
            "notebook": a["notebook"],
            "logs": [],
        }
        for a in AGENTS_PIPELINE
    },
    "recent_logs": [
        "[SYSTEM READY] Servidor Python inicializado em 127.0.0.1:8765",
        "[SYSTEM READY] 8 Agentes mapeados e prontos para execução real.",
    ],
}

active_process = None


def add_log(msg: str) -> None:
  with state_lock:
    timestamp = time.strftime("%H:%M:%S")
    pipeline_state["recent_logs"].append(f"[{timestamp}] {msg}")
    if len(pipeline_state["recent_logs"]) > 250:
      pipeline_state["recent_logs"].pop(0)


def is_allowed_static_path(url_path: str) -> bool:
  """Allows only explicit project subdirectories/files and blocks dotfiles."""
  rel = urllib.parse.unquote(url_path).lstrip("/")
  if not rel or any(part.startswith(".") for part in Path(rel).parts):
    return False
  if rel in ALLOWED_STATIC_FILES:
    return True
  return any(rel.startswith(prefix) for prefix in ALLOWED_STATIC_PREFIXES)


def run_pipeline_worker() -> None:
  global active_process

  with state_lock:
    pipeline_state["running"] = True
    pipeline_state["aborted"] = False
    pipeline_state["completed_count"] = 0
    pipeline_state["global_progress_pct"] = 0
    for k in pipeline_state["agent_states"]:
      pipeline_state["agent_states"][k]["status"] = "standby"
      pipeline_state["agent_states"][k]["pct"] = 0
      pipeline_state["agent_states"][k]["action"] = "A aguardar..."
      pipeline_state["agent_states"][k]["logs"] = []

  add_log(">>> INICIALIZANDO EXECUÇÃO SEQUENCIAL DOS 8 AGENTES <<<")

  python_executable = sys.executable
  exec_script = str(BASE_DIR / "scripts" / "execute_notebook.py")

  for idx, agent in enumerate(AGENTS_PIPELINE):
    key = agent["key"]
    nb_path = agent["notebook"]

    with state_lock:
      if pipeline_state["aborted"]:
        add_log("[ABORTED] Execução cancelada pelo utilizador.")
        break
      pipeline_state["current_agent"] = key
      pipeline_state["current_notebook"] = nb_path
      pipeline_state["agent_states"][key]["status"] = "processing"
      pipeline_state["agent_states"][key]["action"] = f"A iniciar {nb_path}..."

    add_log(f"-> A iniciar Nó {idx+1}/8: {agent['name']} ({nb_path})")
    target_nb = str(BASE_DIR / nb_path)
    cmd = [str(python_executable), exec_script, target_nb]

    try:
      proc = subprocess.Popen(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          encoding="utf-8",
          errors="replace",
          bufsize=1,
          cwd=str(BASE_DIR),
      )
      with state_lock:
        active_process = proc

      cell_count = 0
      assert proc.stdout is not None
      for line in iter(proc.stdout.readline, ""):
        line_str = line.strip()
        if not line_str:
          continue

        if "Executing Cell" in line_str:
          cell_count += 1
          sim_pct = min(95, 10 + cell_count * 12)
          with state_lock:
            pipeline_state["agent_states"][key]["pct"] = sim_pct
            pipeline_state["agent_states"][key]["action"] = line_str

        add_log(f"[{key.upper()}] {line_str}")
        with state_lock:
          pipeline_state["agent_states"][key]["logs"].append(line_str)
          total_pct = sum(
              pipeline_state["agent_states"][k]["pct"]
              for k in pipeline_state["agent_states"]
          )
          pipeline_state["global_progress_pct"] = int(
              total_pct / len(AGENTS_PIPELINE)
          )

      proc.stdout.close()
      ret_code = proc.wait()
      with state_lock:
        active_process = None

      if ret_code == 0:
        with state_lock:
          pipeline_state["agent_states"][key]["status"] = "complete"
          pipeline_state["agent_states"][key]["pct"] = 100
          pipeline_state["agent_states"][key]["action"] = (
              "Concluído com sucesso (Quality Gate aprovado)"
          )
          pipeline_state["completed_count"] += 1
          total_pct = sum(
              pipeline_state["agent_states"][k]["pct"]
              for k in pipeline_state["agent_states"]
          )
          pipeline_state["global_progress_pct"] = int(
              total_pct / len(AGENTS_PIPELINE)
          )
        add_log(f"[CONCLUÍDO] {agent['name']} finalizado com sucesso!")
      else:
        with state_lock:
          pipeline_state["agent_states"][key]["status"] = "error"
          pipeline_state["agent_states"][key]["action"] = (
              f"Erro na execução (código {ret_code})"
          )
        add_log(f"[ERRO] {agent['name']} falhou com código {ret_code}")
        break

    except Exception as e:
      with state_lock:
        active_process = None
        pipeline_state["agent_states"][key]["status"] = "error"
        pipeline_state["agent_states"][key]["action"] = str(e)
      add_log(f"[EXCEPTION] Falha ao executar {nb_path}: {e}")
      break

    time.sleep(0.4)

  with state_lock:
    pipeline_state["running"] = False
    pipeline_state["current_agent"] = None
    pipeline_state["current_notebook"] = None
    if not pipeline_state["aborted"]:
      pipeline_state["global_progress_pct"] = 100
  add_log(">>> EXECUÇÃO COMPLETA: TODOS OS AGENTES PROCESSADOS <<<")


class CanvasHTTPRequestHandler(SimpleHTTPRequestHandler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=str(BASE_DIR), **kwargs)

  def end_headers(self):
    origin = self.headers.get("Origin", "")
    if origin.startswith("http://127.0.0.1:") or origin.startswith(
        "http://localhost:"
    ):
      self.send_header("Access-Control-Allow-Origin", origin)
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    self.send_header("Access-Control-Allow-Headers", "Content-Type")
    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    super().end_headers()

  def do_OPTIONS(self):
    self.send_response(200)
    self.end_headers()

  def do_GET(self):
    parsed = urllib.parse.urlparse(self.path)

    if parsed.path in ["", "/", "/index.html", "/canvas"]:
      canvas_file = BASE_DIR / "reports" / "workflow_canvas.html"
      if canvas_file.exists():
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(canvas_file.read_bytes())
        return

    if parsed.path == "/api/status":
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      with state_lock:
        data = json.dumps(pipeline_state).encode("utf-8")
      self.wfile.write(data)
      return

    if parsed.path == "/api/logs":
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      with state_lock:
        data = json.dumps({"logs": pipeline_state["recent_logs"]}).encode(
            "utf-8"
        )
      self.wfile.write(data)
      return

    if parsed.path == "/api/open-file":
      query_params = urllib.parse.parse_qs(parsed.query)
      file_rel = query_params.get("path", [None])[0]
      self._handle_open_file(file_rel)
      return

    if not is_allowed_static_path(parsed.path):
      self.send_response(403)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps({"error": "Acesso proibido a este caminho"}).encode("utf-8")
      )
      return

    super().do_GET()

  def do_POST(self):
    parsed = urllib.parse.urlparse(self.path)

    if parsed.path == "/api/run-all":
      with state_lock:
        if pipeline_state["running"]:
          self.send_response(409)
          self.send_header("Content-Type", "application/json")
          self.end_headers()
          self.wfile.write(
              json.dumps({"error": "Pipeline já se encontra em execução"}).encode(
                  "utf-8"
              )
          )
          return

      thread = threading.Thread(target=run_pipeline_worker, daemon=True)
      thread.start()

      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps({"status": "Pipeline iniciado com sucesso"}).encode("utf-8")
      )
      return

    if parsed.path == "/api/abort":
      with state_lock:
        pipeline_state["aborted"] = True
        pipeline_state["running"] = False
        proc = active_process
      if proc and proc.poll() is None:
        proc.terminate()
      add_log("[SYSTEM] Sinal de abortar recebido.")
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(json.dumps({"status": "Abortado"}).encode("utf-8"))
      return

    if parsed.path == "/api/open-file":
      query_params = urllib.parse.parse_qs(parsed.query)
      file_rel = query_params.get("path", [None])[0]
      if not file_rel:
        try:
          content_len = int(self.headers.get("Content-Length", 0))
          if content_len > 0:
            body = self.rfile.read(content_len).decode("utf-8")
            file_rel = json.loads(body).get("path")
        except Exception:
          file_rel = None
      self._handle_open_file(file_rel)
      return

    self.send_response(404)
    self.end_headers()

  def _handle_open_file(self, file_rel):
    if not file_rel:
      self.send_response(400)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps({"error": "Parametro 'path' em falta"}).encode("utf-8")
      )
      return

    clean_path = urllib.parse.unquote(file_rel).strip().replace("/", os.sep)
    target_path = (BASE_DIR / clean_path).resolve()

    try:
      rel_path = target_path.relative_to(BASE_DIR)
    except ValueError:
      self.send_response(403)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps(
              {"error": "Acesso nao permitido fora do diretorio do projeto"}
          ).encode("utf-8")
      )
      return

    if any(part.startswith(".") for part in rel_path.parts):
      self.send_response(403)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps({"error": "Acesso proibido a ficheiros ocultos"}).encode(
              "utf-8"
          )
      )
      return

    if not target_path.exists():
      add_log(f"[AVISO] Ficheiro ainda nao gerado: {target_path.name}")
      self.send_response(404)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps(
              {"error": f"Ficheiro ainda nao gerado: {target_path.name}"}
          ).encode("utf-8")
      )
      return

    try:
      os.startfile(str(target_path))
      add_log(f"[ABRIR] Ficheiro aberto no Windows: {target_path.name}")
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps({"status": "ok", "opened": str(target_path)}).encode(
              "utf-8"
          )
      )
    except Exception:
      try:
        subprocess.Popen(["explorer", f"/select,{target_path}"])
        add_log(f"[ABRIR] Localizado no Explorador: {target_path.name}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({
                "status": "ok",
                "opened": str(target_path),
                "fallback": "explorer",
            }).encode("utf-8")
        )
      except Exception as e2:
        add_log(f"[ERRO] Falha ao abrir {target_path.name}: {e2}")
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": str(e2)}).encode("utf-8"))

  def log_message(self, format, *args):
    pass


def start_server(port: int = 8765) -> None:
  server_address = ("127.0.0.1", port)
  httpd = HTTPServer(server_address, CanvasHTTPRequestHandler)
  print("=" * 72)
  print("MIBEL WORKFLOW CANVAS • SERVIDOR PYTHON ATIVO")
  print("=" * 72)
  print(f"[LOCAL SERVER] URL: http://127.0.0.1:{port}/")
  print("[STATUS] Ligação direta a todos os notebooks do projeto.")
  print("[INFO] Prima Ctrl+C para encerrar o servidor a qualquer momento.")
  print("=" * 72)

  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    print("\n[SHUTDOWN] Servidor local encerrado com sucesso.")
    httpd.server_close()


if __name__ == "__main__":
  port = 8765
  if len(sys.argv) > 1:
    try:
      port = int(sys.argv[1])
    except ValueError:
      pass
  start_server(port)
