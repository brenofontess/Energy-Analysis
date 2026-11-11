import json
import sys
import io
import re
from datetime import datetime, timezone
import traceback
from contextlib import redirect_stdout, redirect_stderr
import base64
from pathlib import Path
import matplotlib

# Set non-interactive Agg backend to ensure NO popup GUI windows ever open
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def run_notebook(nb_path, output_nb_path=None):
    if output_nb_path is None:
        output_nb_path = nb_path
        
    print(f"\n=======================================================")
    print(f"EXECUTING NOTEBOOK: {nb_path} (INLINE MATPLOTLIB MODE)")
    print(f"=======================================================")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # -------------------------------------------------------------------------
    # 1. ATUALIZAR DATA E HORA DE CARREGAMENTO / EXECUÇÃO NO CABEÇALHO INICIAL
    # -------------------------------------------------------------------------
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    date_pattern = re.compile(r"^(\s*-\s*\*\*Data e Hora[^\*]*\*\*[:\s]*).*", re.IGNORECASE)
    english_pattern = re.compile(r"^(\s*-\s*\*\*Date and Time[^\*]*\*\*[:\s]*).*", re.IGNORECASE)
    
    updated_timestamp = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", [])
            new_src = []
            for line in src:
                m_pt = date_pattern.match(line)
                m_en = english_pattern.match(line)
                if m_pt:
                    prefix = m_pt.group(1).rstrip()
                    new_src.append(f"{prefix} {now_utc}\n")
                    updated_timestamp = True
                elif m_en:
                    prefix = m_en.group(1).rstrip()
                    new_src.append(f"{prefix} {now_utc}\n")
                    updated_timestamp = True
                else:
                    new_src.append(line)
            cell["source"] = new_src
            if updated_timestamp:
                print(f"[TIMESTAMP] Data e Hora atualizada no cabecalho: {now_utc}")
                break

    if not updated_timestamp:
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                src = cell.get("source", [])
                insert_idx = None
                for i, line in enumerate(src):
                    if line.strip().startswith("- **"):
                        insert_idx = i + 1
                if insert_idx is not None:
                    src.insert(insert_idx, f"- **Data e Hora de Execucao:** {now_utc}\n")
                    cell["source"] = src
                    updated_timestamp = True
                    print(f"[TIMESTAMP] Nova linha de Data e Hora inserida no cabecalho: {now_utc}")
                break

    # Guardar imediatamente o ficheiro com o cabecalho atualizado
    with open(output_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    nb_globals = {
        "__name__": "__main__",
        "__file__": str(Path(nb_path).resolve())
    }
    
    exec_count = 1
    
    for idx, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type")
        if cell_type != "code":
            continue
            
        src_lines = cell.get("source", [])
        
        # Check if cell contains % magics (mock them during exec, but keep them in notebook source for PyCharm)
        filtered_lines = []
        for line in src_lines:
            stripped = line.strip()
            if stripped.startswith("%uv ") or stripped.startswith("%pip ") or stripped.startswith("%matplotlib"):
                filtered_lines.append(f"# {line}")
            else:
                filtered_lines.append(line)
                
        code_to_exec = "".join(filtered_lines)
        cell["execution_count"] = exec_count
        cell["outputs"] = []
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        captured_figures = []
        
        # Define hook for plt.show to capture figure before closing
        def custom_show(*args, **kwargs):
            fignums = plt.get_fignums()
            if not fignums:
                fig = plt.gcf()
                if fig and fig.get_axes():
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                    buf.seek(0)
                    captured_figures.append(base64.b64encode(buf.read()).decode("utf-8"))
            else:
                for fig_num in fignums:
                    fig = plt.figure(fig_num)
                    if fig.get_axes():
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                        buf.seek(0)
                        captured_figures.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close("all")
            
        plt.show = custom_show
        
        print(f"-> Executing Cell {idx:02d} ...", end=" ", flush=True)
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code_to_exec, nb_globals)
            out_str = stdout_buf.getvalue()
            err_str = stderr_buf.getvalue()
            
            # Also capture any remaining open figures that didn't call plt.show()
            fignums = plt.get_fignums()
            for fig_num in fignums:
                fig = plt.figure(fig_num)
                if fig.get_axes():
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                    buf.seek(0)
                    captured_figures.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close("all")
            
            outputs = []
            if out_str:
                outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": out_str.splitlines(keepends=True)
                })
            if err_str:
                outputs.append({
                    "output_type": "stream",
                    "name": "stderr",
                    "text": err_str.splitlines(keepends=True)
                })
                
            # Attach captured figures as display_data (PNG base64) for inline display in PyCharm / Jupyter
            for img_b64 in captured_figures:
                outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {}
                })
                
            cell["outputs"] = outputs
            exec_count += 1
            print(f"DONE (Captured {len(captured_figures)} inline figures)")
            if out_str:
                summary_lines = out_str.strip().split("\n")
                if len(summary_lines) <= 6:
                    print("   " + "\n   ".join(summary_lines))
                else:
                    print("   " + "\n   ".join(summary_lines[:3]))
                    print(f"   ... [{len(summary_lines)-5} output lines] ...")
                    print("   " + "\n   ".join(summary_lines[-2:]))
                    
        except Exception as e:
            err_msg = traceback.format_exc()
            print("ERROR!")
            print(err_msg)
            cell["outputs"] = [{
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": err_msg.splitlines(keepends=True)
            }]
            with open(output_nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2, ensure_ascii=False)
            raise e
            
    with open(output_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Notebook {nb_path} executed cleanly and saved with {sum(1 for c in nb['cells'] for o in c.get('outputs', []) if 'image/png' in o.get('data', {}))} embedded inline figures!")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "reports/07_analise_resultados.ipynb"
    run_notebook(target)
