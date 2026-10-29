# Política de Segurança e Proteção de Dados (Security Policy)

Este projeto foi desenhado sob princípios estritos de **Segurança por Conceção (*Security by Design*)**, proteção de privacidade e **Zero Exposição de Credenciais**, garantindo total imunidade contra fuga de dados sensíveis ou vetores de acesso remoto indevido.

---

## 1. Princípios de Proteção do Repositório

1. **Execução 100% Local e Isolada:**
   - O projeto executa de forma totalmente autónoma na máquina local através de scripts Python locais.
   - Não existem portas de rede abertas, serviços de escuta (*listeners*), *sockets* remotos, servidores web expostos nem acessos remotos (*backdoors* ou pontes de telemetria).
2. **Zero Informação Pessoal Identificável (PII):**
   - Todos os caminhos de ficheiros estão normalizados para caminhos relativos ao projeto (`.`, `data/`, `models/`, `logs/`).
   - Não existem nomes de utilizadores locais, diretórios do sistema operativo (ex.: `C:\Users\...` ou `D:\...`), endereços IP nem dados pessoais no código, metadados ou *outputs* de notebooks.
3. **Gestão Segura de Segredos e APIs:**
   - Todas as chaves e credenciais sensíveis devem residir exclusivamente no ficheiro `.env` (ignorado automaticamente pelo `.gitignore`).
   - O ficheiro `.env.example` fornecido contém unicamente marcadores genéricos vazios (`your_api_key_here`).
   - Os dados utilizados provêm exclusivamente de APIs públicas abertas (OMIE, Open-Meteo, Yahoo Finance).
4. **Verificação de Integridade Criptográfica:**
   - Todos os ficheiros de dados e partições processadas possuem *checksums* criptográficos SHA-256 registados em `data/processed/manifest.json`.
   - Qualquer adulteração inadvertida ou corrupção de ficheiros é detetada de imediato no arranque dos algoritmos.

---

## 2. Verificação de Segurança Pré-Publicação

Antes de submeter código ou criar novos *commits*, execute o verificador de segurança local:

```bash
python scripts/security_check.py
```

O script verifica automaticamente a ausência de:
- Ficheiros `.env` rastreados pelo Git;
- Chaves de API, *tokens* ou credenciais privadas no código e nos ficheiros `.ipynb`;
- Caminhos absolutos de sistema ou nomes de utilizadores locais;
- Portas de escuta ou módulos de acesso remoto.

---

## 3. Reporte de Vulnerabilidades

Se identificar alguma vulnerabilidade potencial ou falha de segurança no código deste repositório, por favor não abra uma *Issue* pública. Contacte o mantenedor de forma responsável através de canal privado no GitHub ou crie um *Advisory* de segurança privado.
