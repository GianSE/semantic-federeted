"""Compila as duas versoes do resumo do SICITE e verifica as regras do evento.

Uso:
    python build.py            # compila as duas versoes
    python build.py --com      # apenas a versao com identificacao
    python build.py --sem      # apenas a versao sem identificacao

Requer pdflatex e bibtex no PATH (TinyTeX, MiKTeX ou TeX Live). Os PDFs sao
gravados em build/, que fica fora do controle de versao.

O script nao substitui a compilacao no Overleaf, mas verifica automaticamente
as duas regras eliminatorias do evento -- limite de 2 paginas e ausencia de
vazamento de autoria na versao cega -- alem de citacoes ou referencias
pendentes.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).parent
SAIDA = RAIZ / "build"
LIMITE_PAGINAS = 2

# As duas versoes compartilham o mesmo nome-base de proposito: um nome de
# arquivo que identificasse o autor anularia o anonimato da avaliacao cega.
DOC_COM = "sicite2026-comunicacao-semantica-federada-com-identificacao"
DOC_SEM = "sicite2026-comunicacao-semantica-federada-sem-identificacao"

# Termos que jamais devem aparecer na versao destinada a avaliacao cega.
TERMOS_AUTORIA = [
    "Gian", "Pedro Rodrigues", "Herman", "dos Santos",
    "utfpr.edu", "Cornélio", "Cornelio", "Tecnológica", "Tecnologica",
]


def executa(programa, *args, cwd):
    resultado = subprocess.run(
        [programa, *args], cwd=cwd, capture_output=True, text=True, errors="replace"
    )
    return resultado


def compila(nome):
    """Roda pdflatex/bibtex/pdflatex/pdflatex e devolve o caminho do PDF."""
    SAIDA.mkdir(exist_ok=True)
    passos = [
        ("pdflatex", ["-interaction=nonstopmode", "-halt-on-error",
                      f"-output-directory={SAIDA.name}", f"{nome}.tex"]),
        ("bibtex", [f"{SAIDA.name}/{nome}"]),
        ("pdflatex", ["-interaction=nonstopmode", "-halt-on-error",
                      f"-output-directory={SAIDA.name}", f"{nome}.tex"]),
        ("pdflatex", ["-interaction=nonstopmode", "-halt-on-error",
                      f"-output-directory={SAIDA.name}", f"{nome}.tex"]),
    ]
    for programa, args in passos:
        resultado = executa(programa, *args, cwd=RAIZ)
        # O bibtex reclama de .aux ausente na primeira passada de um projeto
        # limpo; so o pdflatex e tratado como erro fatal.
        if resultado.returncode != 0 and programa == "pdflatex":
            print(f"\n[ERRO] {programa} falhou em {nome}:\n")
            for linha in resultado.stdout.splitlines():
                if linha.startswith("!") or "Error" in linha:
                    print("   ", linha)
            return None
    return SAIDA / f"{nome}.pdf"


def conta_paginas(nome):
    log = SAIDA / f"{nome}.log"
    if not log.exists():
        return None
    texto = log.read_text(encoding="utf-8", errors="replace")
    achado = re.search(r"Output written on .*?\((\d+) pages?", texto)
    return int(achado.group(1)) if achado else None


def problemas_no_log(nome):
    log = SAIDA / f"{nome}.log"
    texto = log.read_text(encoding="utf-8", errors="replace")
    return {
        "citacoes indefinidas": len(re.findall(r"Citation .* undefined", texto)),
        "referencias indefinidas": len(re.findall(r"Reference .* undefined", texto)),
        "overfull hbox": len(re.findall(r"Overfull \\hbox", texto)),
    }


def verifica_anonimato(nome):
    """Procura dados de autoria no texto extraido do log da versao cega."""
    log = (SAIDA / f"{nome}.log").read_text(encoding="utf-8", errors="replace")
    fontes = [RAIZ / "preenchimento" / "resumo.tex",
              RAIZ / "preenchimento" / "dados-sem-identificacao.tex",
              RAIZ / "preenchimento" / "agradecimentos-sem-identificacao.tex",
              RAIZ / "preenchimento" / "referencias-sem-identificacao.bib"]
    conteudo = "\n".join(f.read_text(encoding="utf-8") for f in fontes if f.exists())
    return [t for t in TERMOS_AUTORIA if t in conteudo or t in log]


def relatorio(nome, rotulo, checar_anonimato=False):
    print(f"\n{'=' * 62}\n{rotulo}\n{'=' * 62}")
    pdf = compila(nome)
    if pdf is None:
        return False

    paginas = conta_paginas(nome)
    ok = paginas is not None and paginas <= LIMITE_PAGINAS
    marca = "OK" if ok else "ESTOUROU"
    print(f"  paginas: {paginas} (limite {LIMITE_PAGINAS})  -> {marca}")

    for chave, quantidade in problemas_no_log(nome).items():
        if quantidade:
            print(f"  {chave}: {quantidade}")

    if checar_anonimato:
        vazamentos = verifica_anonimato(nome)
        if vazamentos:
            print(f"  VAZAMENTO DE AUTORIA: {vazamentos}")
            ok = False
        else:
            print("  anonimato: nenhum termo de autoria encontrado -> OK")

    print(f"  PDF: {pdf}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--com", action="store_true", help="so a versao com identificacao")
    parser.add_argument("--sem", action="store_true", help="so a versao sem identificacao")
    args = parser.parse_args()

    if shutil.which("pdflatex") is None:
        sys.exit("pdflatex nao encontrado no PATH. Instale TinyTeX, MiKTeX ou TeX Live.")

    fazer_com = args.com or not args.sem
    fazer_sem = args.sem or not args.com

    tudo_ok = True
    if fazer_com:
        tudo_ok &= relatorio(DOC_COM, "VERSAO COM IDENTIFICACAO (anais)")
    if fazer_sem:
        tudo_ok &= relatorio(
            DOC_SEM, "VERSAO SEM IDENTIFICACAO (avaliacao cega)",
            checar_anonimato=True,
        )

    print()
    if tudo_ok:
        print("Tudo dentro das regras. Confira o resultado visual antes de submeter.")
    else:
        print("Ha pendencias acima. Corrija antes de submeter.")
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
