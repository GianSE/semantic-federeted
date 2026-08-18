# Resumo SICITE 2026

Resumo estruturado para o **XXXI Seminário de Iniciação Científica e Tecnológica
da UTFPR**, derivado do artigo do SBrT 2026 sem reexecutar experimentos: todos os
valores reportados são os já obtidos (CIFAR-10, 5 clientes, 3 rodadas, semente 42).

Template oficial baixado de <https://nuvem.utfpr.edu.br/index.php/s/A44PSTJRCwThvum>
e mantido sem alterações — só os arquivos de `preenchimento/` foram editados.

## Como compilar

**No Overleaf** (recomendado; o pacote oficial traz um passo a passo em PDF):

1. Suba esta pasta inteira como projeto.
2. Menu > Compiler > **pdfLaTeX**.
3. Defina o *Main document* conforme a versão desejada:
   - `main-com-identificacao.tex` → PDF **com** autores (para os anais)
   - `main-sem-identificacao.tex` → PDF **sem** autores (avaliação cega)

Ambos os PDFs são obrigatórios na submissão.

## Estrutura

| Arquivo | O que contém |
|---|---|
| `preenchimento/resumo.tex` | O texto do resumo (compartilhado pelas duas versões) |
| `preenchimento/dados-com-identificacao.tex` | Título, autores, afiliação, e-mail, área, palavras-chave |
| `preenchimento/dados-sem-identificacao.tex` | Idem, sem dados de autoria |
| `preenchimento/agradecimentos-*.tex` | Agradecimentos |
| `preenchimento/referencias-*.bib` | Referências (mesmas chaves nos dois arquivos) |
| `configuracoes/`, `sicite.sty` | Template oficial — não editar |

## Compilar localmente

```bash
python build.py          # as duas versões
python build.py --com    # só a versão com identificação
python build.py --sem    # só a versão cega
```

O script roda `pdflatex → bibtex → pdflatex → pdflatex`, grava os PDFs em
`build/` e verifica automaticamente as duas regras eliminatórias do evento
(limite de 2 páginas e ausência de vazamento de autoria na versão cega), além
de citações e referências pendentes. Requer `pdflatex` e `bibtex` no PATH.

## Estado da verificação

Compilado com pdfLaTeX (TeX Live 2025) — o mesmo motor indicado pelo template e
usado no Overleaf.

- Versão com identificação: **2 páginas** ✅
- Versão sem identificação: **2 páginas** ✅
- Citações indefinidas: nenhuma, nas duas versões
- Versão cega: nenhum termo de autoria encontrado ✅

Sobra cerca de um quarto da segunda página. Se precisar cortar depois de
preencher título e área, os pontos naturais são o *Contexto* e a lista de
trabalhos futuros na *Conclusão*; se sobrar espaço, são os mesmos pontos para
reexpandir.

## Pendências antes de submeter

- [ ] **Título idêntico ao do SISPEQ.** O regulamento exige correspondência
      exata com o plano de trabalho cadastrado pelo orientador. O título atual
      foi adaptado do artigo do SBrT e precisa ser conferido — divergência
      obriga reenvio.
- [x] **Área temática:** `09 -- Engenharia Elétrica`, preenchida nas duas
      versões. Escolhida por ser a específica de telecomunicações e
      corresponder ao departamento; `14 -- Engenharias` é a categoria genérica
      e `05 -- Ciência/Engenharia da Computação/Software` deslocaria o enfoque
      para aprendizado de máquina. Selecione a mesma no Even3.
- [ ] **E-mail.** Deve ser o mesmo cadastrado no SISPEQ.
- [ ] **Agradecimentos.** Incluir a agência de fomento correta se houver bolsa
      (CNPq, Fundação Araucária, PIBIC/PIBITI); caso contrário, manter apenas
      a menção à UTFPR.
- [ ] **Ordem dos autores:** apresentador, coautores e, por último, o
      orientador — igual nos dois documentos.

Prazo de submissão: **1º de julho a 23 de agosto de 2026**.

## Restrições do template

Não são permitidos subseções, gráficos, figuras ou tabelas. Por isso os
resultados aparecem no corpo do texto, e não em tabela — diferentemente da
versão do SBrT em `docs/paper/`.
