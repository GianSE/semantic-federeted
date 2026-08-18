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

## Estado da verificação

Compilado localmente com Tectonic (XeTeX), substituindo *TeX Gyre Termes* por
*Times New Roman* apenas na cópia de teste, por indisponibilidade da primeira
neste ambiente. As duas fontes são metricamente compatíveis, então a paginação
deve se manter; ainda assim, **reconfira ao compilar no Overleaf**.

- Versão com identificação: **2 páginas** ✅
- Versão sem identificação: **2 páginas** ✅
- Citações indefinidas: nenhuma, nas duas versões
- Versão cega: sem vazamento de autoria no log

O limite de 2 páginas é eliminatório e a margem é pequena. Se estourar ao
compilar, os pontos naturais para cortar são o *Contexto* e a lista de trabalhos
futuros na *Conclusão*.

## Pendências antes de submeter

- [ ] **Título idêntico ao do SISPEQ.** O regulamento exige correspondência
      exata com o plano de trabalho cadastrado pelo orientador. O título atual
      foi adaptado do artigo do SBrT e precisa ser conferido — divergência
      obriga reenvio.
- [ ] **Área temática.** O evento tem 19; preencher o número e o nome em
      `dados-com-identificacao.tex` e `dados-sem-identificacao.tex`
      (os dois devem coincidir com a área escolhida na submissão).
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
