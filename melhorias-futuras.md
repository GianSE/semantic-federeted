# Melhorias futuras: branch `melhorias-sbrt2026`

Existe uma branch remota, `origin/melhorias-sbrt2026`, com um histórico próprio e divergente da `main` (parte do mesmo commit inicial `74fcd88`). Ela não está mesclada e não deve ser apagada — implementa, item por item, a seção "Limitações e Trabalhos Futuros" do paper aceito na SBrT 2026, numa sessão anterior (commits coautorados por "Claude Opus 5").

**Importante — isso não são bugs urgentes.** Toda limitação que essa branch resolve já está declarada explicitamente, com precisão técnica correta, no próprio `docs/paper/main.tex` (Seções III.B e V). O paper aceito é honesto e transparente sobre o próprio escopo; nada ali está errado ou enganoso. O valor dessa branch é transformar "trabalho futuro" em resultado validado — útil para um camera-ready, uma extensão de journal, ou a próxima IC/TCC.

## O que tem lá, por commit

| Commit | Fase | O que implementa | Limitação do paper que resolve |
|---|---|---|---|
| `384ee66` | 0 | `device.py` (auto CPU/CUDA), persistência resumível por hash (`results/runs/<hash>.json`, nunca duplica), histórico por rodada, `data.py` com `--train-fraction`, remove código morto | infraestrutura (não corresponde a uma limitação específica do texto) |
| `aa369ae` | 1 | `channel.py`: normalização de potência (`E[\|z_i\|²]=1`) + ruído parametrizado por SNR em dB | *"σ deve ser lido como [...] não como uma relação sinal-ruído de canal em dB [...] é uma extensão necessária deste trabalho"* (Seção III.B) |
| `a1a313d` | 2 | imagem bruta contada em 8 bits/pixel (uint8 nativo); quantização do latente com straight-through estimator (`--latent-bits`) | *"a quantização do latente para 8 bits [...] não implementada aqui"* (Seção IV.A, "Convenção de Contagem de Bits") |
| `cf57fc2` | 3 | Rayleigh/Rician com mapeamento complexo em banda base e equalização CSI perfeita, validando amplificação de ruído por `1/\|h\|` | *"Canais representativos de redes móveis [...] não são contemplados"* (Seção V) |
| `f3faa98` | 4 | `perturb_state_dict()`: AWGN no uplink dos pesos do FedAvg antes da agregação, SNR por tensor | *"não modelamos [...] o efeito de um enlace ruidoso sobre a transmissão dos pesos no FedAvg"* (Seção V) |
| `5c65805` | 5 | Partição Dirichlet com rejeição de sorteios degenerados + heatmap de heterogeneidade (`--beta`) | *"requer particionamento não-IID — por distribuição de Dirichlet [...] por exemplo"* (Seção V) |
| `1c5a524` | 6 | Bibliografia: adiciona PSFL e FedCL (citados nominalmente pelos revisores), reposiciona o trabalho como complementar | nota "Bibliografia: Needs improvement" dos revisores 1 e 2 |
| `7ed8fc4` | — | Sincroniza `docs/paper/` com `docs/overleaf/` (estavam divergindo) e corrige typo | — |

`main.py` dessa branch já tem grade via `itertools.product` sobre todos os eixos (datasets, seeds, latent_dims, latent_bits, channels, snr_train, snr_test, weight_snr, betas), deduplicação por `run_id`, resumibilidade (pula o que já existe em `results/runs/`) e `--export-only` (só reconstrói CSV/figuras sem rodar nada).

**Nota técnica**: o `federated.py` dessa branch ainda tem as barras `tqdm` aninhadas ("Batches"/"Clients") que causavam spam de log — isso foi corrigido na `main` depois, então precisa ser reaplicado por cima ao mesclar.

## Comparação com o que existe hoje na `main`

| Aspecto | `main` (atual) | `melhorias-sbrt2026` |
|---|---|---|
| Ruído no latente | σ absoluto (AWGN simples) | Potência normalizada + SNR em dB |
| Canal com desvanecimento | Multiplicativo direto, sem equalização | Complexo + equalização CSI |
| Contagem de bits | 32 bits em ambos os lados | 8 bits raw + quantização do latente |
| Ruído nos pesos do FedAvg | Não implementado | `perturb_state_dict`, SNR por tensor |
| Non-IID Dirichlet | Sem rejeição de clientes degenerados | Com rejeição + heatmap |
| Persistência de resultados | Append incremental em CSV | Hash-based, resumível, CSV reconstruído |
| Estrutura de código | Pacote `semantic_federated/` + testes pytest | Arquivos soltos na raiz |
| Verbosidade de log | Corrigida | Ainda tem o spam de barras aninhadas |
| Notebooks Colab | 2 notebooks prontos | Nenhum |

## Como mesclar quando for a hora

Trazer a lógica de canal/bits/persistência da `melhorias-sbrt2026` para dentro do pacote `semantic_federated/` da `main`, mantendo a estrutura de pacote, os testes, os notebooks Colab e o fix de verbosidade de log:

1. Portar `channel.py` → `semantic_federated/noise.py`
2. Portar `comm_cost.py` → `semantic_federated/compression.py`
3. Criar `semantic_federated/device.py` (novo)
4. Portar `data.py` (Dirichlet robusto) → `semantic_federated/data.py`
5. Portar `federated.py` (ruído nos pesos + métricas ponderadas) → `semantic_federated/federated.py`, **reaplicando o fix de verbosidade por cima**
6. Portar `save_results.py` (runs resumíveis) → `semantic_federated/reporting/save_results.py`
7. Reescrever `main.py` com a grade via `itertools.product` + resumibilidade + `--export-only`
8. Atualizar `semantic_federated/training/{baseline,compressed}.py` para a nova API de canal/quantização
9. Estender `tables.py` e `plot_results.py` com as novas colunas/gráficos (`accuracy_vs_snr`, `snr_mismatch`), mantendo a paleta validada (skill `dataviz`)
10. Reescrever a suíte de testes para a nova lógica
11. Atualizar os 2 notebooks Colab para a nova CLI (SNR em dB em vez de `--noise-levels`, `--device`, `--export-only`)
12. Cherry-pick da Fase 6 (bibliografia) e da sincronização de docs, verificando conflito com o que mudou na `main` depois do ponto de divergência (`720e543`)
13. Rodar a suíte de testes completa + smoke test local antes de rodar no Colab

**Mudança de CLI que isso implica** (quebra os notebooks atuais): `--noise-levels` (σ) vira `--snr-train-db`/`--snr-test-db` (dB); `--rician-k`/`--fading-scale` viram `--rician-k-db`; `--partition dirichlet --dirichlet-alpha` vira `--betas`.

**Notebook "legado"**: depois da mesclagem, reproduzir os números exatos do paper (ex. L=64 ~0,6509) fica semanticamente diferente — o σ=0,05 original não equivale a nenhum SNR em dB específico sem remedir (~32 dB segundo a branch). Precisa decidir como documentar isso.
