# 📡 Comunicação Semântica e Aprendizado Federado na Borda

> **Uma Arquitetura baseada em Autoencoders para Redes 6G**

Este repositório contém a implementação completa do testbed experimental descrito no artigo acadêmico *"Comunicação Semântica e Aprendizado Federado na Borda: Uma Arquitetura baseada em Autoencoders para Redes 6G"*, submetido em formato IEEE.

O projeto demonstra que **autoencoders convolucionais leves**, combinados com **Aprendizado Federado (FedAvg)**, conseguem comprimir a informação semântica de imagens em até **192x** (economia de 99,48% de banda) mantendo acurácia de classificação competitiva — provando que a maioria dos bits em transmissões convencionais é redundância sem valor semântico.

## 🚀 Rodar no Google Colab

Sem precisar instalar nada localmente — o notebook clona o repositório, monta o Google Drive para persistir dados e resultados, e já usa GPU automaticamente quando disponível.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GianSE/semantic-federeted/blob/main/notebooks/run_experiments_colab.ipynb)

### Reprodução dos resultados legados (paper submetido)

Para reproduzir a mesma configuração de experimento do artigo submetido/aceito na SBrT 2026 — seed única (42), sem non-IID, sem canais com desvanecimento — mas já se beneficiando de GPU e salvamento incremental do pipeline atual, use este notebook. Ele roda o código atual (não uma versão antiga fixada), só restringindo o escopo do experimento para bater com o que foi reportado no paper.

[![Abrir no Colab (legado)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GianSE/semantic-federeted/blob/main/notebooks/run_experiments_colab_legacy.ipynb)

### Melhorias futuras já prontas numa branch separada

A branch `origin/melhorias-sbrt2026` implementa, sem estar mesclada ainda, a seção de "Limitações e Trabalhos Futuros" do paper (SNR em dB, quantização do latente, canal com desvanecimento e equalização CSI, ruído nos pesos do FedAvg, non-IID mais robusto, bibliografia). Ver [`melhorias-futuras.md`](melhorias-futuras.md) para o detalhamento completo e o plano de mesclagem.

---

## 📂 Estrutura do Projeto

```
semantic-federeted/
│
├── main.py                       # 🚀 Ponto de entrada — orquestra todos os experimentos
│
├── semantic_federated/           # 📦 Pacote principal
│   ├── data.py                   #   Carregamento e particionamento federado dos datasets
│   ├── federated.py              #   Motor de Aprendizado Federado (FedAvg)
│   ├── compression.py            #   Cálculos de taxa de compressão e custo de comunicação
│   ├── noise.py                  #   Injeção de ruído gaussiano (AWGN) e dropout
│   ├── metrics.py                #   Métricas de avaliação (acurácia, médias)
│   ├── models/
│   │   ├── autoencoder.py        #   🧠 Autoencoders (Encoder + Decoder) para MNIST e CIFAR-10
│   │   └── classifier.py         #   🎯 Classificadores (Raw e Latente)
│   ├── training/
│   │   ├── baseline.py           #   📊 Treinamento federado do baseline (sem compressão)
│   │   └── compressed.py         #   🔬 Treinamento federado com compressão semântica
│   └── reporting/
│       ├── save_results.py       #   💾 Persistência de resultados (CSV + JSON, acumulativo)
│       ├── plot_results.py       #   📊 Geração de gráficos acadêmicos (estilo IEEE)
│       └── tables.py             #   📋 Geração de tabelas LaTeX para o artigo
│
├── scripts/
│   └── gera_exemplo_real.py      # 🖼️ Gera mosaico visual (Original → Embedding → Reconstrução)
│
├── tests/                        # 🧪 Testes unitários
├── notebooks/
│   └── run_experiments_colab.ipynb  # ▶️ Notebook para rodar no Google Colab (GPU)
├── requirements.txt              # 📦 Dependências Python
│
├── results/                 # Resultados gerados pelos experimentos
│   ├── data/                #   ├── experiment_results.csv / .json
│   ├── plots/                #   ├── Gráficos PNG (accuracy, noise, compression)
│   └── tables/              #   └── Tabelas CSV e LaTeX
│
├── data/                    # Datasets baixados automaticamente (MNIST, CIFAR-10)
│
└── docs/
    └── overleaf/            # Artigo LaTeX completo (formato IEEE)
        ├── main.tex
        ├── acronym.tex
        ├── ref.bib
        └── figures/
```

---

## 🏗️ Arquitetura do Sistema

O pipeline de comunicação semântica federada funciona em 3 fases:

```
┌────────────────────────────────────────────────────────────────┐
│                  DISPOSITIVO DE BORDA (Cliente)                │
│                                                                │
│   Imagem ──► Encoder (CNN) ──► Espaço Latente z ──┬──► Classificador ──► Decisão
│              (3 blocos conv)     (L dimensões)    │   (Task-Oriented)
│                                      │            │
│                                  [+ Ruído σ]      └──► Decoder ──► Reconstrução
│                                      │
│                                      ▼
│                              Vetor Latente z̃
└──────────────────────────────┬───────────────────────┘
                               │  Pesos Locais (upload)
                               ▼
                    ┌─────────────────────┐
                    │   SERVIDOR GLOBAL   │
                    │  Agregação FedAvg   │   w_{t+1} = Σ (n_k/n) * w_t^k
                    └─────────┬───────────┘
                              │  Modelo Global (download)
                              ▼
                    [Próxima Rodada Federada]
```

### Perda Multitarefa

O modelo é treinado com uma perda combinada:

```
L = L_CE(Classificador(z̃), y) + α · L_MSE(Decoder(z̃), x)
```

Onde `α = 0.5` balanceia classificação e reconstrução.

---

## 🚀 Como Rodar

### 1. Pré-requisitos

- **Python 3.8+**
- **pip** (gerenciador de pacotes)

### 2. Instalar dependências

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar o ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

As dependências são:
| Pacote         | Função                              |
|----------------|-------------------------------------|
| `torch`        | Framework de deep learning          |
| `torchvision`  | Datasets (MNIST, CIFAR-10) e transforms |
| `numpy`        | Operações numéricas                 |
| `pandas`       | Manipulação de dados tabulares      |
| `matplotlib`   | Geração de gráficos acadêmicos     |
| `scikit-learn` | Métricas auxiliares                 |
| `tqdm`         | Barras de progresso                 |

### 3. Executar os experimentos completos

```bash
python main.py
```

Este comando executa **todos** os experimentos com os hiperparâmetros padrão do artigo:

| Parâmetro             | Valor Padrão              |
|-----------------------|---------------------------|
| Datasets              | `mnist`, `cifar10`        |
| Dimensões Latentes    | `16, 32, 64, 128`        |
| Níveis de Ruído (σ)   | `0.0, 0.01, 0.05, 0.1`   |
| Clientes Federados    | `5`                       |
| Rodadas Federadas     | `3`                       |
| Épocas Locais         | `1`                       |
| Learning Rate         | `0.001`                   |
| Batch Size            | `64`                      |
| Alpha (α)             | `0.5`                     |
| Seeds                 | `42`                      |
| Partição              | `iid`                     |
| Canal                 | `awgn`                    |

### 4. Personalizar experimentos

Todos os hiperparâmetros podem ser ajustados via linha de comando:

```bash
# Rodar apenas CIFAR-10 com dimensões latentes específicas
python main.py --datasets cifar10 --latent-dims 16 32 64

# Testar com mais rodadas federadas e mais clientes
python main.py --datasets cifar10 --rounds 5 --num-clients 10

# Variar apenas o ruído para L=64
python main.py --datasets cifar10 --latent-dims 64 --noise-levels 0.0 0.01 0.05 0.1 0.2

# Definir um orçamento fixo de comunicação (em bits)
python main.py --fixed-comm-budget 100000000

# Múltiplas seeds (recomendado para o paper: reporta média +/- desvio padrão em results_summary.csv)
python main.py --datasets cifar10 --latent-dims 64 --noise-levels 0.0 0.05 --seeds 1 2 3 4 5

# Partição não-IID via Dirichlet (alpha baixo = mais heterogêneo entre clientes)
python main.py --datasets cifar10 --partition dirichlet --dirichlet-alpha 0.1

# Canal com desvanecimento (além do AWGN padrão)
python main.py --datasets cifar10 --channel-type rayleigh --fading-scale 1.0
python main.py --datasets cifar10 --channel-type rician --fading-scale 1.0 --rician-k 5.0
```

### 5. Scripts individuais

Cada componente pode ser executado separadamente (sempre a partir da raiz do repositório, usando `-m` porque agora são módulos do pacote `semantic_federated`):

```bash
# Apenas o baseline (classificador sem compressão)
python -m semantic_federated.training.baseline --dataset cifar10 --rounds 3

# Apenas o modelo comprimido (autoencoder + classificador latente)
python -m semantic_federated.training.compressed --dataset cifar10 --latent-dim 64 --noise-level 0.05

# Regenerar agregação multi-seed (necessário antes dos gráficos, gera results_summary.csv)
python -m semantic_federated.reporting.aggregate

# Regenerar gráficos a partir do results_summary.csv
python -m semantic_federated.reporting.plot_results

# Regenerar tabelas LaTeX (a partir do CSV bruto, uma linha por execução)
python -m semantic_federated.reporting.tables

# Gerar mosaico visual (Original → Embedding → Reconstrução)
python -m scripts.gera_exemplo_real
```

### 6. Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Cobrem a aritmética de compressão/bits (`compression.py`), a agregação ponderada do FedAvg (Eq. 3 do paper, em `federated.py`), o ruído gaussiano/dropout e os canais com desvanecimento (`noise.py`), as métricas (`metrics.py`), o particionamento Dirichlet (`data.py`), a agregação multi-seed (`reporting/aggregate.py`), a geração de gráficos (`reporting/plot_results.py`) e o acúmulo incremental de resultados (`reporting/save_results.py`).

---

## 📊 Resultados e Saídas

Após a execução, o diretório `results/` conterá:

### `results/data/`
- **`experiment_results.csv`** — Tabela com todas as métricas (acurácia, compressão, custo)
- **`experiment_results.json`** — Mesmos dados em formato JSON

### `results/plots/`
Todos os gráficos usam o `results_summary.csv` (média ± desvio padrão entre seeds) e uma paleta categórica fixa, validada contra daltonismo (ver `dataviz` skill): a condição padrão (`iid`, `awgn`) é sempre azul; alternativas (`dirichlet`, `rayleigh`, `rician`) têm cor fixa própria em qualquer gráfico onde apareçam.

| Arquivo                              | Descrição                                           |
|--------------------------------------|-----------------------------------------------------|
| `accuracy_vs_compression_ratio.png`  | Trade-off entre compressão e acurácia, com barra de erro e linha de referência do baseline |
| `accuracy_vs_latent_dim.png`         | Acurácia em função da dimensão latente, com ajuste logarítmico ($R^2$ anotado) |
| `accuracy_vs_noise_level.png`        | Impacto do ruído na acurácia, por dimensão latente (barras de erro) |
| `communication_cost_vs_latent_dim.png` | Custo de comunicação vs dimensão latente          |
| `accuracy_iid_vs_dirichlet.png`      | **Novo.** IID vs. não-IID (Dirichlet) — só gerado se você rodou `--partition dirichlet` |
| `accuracy_by_channel_type.png`       | **Novo.** AWGN vs. Rayleigh vs. Rician — só gerado se você rodou `--channel-type rayleigh`/`rician` |

### `results/tables/`
- **`results_table.csv`** — Tabela formatada em CSV
- **`results_table.tex`** — Tabela formatada em LaTeX (pronta para o artigo)
- **`results_summary.csv`** — Agregação por configuração (dataset, latent_dim, noise_level, channel_type, partition) com média, desvio padrão e contagem de seeds para cada métrica. Use esta tabela quando rodar `--seeds` com múltiplos valores — é o que permite reportar "acurácia X ± Y" em vez de um único número de uma seed só.

> **Nota:** Os resultados são **acumulativos**. Cada nova execução do `main.py` **adiciona** os novos dados aos arquivos existentes, permitindo rodar diferentes configurações iterativamente.

---

## 🔬 Como Reproduzir e Provar a Pesquisa

### Hipótese 1: Compressão Semântica Extrema sem Perda Significativa de Acurácia

**Afirmação do artigo:** A arquitetura atinge economia de 97,9% de tráfego com apenas 2,6 pontos percentuais de penalidade.

**Como verificar:**
```bash
# Executar o baseline e a compressão com L=64
python main.py --datasets cifar10 --latent-dims 64 --noise-levels 0.0

# Verificar os resultados
python -c "import pandas as pd; df = pd.read_csv('results/data/experiment_results.csv'); print(df[['dataset','latent_dim','accuracy_baseline','accuracy_compressed','compression_ratio','communication_cost_bits']].to_string())"
```

**Resultado esperado:**
- Baseline: ~0.677 acurácia, 4.92×10⁹ bits
- L=64: ~0.649 acurácia, 1.02×10⁸ bits (CR = 48x)
- L=16: ~0.607 acurácia, 2.56×10⁷ bits (CR = 192x)

### Hipótese 2: Ruído Gaussiano como Regularizador

**Afirmação do artigo:** Ruído moderado (σ=0.05) **melhora** a acurácia em relação ao cenário sem ruído.

**Como verificar:**
```bash
# Executar com diferentes níveis de ruído
python main.py --datasets cifar10 --latent-dims 64 --noise-levels 0.0 0.05 0.1

# Comparar os resultados
python -c "
import pandas as pd
df = pd.read_csv('results/data/experiment_results.csv')
df = df[(df['dataset']=='cifar10') & (df['latent_dim']==64)]
print(df[['noise_level','accuracy_compressed']].to_string())
"
```

**Resultado esperado:**
| Ruído (σ) | Acurácia   |
|-----------|------------|
| 0.00      | ~0.6488    |
| 0.05      | **~0.6509** ← maior que sem ruído! |
| 0.10      | ~0.6492    |

Isto comprova a teoria do **Information Bottleneck**: o ruído impede overfitting e força o encoder a aprender representações semânticas mais robustas.

### Hipótese 3: Queda Logarítmica de Acurácia com a Compressão

**Afirmação:** A acurácia cai de forma logarítmica (não linear) conforme a dimensão latente diminui.

**Como verificar:**
```bash
python main.py --datasets cifar10 --latent-dims 16 32 64 128 256 --noise-levels 0.0
# main.py já gera os gráficos ao final; para regenerar sem rodar tudo de novo:
python -m semantic_federated.reporting.aggregate
python -m semantic_federated.reporting.plot_results
# Abrir results/plots/accuracy_vs_latent_dim.png (inclui ajuste logarítmico com R²)
```

### Validação Cruzada Completa

Para uma validação completa com todas as combinações:
```bash
# Limpar resultados anteriores (opcional)
del results\data\experiment_results.csv
del results\data\experiment_results.json

# Execução completa
python main.py --datasets cifar10 --latent-dims 16 32 64 128 --noise-levels 0.0 0.01 0.05 0.1 --rounds 3 --seeds 42
```

---

## 🧩 Descrição dos Módulos

### `main.py` — Orquestrador de Experimentos
Coordena todo o pipeline: itera sobre datasets, dimensões latentes e níveis de ruído. Executa o baseline e todos os cenários comprimidos, salva resultados e gera gráficos/tabelas automaticamente.

### `semantic_federated/data.py` — Carregamento e Particionamento Federado
Carrega MNIST ou CIFAR-10 via `torchvision` e particiona os dados de treino em `N` clientes. Duas estratégias de particionamento:
- **`iid`** (padrão): split aleatório homogêneo (`split_clients`).
- **`dirichlet`**: não-IID por rótulo via distribuição de Dirichlet (`split_clients_dirichlet`) — `--dirichlet-alpha` baixo (ex. 0.1) concentra cada classe em poucos clientes; alto (ex. 100) se aproxima do IID.

Aplica normalização padrão por dataset.

### `semantic_federated/models/autoencoder.py` — Codificadores Semânticos
Define dois autoencoders convolucionais:
- **MNISTAutoencoder**: 2 blocos conv (16→32 filtros) para imagens 1×28×28
- **CIFAR10Autoencoder**: 3 blocos conv (32→64→128 filtros) para imagens 3×32×32

O encoder mapeia a entrada para um vetor latente `z ∈ ℝ^L`. O decoder espelha a estrutura com convoluções transpostas.

### `semantic_federated/models/classifier.py` — Classificadores
Três variantes:
- **RawMNISTClassifier / RawCIFAR10Classifier**: Classificadores CNN para o baseline (sem compressão)
- **LatentClassifier**: Rede densa (Linear→ReLU→Linear) que classifica diretamente a partir do vetor latente

### `semantic_federated/training/compressed.py` — Pipeline Comprimido
Combina Autoencoder + LatentClassifier em um `CompressedModel` que:
1. Codifica a imagem → vetor latente `z`
2. Injeta ruído AWGN → `z̃ = z + N(0, σ²)`
3. Classifica a partir de `z̃`
4. Reconstrói a imagem a partir de `z`
5. Otimiza com perda multitarefa: `L = L_CE + α·L_MSE`

### `semantic_federated/training/baseline.py` — Pipeline Baseline
Treina um classificador CNN padrão via FedAvg **sem compressão**. Serve como linha de base para comparação de acurácia e custo de comunicação.

### `semantic_federated/federated.py` — Motor FedAvg
Implementa o algoritmo Federated Averaging:
1. Cada cliente recebe o modelo global
2. Treina localmente por `E` épocas
3. Envia os pesos atualizados ao servidor
4. O servidor calcula a média ponderada: `w_{t+1} = Σ(n_k/n)·w_t^k`

### `semantic_federated/compression.py` — Métricas de Comunicação
Calcula o custo em bits para cada cenário:
- **Raw**: `pixels × canais × 32 bits` por amostra
- **Latente**: `L × 32 bits` por amostra
- **Razão de compressão**: `bits_raw / bits_latente`

### `semantic_federated/noise.py` — Simulação de Canal
Simula imperfeições do canal sem fio, selecionáveis via `--channel-type`:
- **`awgn`** (padrão): `z̃ = z + N(0, σ²)` — ruído aditivo gaussiano puro
- **`rayleigh`**: `z̃ = h·z + N(0, σ²)`, com `h` seguindo distribuição Rayleigh (desvanecimento sem linha de visada) — controlado por `--fading-scale`
- **`rician`**: como o Rayleigh, mas com componente de linha de visada (`--rician-k`); `k` alto → canal quase determinístico, `k` próximo de 0 → se aproxima do Rayleigh puro
- **Dropout**: Zera aleatoriamente dimensões do vetor latente (independente do modelo de canal)

### `semantic_federated/reporting/plot_results.py` — Visualização Acadêmica
Gera até 6 gráficos em estilo IEEE (fonte serif, DPI 300) a partir do `results_summary.csv` agregado, com barras de erro entre seeds, ajuste logarítmico (com $R^2$) na curva de acurácia vs. dimensão latente, e comparações IID vs. não-IID / tipos de canal quando os dados existirem.

### `semantic_federated/reporting/tables.py` — Tabelas para o Artigo
Exporta os resultados como tabela LaTeX formatada, pronta para inclusão no `main.tex`.

### `semantic_federated/reporting/aggregate.py` — Agregação Multi-Seed
Agrupa por configuração (dataset, latent_dim, noise_level, channel_type, partition) e calcula média, desvio padrão e contagem de seeds por métrica, salvando em `results/tables/results_summary.csv`.

### `scripts/gera_exemplo_real.py` — Mosaico Visual
Gera uma figura demonstrativa com 3 painéis:
1. Imagem original do CIFAR-10
2. Vetor de embedding (gráfico de barras)
3. Imagem reconstruída pelo decoder

---

## 📄 Artigo Acadêmico

O artigo completo em formato IEEE está em `docs/overleaf/`. Para compilar:

```bash
cd docs/overleaf
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Para enviar ao Overleaf, faça upload dos seguintes arquivos:
- `main.tex`, `acronym.tex`, `ref.bib`
- Pasta `figures/` com: `mosaico_real.png`, `accuracy_vs_noise_level.png`, `accuracy_vs_latent_dim.png`, `results_table.tex`

---

## 📬 Autores

- **Gian Pedro Rodrigues** — gian.2000@alunos.utfpr.edu.br
- **Herman L. dos Santos** — hermansantos@utfpr.edu.br

Departamento Acadêmico de Engenharia Elétrica — Universidade Tecnológica Federal do Paraná (UTFPR), Cornélio Procópio, Brasil.

---

## 📜 Licença

Este projeto é parte de uma pesquisa acadêmica da UTFPR. Para uso ou citação, entre em contato com os autores.
