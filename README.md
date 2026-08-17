# 📡 Comunicação Semântica e Aprendizado Federado na Borda

> **Uma Arquitetura baseada em Autoencoders para Redes 6G**

Este repositório contém a implementação completa do testbed experimental descrito no artigo acadêmico *"Comunicação Semântica e Aprendizado Federado na Borda: Uma Arquitetura baseada em Autoencoders para Redes 6G"*, submetido em formato IEEE.

O projeto demonstra que **autoencoders convolucionais leves**, combinados com **Aprendizado Federado (FedAvg)**, conseguem comprimir a informação semântica de imagens em até **192x** (economia de 99,48% de banda) mantendo acurácia de classificação competitiva — provando que a maioria dos bits em transmissões convencionais é redundância sem valor semântico.

> A razão de compressão compara o payload de inferência contra a imagem bruta em
> **uint8** (8 bits/pixel, o formato nativo de armazenamento). Os 192x
> pressupõem latente quantizado em **8 bits/dimensão** com `L=16`
> (`--latent-dims 16 --latent-bits 8`). Com latente em float32 a razão cai para
> 48x. Ver `comm_cost.py`.

---

## 📂 Estrutura do Projeto

```
semantic-federeted/
│
├── main.py                  # 🚀 Ponto de entrada — orquestra todos os experimentos
├── data.py                  # 📦 Carregamento e particionamento federado dos datasets
├── model_autoencoder.py     # 🧠 Autoencoders (Encoder + Decoder) para MNIST e CIFAR-10
├── model_classifier.py      # 🎯 Classificadores (Raw e Latente)
├── train_baseline.py        # 📊 Treinamento federado do baseline (sem compressão)
├── train_compressed.py      # 🔬 Treinamento federado com compressão semântica
├── federated.py             # 🔄 Motor de Aprendizado Federado (FedAvg)
├── comm_cost.py             # 📐 Cálculos de taxa de compressão e custo de comunicação
├── channel.py               # 📶 Modelo de canal: normalização de potência + AWGN (SNR em dB)
├── device.py                # 💻 Seleção de dispositivo (CPU/CUDA)
├── metrics.py               # 📈 Métricas de avaliação (acurácia, médias)
├── save_results.py          # 💾 Persistência por run (JSON) + exportação de CSVs
├── plot_results.py          # 📊 Geração de gráficos acadêmicos (estilo IEEE)
├── tables.py                # 📋 Geração de tabelas LaTeX para o artigo
├── gera_exemplo_real.py     # 🖼️ Gera mosaico visual (Original → Embedding → Reconstrução)
├── requirements.txt         # 📦 Dependências Python
│
├── results/                 # Resultados gerados pelos experimentos
│   ├── runs/                #   ├── Um JSON por configuração (execuções retomáveis)
│   ├── data/                #   ├── experiment_results.csv / history.csv
│   ├── plots/               #   ├── Gráficos PNG (accuracy, noise, compression)
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
│                            [normaliza + canal AWGN]  └──► Decoder ──► Reconstrução
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

### Duas contabilidades de comunicação

O sistema tem **dois** custos de comunicação distintos, reportados em colunas
separadas (`comm_cost.py`) porque respondem a perguntas diferentes:

| | O que mede | Coluna |
|---|---|---|
| **Inferência** | payload semântico por amostra transmitida | `inference_bits_per_sample`, `compression_ratio` |
| **Treinamento** | pesos trocados no FedAvg (uplink + downlink) | `training_bits_total`, `model_params` |

A compressão semântica atua sobre a **primeira**; a segunda depende apenas do
tamanho do modelo e do número de rodadas/clientes. Somar as duas, ou reportar
apenas uma como se fosse o custo total, superestima o ganho.

Para CIFAR-10, o modelo semântico também é **menor** que o baseline até `L=64`
(0,33x em `L=16`; 0,66x em `L=64`), porque o baseline carrega uma camada densa
`Linear(2048, 256)`. Só em `L=128` ele passa o baseline (1,09x).

---

## 🚀 Como Rodar

### 1. Pré-requisitos

- **Python 3.10+** (testado em 3.14)
- **pip** (gerenciador de pacotes)
- GPU **opcional** — todo o pipeline roda em CPU (ver perfis de execução abaixo)

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
| `tqdm`         | Barras de progresso                 |

As versões estão **fixadas** no `requirements.txt`. Para GPU, instale
`torch`/`torchvision` pelo índice CUDA correspondente (ver comentário no arquivo).

### 2.1. CPU ou GPU

Todos os scripts aceitam `--device`:

```bash
python main.py --device auto   # padrão: usa CUDA se disponível, senão CPU
python main.py --device cpu    # força CPU
python main.py --device cuda   # força GPU (erro explícito se indisponível)
```

Com GPU, `--num-workers 4` costuma ser o ganho maior (o padrão `0` é gargalo).

> Resultados **não** são bit-idênticos entre CPU e CUDA. O dispositivo usado fica
> registrado em cada run; fixe um único dispositivo para os números do artigo.

### 2.2. Perfis de execução

| Perfil | Comando | Custo aprox. (CPU) | Uso |
|--------|---------|--------------------|-----|
| `smoke` | `python main.py --datasets mnist --latent-dims 16 --snr-train-db none --num-clients 2 --rounds 2 --train-fraction 0.02` | ~15 s | validar o pipeline |
| `dev` | `python main.py --datasets cifar10 --latent-dims 16 64 --snr-train-db none 10 --num-clients 5 --rounds 5` | ~1–2 h | verificar tendências |
| `paper` | grade completa (ver seção de reprodução) | GPU recomendada | números finais |

`--train-fraction` subamostra o conjunto de treino, permitindo rodadas rápidas
sem alterar a estrutura do experimento.

### 2.3. Execuções retomáveis

Cada configuração vira um arquivo em `results/runs/<hash>.json`, nomeado por um
hash determinístico da configuração. Consequências práticas:

- Rodar o mesmo comando duas vezes **não refaz nada** e produz CSVs idênticos.
- Uma grade longa pode ser acumulada em várias sessões — ou migrada para GPU sem
  refazer o que já rodou (o dispositivo não entra no hash).
- `--force` refaz runs existentes; `--export-only` só reconstrói CSVs e figuras.

Os CSVs agregados são sempre **reconstruídos** a partir dos runs em disco, nunca
por append.

### 3. Executar os experimentos completos

```bash
python main.py
```

Este comando executa **todos** os experimentos com os hiperparâmetros padrão do artigo:

| Parâmetro             | Valor Padrão              |
|-----------------------|---------------------------|
| Datasets              | `mnist`, `cifar10`        |
| Dimensões Latentes    | `16, 32, 64, 128`        |
| SNR de Treino (dB)    | `none, 20, 10, 5, 0, -5, -10` |
| Clientes Federados    | `5`                       |
| Rodadas Federadas     | `3`                       |
| Épocas Locais         | `1`                       |
| Learning Rate         | `0.001`                   |
| Batch Size            | `64`                      |
| Alpha (α)             | `0.5`                     |
| Seed                  | `42`                      |

### 4. Personalizar experimentos

Todos os hiperparâmetros podem ser ajustados via linha de comando:

```bash
# Rodar apenas CIFAR-10 com dimensões latentes específicas
python main.py --datasets cifar10 --latent-dims 16 32 64

# Testar com mais rodadas federadas e mais clientes
python main.py --datasets cifar10 --rounds 5 --num-clients 10

# Variar apenas o ruído para L=64
python main.py --datasets cifar10 --latent-dims 64 --snr-train-db 20 10 5 0 -5 -10

# Múltiplas seeds (para média e desvio-padrão nas figuras)
python main.py --datasets cifar10 --seeds 42 43 44
```

### 5. Scripts individuais

Cada componente pode ser executado separadamente:

```bash
# Apenas o baseline (classificador sem compressão)
python train_baseline.py --dataset cifar10 --rounds 3

# Apenas o modelo comprimido (autoencoder + classificador latente)
python train_compressed.py --dataset cifar10 --latent-dim 64 --snr-train-db 10

# Regenerar gráficos a partir dos resultados existentes
python plot_results.py

# Regenerar tabelas LaTeX
python tables.py

# Gerar mosaico visual (Original → Embedding → Reconstrução)
python gera_exemplo_real.py
```

---

## 📊 Resultados e Saídas

Após a execução, o diretório `results/` conterá:

### `results/data/`
- **`experiment_results.csv`** — Tabela com todas as métricas (acurácia, compressão, custo)
- **`experiment_results.json`** — Mesmos dados em formato JSON

### `results/plots/`
| Arquivo                              | Descrição                                           |
|--------------------------------------|-----------------------------------------------------|
| `accuracy_vs_compression_ratio.png`  | Trade-off entre compressão e acurácia               |
| `accuracy_vs_latent_dim.png`         | Acurácia em função da dimensão latente              |
| `accuracy_vs_snr.png`                | Acurácia vs. SNR do canal (por L)                   |
| `snr_mismatch_<ds>_L<n>.png`         | Matriz SNR de treino × SNR de teste                 |
| `accuracy_vs_round_<ds>.png`         | Convergência federada (acurácia por rodada)         |
| `communication_cost_vs_latent_dim.png` | Custo de comunicação vs dimensão latente          |

### `results/tables/`
- **`results_table.csv`** — Tabela formatada em CSV
- **`results_table.tex`** — Tabela formatada em LaTeX (pronta para o artigo)

> **Nota:** Os resultados são **acumulativos**. Cada nova execução do `main.py` **adiciona** os novos dados aos arquivos existentes, permitindo rodar diferentes configurações iterativamente.

---

## 🔬 Como Reproduzir e Provar a Pesquisa

### Hipótese 1: Compressão Semântica Extrema sem Perda Significativa de Acurácia

**Afirmação do artigo:** A arquitetura atinge economia de 97,9% de tráfego com apenas 2,6 pontos percentuais de penalidade.

**Como verificar:**
```bash
# Executar o baseline e a compressão com L=64
python main.py --datasets cifar10 --latent-dims 64 --snr-train-db none

# Verificar os resultados
python -c "import pandas as pd; df = pd.read_csv('results/data/experiment_results.csv'); print(df[['dataset','latent_dim','accuracy_baseline','accuracy_compressed','compression_ratio','communication_cost_bits']].to_string())"
```

**Payload de inferência por amostra (CIFAR-10, bruto = 24.576 bits em uint8):**

| L   | 32 bits/dim | 8 bits/dim | 4 bits/dim |
|-----|-------------|------------|------------|
| 16  | 48x (97,92%) | **192x (99,48%)** | 384x (99,74%) |
| 32  | 24x (95,83%) | 96x (98,96%) | 192x (99,48%) |
| 64  | 12x (91,67%) | **48x (97,92%)** | 96x (98,96%) |
| 128 | 6x (83,33%)  | 24x (95,83%) | 48x (97,92%) |

Os dois números do artigo (97,9% para `L=64` e 99,48% para `L=16`) são exatamente
recuperados com quantização de 8 bits por dimensão.

### Hipótese 2: Ruído Gaussiano como Regularizador

**Afirmação do artigo:** Ruído moderado (σ=0.05) **melhora** a acurácia em relação ao cenário sem ruído.

> ⚠️ **Em reverificação.** Esta afirmação foi obtida com a parametrização
> anterior, na qual σ era uma amplitude absoluta aplicada a um latente de escala
> livre — condição em que σ não define uma condição de canal. Medindo a potência
> do latente após o treino naquela formulação (E[|z_i|²] ≈ 3,96 para L=64),
> σ=0,05 correspondia a uma SNR efetiva de **≈ 32 dB**, isto é, um canal muito
> favorável. Além disso, a diferença reportada (0,6509 vs. 0,6488) é de 2
> milésimos, medida com uma única seed. A reverificação exige múltiplas seeds e
> a nova parametrização por SNR.

**Como verificar:**
```bash
# Executar com diferentes níveis de ruído
python main.py --datasets cifar10 --latent-dims 64 --snr-train-db none 26 20

# Comparar os resultados
python -c "
import pandas as pd
df = pd.read_csv('results/data/experiment_results.csv')
df = df[(df['dataset']=='cifar10') & (df['latent_dim']==64)]
print(df[['snr_train_db','accuracy_compressed']].to_string())
"
```

**O que verificar:** se a acurácia em SNR alta (canal levemente ruidoso) supera a
do canal ideal, com a diferença maior que o desvio-padrão entre seeds. Use
`--seeds 42 43 44` — as figuras já desenham a faixa de ±1 desvio-padrão.

A hipótese subjacente é a do **Information Bottleneck**: o ruído impediria o
overfitting, forçando o encoder a representações semânticas mais robustas. O
efeito só pode ser afirmado se sobreviver à variabilidade entre seeds.

### Hipótese 3: Queda Logarítmica de Acurácia com a Compressão

**Afirmação:** A acurácia cai de forma logarítmica (não linear) conforme a dimensão latente diminui.

**Como verificar:**
```bash
python main.py --datasets cifar10 --latent-dims 16 32 64 128 256 --snr-train-db none
python plot_results.py
# Abrir results/plots/accuracy_vs_latent_dim.png
```

### Validação Cruzada Completa

Para uma validação completa com todas as combinações:
```bash
# Limpar resultados anteriores (opcional)
del results\data\experiment_results.csv
del results\data\experiment_results.json

# Execução completa
python main.py --datasets cifar10 --latent-dims 16 32 64 128 --snr-train-db none 20 10 0 --rounds 3 --seeds 42
```

---

## 🧩 Descrição dos Módulos

### `main.py` — Orquestrador de Experimentos
Coordena todo o pipeline: itera sobre datasets, dimensões latentes e níveis de ruído. Executa o baseline e todos os cenários comprimidos, salva resultados e gera gráficos/tabelas automaticamente.

### `data.py` — Carregamento e Particionamento Federado
Carrega MNIST ou CIFAR-10 via `torchvision` e particiona os dados de treino em `N` splits IID (distribuição homogênea) para simular clientes federados. Aplica normalização padrão por dataset.

### `model_autoencoder.py` — Codificadores Semânticos
Define dois autoencoders convolucionais:
- **MNISTAutoencoder**: 2 blocos conv (16→32 filtros) para imagens 1×28×28
- **CIFAR10Autoencoder**: 3 blocos conv (32→64→128 filtros) para imagens 3×32×32

O encoder mapeia a entrada para um vetor latente `z ∈ ℝ^L`. O decoder espelha a estrutura com convoluções transpostas.

### `model_classifier.py` — Classificadores
Três variantes:
- **RawMNISTClassifier / RawCIFAR10Classifier**: Classificadores CNN para o baseline (sem compressão)
- **LatentClassifier**: Rede densa (Linear→ReLU→Linear) que classifica diretamente a partir do vetor latente

### `train_compressed.py` — Pipeline Comprimido
Combina Autoencoder + LatentClassifier em um `CompressedModel` que:
1. Codifica a imagem → vetor latente `z`
2. Normaliza a potência (E[|z_i|²]=1) e injeta AWGN → `z̃ = z + N(0, σ²)`, com `σ = 10^(-SNR_dB/20)`
3. Classifica a partir de `z̃`
4. Reconstrói a imagem a partir de `z`
5. Otimiza com perda multitarefa: `L = L_CE + α·L_MSE`

### `train_baseline.py` — Pipeline Baseline
Treina um classificador CNN padrão via FedAvg **sem compressão**. Serve como linha de base para comparação de acurácia e custo de comunicação.

### `federated.py` — Motor FedAvg
Implementa o algoritmo Federated Averaging:
1. Cada cliente recebe o modelo global
2. Treina localmente por `E` épocas
3. Envia os pesos atualizados ao servidor
4. O servidor calcula a média ponderada: `w_{t+1} = Σ(n_k/n)·w_t^k`

### `compression.py` — Métricas de Comunicação
Calcula o custo em bits para cada cenário:
- **Raw**: `pixels × canais × 32 bits` por amostra
- **Latente**: `L × 32 bits` por amostra
- **Razão de compressão**: `bits_raw / bits_latente`

### `noise.py` — Simulação de Canal
Simula imperfeições do canal sem fio:
- **Canal AWGN**: `z̃ = z + N(0, σ²)` sobre latente de potência unitária, com `σ = 10^(-SNR_dB/20)` — a condição de canal é definida pela SNR em dB, não por uma amplitude absoluta
- **Dropout**: Zera aleatoriamente dimensões do vetor latente

### `plot_results.py` — Visualização Acadêmica
Gera 4 gráficos em estilo IEEE (fonte serif, DPI 300) a partir do CSV de resultados.

### `tables.py` — Tabelas para o Artigo
Exporta os resultados como tabela LaTeX formatada, pronta para inclusão no `main.tex`.

### `gera_exemplo_real.py` — Mosaico Visual
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
- Pasta `figures/` com: `mosaico_real.png`, `accuracy_vs_snr.png`, `accuracy_vs_latent_dim.png`, `results_table.tex`

---

## 📬 Autores

- **Gian Pedro Rodrigues** — gian.2000@alunos.utfpr.edu.br
- **Herman L. dos Santos** — hermansantos@utfpr.edu.br

Departamento Acadêmico de Engenharia Elétrica — Universidade Tecnológica Federal do Paraná (UTFPR), Cornélio Procópio, Brasil.

---

## 📜 Licença

Este projeto é parte de uma pesquisa acadêmica da UTFPR. Para uso ou citação, entre em contato com os autores.
