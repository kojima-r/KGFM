# kgfm

大規模 (マルチ TB) TSV コーパス上でリレーション予測を行う、ストリーミング型の **K**nowledge-**g**raph **F**oundation **M**odel です。

`kgfm` は DistMult スタイルのスコアラを学習します。

```
score(h, r, t) = Σ_d  h_d · r_d · t_d
```

ここで `h`, `r`, `t` は差し替え可能なテキストエンコーダで生成される密ベクトルです。標準で 2 種類のエンコーダを同梱しています。

- **`ngram`** — 学習可能なハッシュ化文字 n-gram (FastText 風)。軽量・高速で PyTorch 以外の依存を持ちません。
- **`transformer`** — HuggingFace `AutoModel` 全般 (例: 多言語 BERT)。`--freeze-encoder` と `--proj-dim` を併用すれば、凍結 LM を特徴抽出器として使い、上に学習可能な射影ヘッドだけを載せた構成にできます。

データはファイル単位でストリーミング読み込みし、メモリ内シャッフルバッファを併用するため、巨大コーパスでも全体をマテリアライズせずに学習できます。

---

## インストール

```bash
# コア (ngram エンコーダのみ)
pip install -e .

# HuggingFace transformer サポートを追加
pip install -e ".[transformer]"

# レポートのグラフを強化 (matplotlib + plotly + umap-learn。
# 未導入でも内蔵 SVG / PCA で動作します)
pip install -e ".[report]"
```

PyTorch は CUDA / ドライバに依存します。`pip install torch` で適切な wheel が入らない場合は、<https://pytorch.org/get-started/locally/> から直接導入してください。

インストール後、`kgfm` コマンドが利用可能になります (ベンチマークのベースラインとして `kgfm-ultra` / `kgfm-motif` も入ります)。

```bash
kgfm train --help
kgfm eval  --help
```

モジュールとして直接呼ぶこともできます。

```bash
python -m kgfm train --help
python -m kgfm eval  --help
```

---

## データレイアウト

各 TSV 行は **6 列タブ区切り** で記述されている必要があります。

| 列 | 内容 |
| --- | --- |
| 1 | ノード 1 タイプ |
| 2 | 関係タイプ |
| 3 | ノード 2 タイプ |
| 4 | ノード 1 テキスト |
| 5 | 関係テキスト |
| 6 | ノード 2 テキスト |

デフォルトでは `data/**/latest/*.tsv` にマッチするファイルを再帰探索します。`--data-root` と `--pattern` で上書きできます。

### Train / valid / test 分割

明示的なファイルリストを与える方法と、ファイル名のハッシュで決定論的に自動分割する方法のどちらでも使えます。

**明示的リスト** — 1 行 1 TSV パス。空行と `#` 始まりのコメント行は無視されます。

```bash
kgfm train \
    --train-list list_small/train.txt \
    --valid-list list_small/valid.txt \
    --test-list  list_small/test.txt
```

**自動分割フォールバック** — リストが指定されない場合、`--data-root` 配下のファイルがハッシュバケットで分割されます。デフォルトは 80 / 10 / 10 です。

```bash
kgfm train --data-root data --valid-buckets 1 --test-buckets 1 --n-buckets 10
```

混在も可能です。`--train-list` だけ与えれば、リストに含まれないファイルが valid / test に自動分割されます。

---

## クイックスタート

### 1. 同梱の小規模リストで n-gram モデルを学習

```bash
kgfm train \
    --train-list list_small/train.txt \
    --valid-list list_small/valid.txt \
    --test-list  list_small/test.txt \
    --encoder ngram \
    --max-steps 5000 \
    --batch-size 256 \
    --ckpt-dir checkpoints/ngram
```

valid 上のベストチェックポイントが `checkpoints/ngram/best.pt` に保存され、最終的な test レポートが標準出力に流れます。

### 2. 凍結 mBERT + 射影ヘッドで学習

```bash
kgfm train \
    --train-list list_small/train.txt \
    --valid-list list_small/valid.txt \
    --test-list  list_small/test.txt \
    --encoder transformer \
    --transformer-model bert-base-multilingual-cased \
    --freeze-encoder \
    --proj-dim 256 \
    --batch-size 64 \
    --max-steps 5000 \
    --ckpt-dir checkpoints/bert
```

### 3. 保存済みチェックポイントを評価

```bash
kgfm eval \
    --ckpt checkpoints/ngram/best.pt \
    --test-list list_small/test.txt \
    --n-eval-triples 5000 \
    --pool-size 5000
```

---

## ライブラリとして利用

```python
import torch
from kgfm import (
    DistMultScorer, make_encoder,
    StreamingTripleDataset, collate_triples,
    evaluate,
)

encoder = make_encoder("ngram", embedding_dim=256)
scorer  = DistMultScorer(encoder, normalize=True).cuda()

ds = StreamingTripleDataset(files=["path/to/some.tsv"], shuffle_files=False)
loader = torch.utils.data.DataLoader(
    ds, batch_size=128, collate_fn=collate_triples,
)
batch = next(iter(loader))
h, r, t = scorer.encode_triple(batch["h_text"], batch["r_text"], batch["t_text"])
print((h * r * t).sum(-1))   # 行ごとのスコア
```

---

## リポジトリ構成

```
.
├── kgfm/                # Python パッケージ本体
│   ├── __init__.py      # 公開 API の re-export
│   ├── data.py          # ストリーミング TSV パイプライン
│   ├── encoders.py      # HashedNgram / Transformer エンコーダ + ENCODER_PRESETS
│   ├── heads.py         # 射影ヘッドのレジストリ (auto / identity / linear / mlp / residual_mlp)
│   ├── model.py         # DistMultScorer
│   ├── losses.py        # 学習目的関数 (contrastive / softmax_ce / bce / ...)
│   ├── eval.py          # MRR / Hit@k / nDCG (kgfm eval)
│   ├── train.py         # 学習ループ (kgfm train)
│   ├── cli.py           # `kgfm` コマンド (train / eval / bench / report / viz / hf / scaling)
│   ├── report.py        # 実行結果の集計 (kgfm report)
│   ├── report_html.py   # report.html (比較表 + 学習曲線) の生成
│   ├── charts.py        # グラフ描画 (plotly / matplotlib / 内蔵 SVG)
│   ├── viz.py           # 埋め込みの 2 次元射影 (kgfm viz)
│   ├── hf.py            # HuggingFace Hub への公開 (kgfm hf)
│   ├── runs.py          # 実行ディレクトリと tee ログ
│   ├── envs.py          # conda env 解決 (子プロセスの python を明示的に決める)
│   ├── utils.py         # GPU セレクタ、ファイルハッシュバケット
│   ├── bench/           # kgfm 自身のベンチマーク (kgfm bench ...)
│   ├── scaling/         # スケーリング則の集計 (kgfm scaling)
│   └── baselines/       # ULTRA / MOTIF (kgfm-ultra / kgfm-motif)
├── benchmarks/          # 手法比較のシェルラッパ + config_*.yaml (詳細は benchmarks/README.md)
├── benchmark_scaling/   # スケーリング則の実験 (詳細は benchmark_scaling/README.md)
├── list_small/          # スモークラン用の小さなファイルリスト
├── list_large/          # フルスケールのファイルリスト
├── list_chembl/         # ChEMBL 専用ベンチマークリスト
├── checkpoints/         # 保存先 (gitignore)
├── data -> ...          # 実コーパスへのシンボリックリンク
├── pyproject.toml
├── setup.py
├── requirements.txt
└── README.md
```

---

## エンコーダとヘッド

三つ組は文字列のまま**テキストエンコーダ**に入り、その出力を**ヘッド**が
スコア次元に射影して DistMult のスコアになります。この 2 つは別々に
差し替えられます。

### エンコーダ (`--encoder`)

`ngram`（学習可能な文字 n-gram ハッシュ）、`transformer`
（`--transformer-model` で任意の HuggingFace モデル）、または**プリセット名**を
指定します（`kgfm/encoders.py:ENCODER_PRESETS`）。

| プリセット | モデル | 次元 | 備考 |
|---|---|---|---|
| `bert-tiny` | `prajjwal1/bert-tiny` | 128 | 4.4M。以下 4 つは Turc et al. 2019 の同一系統 |
| `bert-mini` | `prajjwal1/bert-mini` | 256 | 11.2M |
| `bert-small` | `prajjwal1/bert-small` | 512 | 28.8M |
| `bert-medium` | `prajjwal1/bert-medium` | 512 | 41.4M |
| `scratch-tiny` … `scratch-base` | 上記 4 つ + `bert-base-uncased` の config | 128–768 | **ランダム初期化** 4.4M / 11.2M / 28.8M / 41.4M / 109.5M |
| `scratch-xl` | `bert-large-uncased` を 16 層に縮めた config | 1024 | ランダム初期化 234.6M |
| `scratch-large` | `bert-large-uncased`（24 層） | 1024 | ランダム初期化 335.4M。H200 1 枚で回せるサイズ軸の上端 |
| `bert-multilingual` | `bert-base-multilingual-cased` | 768 | `--encoder transformer` で `--transformer-model` を省略したときに読まれるモデル（`kgfm/train.py` の argparse 既定値）。プリセット機構を入れる前の実験はすべてこれで、比較用に残しています |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | 768 | |
| `bge-large` | `BAAI/bge-large-en-v1.5` | 1024 | 検索特化 335M |
| `e5-large` | `intfloat/e5-large-v2` | 1024 | 別系統の事前学習 |
| `gte-large` | `thenlper/gte-large` | 1024 | |
| `xlm-roberta-large` | `xlm-roberta-large` | 1024 | 多言語 |
| `e5-mistral-7b` | `intfloat/e5-mistral-7b-instruct` | 4096 | **frozen 専用**・bf16 |

プリセットは**すべてこの環境でロードと forward を確認済み**です。

- **7B 級は `frozen_only`** です。`encode_triple` は 1 step で 3B 本の系列を
  エンコーダに通すため、実用的なバッチサイズでの fine-tune は 1 GPU に載りません。
  sweep 側もこれらの `freeze=off` セルを自動的に生成しません。重みは更新されない
  ので bf16 でロードし、fp32 の 28 GB を半分にしています。
- bf16 でロードしたエンコーダは出力を fp32 に戻します。ヘッドとスコアラは
  fp32 で、**評価は autocast の外で走る**ため、これをやらないと最初の評価で
  `mat1 and mat2 must have the same dtype` になります。
- **`scratch-*` は重みをダウンロードしません**。`random_init: True` は
  `AutoModel.from_config()` を通り、config と tokenizer だけを取ります。
  事前学習をサイズ軸から外すためのもので、`benchmark_scaling/` の
  スケーリング則測定で使います — `bert-tiny … bert-base` は大きいモデルほど
  長く事前学習されているため、そのままではサイズと事前学習量が交絡します。
  既定学習率も別系統で、`train.SCRATCH_LR = 1e-4` です。既存のどちらの値も
  当てはまらないためです — 3e-5 は「事前学習済みの重みを壊さない」ための値で
  ここでの制約ではなく、1e-3 は BERT を確実に崩壊させます。**ただし 1e-4 が
  適切なのは最小サイズだけで、実際の実験ではサイズごとに設定する必要があります**
  — `scratch-base` は 1e-4 で崩壊し、`scratch-small` は自身の最適値から
  0.31 nats 離れます。実測値と経緯は `benchmark_scaling/README.md` と
  `config_scaling_scratch.yaml` を参照してください。
- `trust_remote_code` が必要なモデルはプリセット側で明示します（自動では
  有効化しません）。
- **2 つは試して落としました**（常にクラッシュするプリセットを置くより
  除外する方がよいため）:
  `microsoft/deberta-v3-large` は現行 `transformers` で tokenizer を
  fast/slow どちらでも構築できず、`Alibaba-NLP/gte-Qwen2-7B-instruct` は
  同梱の `modeling_qwen.py` が `DynamicCache` から削除された
  `get_usable_length()` を呼びます。どちらもバージョン非互換です。

### ヘッド (`--head`)

| `--head` | 中身 | frozen 時の学習パラメータ |
|---|---|---|
| `auto` **(既定)** | 幅が一致すれば `Identity`、違えば `Linear` | 幅が一致すると **0** |
| `identity` | 射影しない | 0 |
| `linear` | `Linear(in, out)` | あり |
| `mlp` | `Linear → GELU → Dropout → Linear` | あり |
| `residual_mlp` | `LayerNorm → MLP` を残差接続してから射影 | あり |

- **`auto` はヘッドが選択肢になる前の挙動そのもの**です（幅が一致すれば
  `nn.Identity`、違えば `nn.Linear`）。`heads` 軸を持たない config
  （`config_small` / `_middle` / `_large` / `_xlarge`、および
  `benchmark_scaling/` の各 config）はこの経路を通るため、既定を
  `linear` ではなく `auto` にしてあります。ただし frozen エンコーダ +
  幅一致だと**学習パラメータが 0** になるので（`proj` が `Identity` に
  縮退し、エンコーダ側は `requires_grad=False`）、凍結して比較するときは
  `linear` 以降を明示してください。
- 中間幅は**入力側**に合わせています。射影は普通は圧縮（1024 → 256）なので、
  狭い側に合わせると非線形性を通す前に情報を捨ててしまうためです。
- `--head-dropout` はエンコーダ出力の直後と MLP 系の内部の両方に効きます。

### ベンチマークでの比較

`encoders × heads × freezes` が sweep の軸です。セルのタグは
`<encoder>[_<head>][_frozen]` で、**`_<head>` の部分は `heads` を 2 つ以上
振ったときだけ**入ります。つまり `heads` を振らない config
（`config_large.yaml` など）のタグは `ngram` / `transformer_frozen` の形の
まま変わらず、結果 JSON のファイル名も変わりません。

```bash
bash benchmarks/run_chembl_xlarge_compare.sh              # 28 セルの比較
bash benchmarks/run_chembl_xlarge_compare.sh --heads linear   # 軸を絞る
bash benchmarks/run_chembl_xlarge_compare.sh --freezes on      # frozen のみ
```

`benchmarks/config_xlarge_compare.yaml` は**確認済みプリセット全 8 種**
（ngram + 7 プリセット）× `heads: [linear, mlp]` × `freezes: [off, on]` で
**28 セル**です。両方の freeze が可能なものはすべて両方を実行し、不可能な
組み合わせ（ngram の frozen、7B の fine-tune）は `cell_specs()` が自動的に
落とします。

この 28 セルは実際に完走しており、結果は
`benchmarks/results/chembl/20260811T230140Z_chembl_xlarge_compare/table.md`
にあります（全 28 行 × 2 プロトコル）。filtered MRR の上位・下位はこうなりました。

| 順位 | セル | filtered MRR |
|---|---|---|
| 1 | `gte-large` + `mlp` + fine-tune | **0.4641** |
| 2 | `bert-multilingual` + `linear` + fine-tune | 0.4538 |
| 3 | `bert-multilingual` + `mlp` + fine-tune | 0.4413 |
| 4 | `e5-large` + `mlp` + fine-tune | 0.4364 |
| 5 | `ngram` + `linear` + fine-tune | 0.4331 |
| … | | |
| 27 | `e5-mistral-7b` + `linear` + frozen | 0.3247 |
| 28 | `ngram` + `mlp` + fine-tune | 0.2539 |

読み方の要点が 3 つあります。**(1) ヘッドはエンコーダごとに向きが違います**
— `gte-large` は `mlp` で最良、`ngram` は `mlp` で最下位（0.4331 → 0.2539）
なので、「どのヘッドが良いか」は単独では決まりません。**(2) エンコーダを
大きくしても効いていません** — 7B (`e5-mistral-7b`) は frozen 専用なので
frozen 同士で比べると 0.3247 / 0.4079 で、178M の `bert-multilingual`
frozen (0.4018 / 0.4267) と同等以下です。**(3) 文字 n-gram は十分強い
ベースラインです** — `ngram` + `linear` が 28 セル中 5 位で、1024 次元の
検索特化エンコーダのほとんどより上です。`config_xxlarge.yaml` の既定が
`gte-large` + `mlp` なのはこの表が根拠です。

比較の公平性のため `benchmarks/config_xlarge_compare.yaml` は
**全セルで `batch_size: 256` と `proj_dim: 256` を固定**しています。B-1 は
負例数そのものであり、`proj_dim` はスコアを計算する次元なので、これらが
セルごとに違うとアーキテクチャの差と交絡します。

**バッチサイズ 256 は選択ではなく「一番重いセルが通る最大値」**です。
`bge-large` + `mlp` の fine-tune は **B=512 でも B=384 でも OOM** します
（143 GiB の H200 で、推定ではなく実際の `kgfm bench cell` をアイドル GPU で
走らせて確認。512 は最初の学習 forward で約 136 GiB を要求して落ち、384 は
142,957 MiB / 143,771 MiB でピークに達して落ちる）。B=256 は 1024 次元の
4 プリセット（bge / e5 / gte / xlm-r）すべてで filtered プロトコル込みの
通し確認済みです。なお **`nvidia-smi` の数字ではセルのサイズは決められません**
— PyTorch のキャッシュアロケータはカードを埋めるまで確保を伸ばすので、
問題なく回っている B=256 でも 137 GiB 使用と表示されます。判定は実コマンドの
成否です。

実測スループット（B=256、fine-tune、`kgfm bench cell` を 12 step ＝ warmup
込みで回した値なので下振れ寄り）:

| エンコーダ (B=256, fine-tune) | ex/s |
|---|---|
| `ngram` | ~17,000 |
| `bge-large` | 615 |
| `e5-large` | 614 |
| `gte-large` | 610 |
| `xlm-roberta-large` | 842 |

frozen セルは LM への backward が無いので、fine-tune セルのおよそ 1.5 倍
速くなります。28 セル × 6000 step で学習が約 15 時間、評価が約 3 時間
（56 パス。filtered は 1 パスごとに 46 万件の tail 語彙をエンコードします）
の見込みで、1 日を見ておけば足ります。

---

## 学習目的関数（損失）

`--loss` で切り替えます（実装は `kgfm/losses.py`）。既定は `contrastive` です。

```bash
kgfm train --loss contrastive --loss-temperature 0.1 ...
kgfm bench run --loss softmax_ce ...            # YAML なら loss: / loss_temperature:
```

`kgfm train` は下記のハイパーパラメータをすべて公開しています
(`--loss-temperature` / `--margin` / `--adversarial-temperature` /
`--label-smoothing`)。`kgfm bench` が受け取るのは `--loss` と
`--loss-temperature` の 2 つだけです。

### 共通の設定: in-batch negative

すべての損失は **バッチ内の他の tail を負例** として使います。バッチサイズ
`B` がそのまま負例数を決めるので、`--batch-size` は速度だけでなく
**モデル側のハイパーパラメータ**でもあります。

```
B                    バッチサイズ（負例数 = B - 1）
h_i, r_i, t_i        i 番目の三つ組の埋め込み（D 次元）
q_i = h_i ⊙ r_i      クエリ（⊙ は要素ごとの積）
S_ij = <q_i, t_j>    スコア行列 [B, B]。正例は対角成分 S_ii
σ(x) = 1 / (1 + e^-x)
```

`DistMultScorer` は既定で `h` と `t` を L2 正規化しますが、**`r` は正規化
しません**（関係の強さを大きさで表現するため）。したがって `S` のスケールは
`‖q‖ = ‖h ⊙ r‖` の成長に委ねられます。これが下記の `contrastive` を既定に
した理由です。

#### 重複 tail のマスク（既定で有効）

バッチ内負例は「自分以外の tail はすべて誤り」と仮定しますが、実データでは
同じ tail が同一バッチに何度も現れます（ChEMBL, B=512 の実測で train 行の
48% / valid 行の 56%）。それらは正例と**同一のベクトル**にエンコードされる
ので分離不可能で、損失に

```
floor_i = log(mult_i)        mult_i = バッチ内で t_i と同じ文字列の個数
L >= (1/B) Σ_i log(mult_i)
```

という下限が生じます（実測 train 1.14 / valid 1.52 nats）。既定ではこれらの
セルを損失から除外します（`losses.duplicate_tail_mask`）。softmax 系は当該
ロジットを `-inf` にし、`bce` / `margin` は平均の分母から外します。無効化は
`--no-mask-duplicate-tails`。

large 実行のチェックポイントで測った効果:

| | unmasked | masked |
|---|---|---|
| ngram train / valid | 2.391 / 4.758 | **1.848 / 4.282** |
| transformer train / valid | 2.266 / 4.884 | **1.679 / 4.387** |

pooled 評価プロトコルが候補プールを一意化しているのと同じ理由の処理で、
これにより損失と評価指標が同じ土俵に乗ります。

### `contrastive`（既定） — InfoNCE / NT-Xent

クエリと tail を L2 正規化して cosine 類似度にし、**温度 τ** で割ってから
softmax 交差エントロピーを取ります。

```
q̂_i = q_i / ‖q_i‖ ,   t̂_j = t_j / ‖t_j‖
S̃_ij = <q̂_i, t̂_j> / τ

L = -(1/B) Σ_i  log [ exp(S̃_ii) / Σ_j exp(S̃_ij) ]
```

- パラメータ: `--loss-temperature` (τ, 既定 0.1)、`--label-smoothing`
- 分布の鋭さが `‖r‖` の副産物ではなく**明示的なハイパーパラメータ**になります。
- **行ごとの正規化は順位を変えません。** 同じ行の全候補を同じ正数で割るだけ
  なので、MRR / Hit@k は不変で損失の幾何だけが変わります。

### `softmax_ce` — 生スコアの softmax 交差エントロピー

`contrastive` から正規化と温度を除いたもの。`kgfm/losses.py`（`--loss` の
切り替え機構）が入る前は損失がこれ 1 つしかなく、**それ以前に取った結果は
すべてこの目的関数で学習されたもの**です。`benchmarks/results/` にある
2026-08-09 以前の run が該当します。

```
L = -(1/B) Σ_i  log [ exp(S_ii) / Σ_j exp(S_ij) ]
```

- パラメータ: `--label-smoothing`
- スケール制御が無いため、学習が進むと `‖r‖` が伸びて logit が鋭くなり、
  **順位が良くても損失が悪化**します（下記「なぜ既定を変えたか」）。

### `bce` — 候補ごとの二値交差エントロピー（ConvE の 1-N scoring）

`B×B` の各セルを独立した yes/no 判定として扱います。

```
y_ij = 1 (i = j),  0 (i ≠ j)
ラベル平滑化 ε:  y_ij ← y_ij (1 - ε) + ε / B

L = -(1/B²) Σ_i Σ_j [ y_ij log σ(S_ij) + (1 - y_ij) log(1 - σ(S_ij)) ]
```

- パラメータ: `--label-smoothing`（ConvE は 0.1 を使用）
- 正例:負例が 1:(B-1) と偏るため、ラベル平滑化が事実上必須です。

### `margin` — max-margin ヒンジ（TransE 系）

すべての負例が正例より `γ` 以上低くなることを要求します。

```
L = 1 / (B(B-1))  Σ_i Σ_{j≠i}  max(0, γ - S_ii + S_ij)
```

- パラメータ: `--margin` (γ)
- 平均は対角を除いた `B(B-1)` 組で取ります（常に 0 の対角を含めると
  `B/(B-1)` 倍だけ損失が薄まるため）。

### `self_adversarial` — RotatE の self-adversarial negative sampling

負例を「難しさ」で重み付けします。重みは負例自身のスコアの softmax で、
**勾配を流しません**（サンプル重みであって目的関数の一部ではないため）。

```
w_ij = softmax_{j≠i} ( α · S_ij )        （detach）

L = -(1/B) Σ_i [ log σ(γ + S_ii) + Σ_{j≠i} w_ij · log σ(-γ - S_ij) ]
```

- パラメータ: `--margin` (γ)、`--adversarial-temperature` (α)
- 元論文は距離 `d = -S` に対する `σ(γ - d)` 形式で、上式はそれを類似度
  表現に書き換えたものです (Sun et al., 2019)。

### なぜ既定を `contrastive` にしたか

`softmax_ce` はスケールを制御しないため、学習が進むと **順位は良いのに
損失が悪化する**という現象が起きます。60k step 学習した transformer での実測:

| 指標 | 値 |
|---|---|
| 平均 `‖r‖` | 157 |
| logit の標準偏差 | 22.3 |
| validation 交差エントロピー | **13.2**（ランダムは ln(512) = 6.24） |
| 同じバッチでの median rank | 512 中 **12 位** |
| 温度で割るだけの CE | **5.3**（順位は 1 つも変わらず） |

MRR は単調変換に不変ですが交差エントロピーは不変ではないため、損失は
「順位の良さ」ではなく「スコアの校正」を測っていたことになります。

ngram で 1500 step 学習した比較（B=512）:

| `--loss` | valid loss | valid MRR |
|---|---|---|
| `softmax_ce` | 6.18 → 6.09 | 0.107 |
| `contrastive` (τ=0.1) | 5.68 → 5.57 | **0.816** |

温度の既定 0.1 も実測で選びました（τ=0.05 は valid loss 6.39 / MRR 0.816、
τ=0.1 は 5.64 / 0.816）。順位性能は同じで、0.1 の方がランダム値
ln(512)=6.24 を明確に下回るため損失が解釈可能なまま保たれます。

> **注意**: `margin` と `self_adversarial` は**生スコア** `S` を使うため、
> `--margin` をそのスケールに合わせて調整する必要があります。既定値のまま
> 差し替えても妥当な学習にはなりません。
>
> **既定が `softmax_ce` から `contrastive` に変わっているので、損失の値は
> 世代をまたいで比較できません**（順位指標 MRR / Hit@k は行ごとの単調変換に
> 不変なので比較できます）。`--loss softmax_ce` を渡せば旧既定を再現できます。

---

## 正則化（エンコーダ / 結合部を別々に）

エンコーダ本体と、その出力をスコアに繋ぐ**結合部（projection head）**は
過学習の速度が違うので、正則化パラメータを分けています。large 実行の実測では
fine-tune した transformer の train 損失が 2.25→1.97 と下がり続ける一方で
valid 損失は step 12.5k を底に上昇しましたが、head しか学習しない frozen セル
では同じ現象が起きませんでした。

| フラグ | 対象 | 既定 |
|---|---|---|
| `--weight-decay` | 下 2 つの共通の既定値 | 0.0 |
| `--encoder-weight-decay` | エンコーダのみ | `--weight-decay` |
| `--head-weight-decay` | projection head のみ | `--weight-decay` |
| `--encoder-dropout` | エンコーダ内部 | 未指定 |
| `--head-dropout` | エンコーダ出力 → head の間 | 0.0 |

```bash
kgfm train --encoder transformer --encoder-weight-decay 0.01 --head-weight-decay 0.0 \
           --encoder-dropout 0.2 --head-dropout 0.1 --proj-dim 256 ...
kgfm bench run --config large --encoder-weight-decay 0.01 --head-dropout 0.1
```

- **`--encoder-dropout` 未指定は 0 ではありません。** transformer では事前学習
  済み config の値（BERT なら 0.1）をそのまま使い、ngram では 0 になります。
  値を渡すと `hidden_dropout_prob` / `attention_probs_dropout_prob` 相当の
  項目をアーキテクチャに応じて書き換えます。
- **bias と LayerNorm（1 次元パラメータ）は常に weight decay の対象外**です。
  エンコーダに decay をかけられるようになったぶん、この除外が効きます。
- 起動時に実際のパラメータ群が `[init]` 行に出ます:
  `encoder=0.01(n=76) encoder_nodecay=0(n=123) head=0(n=1) head_nodecay=0(n=1)`
  （bert-base-multilingual-cased + `--proj-dim 256` の場合）
- frozen エンコーダでは学習されるのが head だけなので、
  `--head-weight-decay` / `--head-dropout` のみが効きます（`--proj-dim` 必須）。

---

## 学習量の指定: `--max-steps` と `--max-epoch`

step 数の代わりに**エポック数（小数可）**でも指定できます。

```bash
kgfm train --max-epoch 1    --train-list list_chembl/train.txt ...   # ちょうど 1 周
kgfm train --max-epoch 0.25 ...                                      # コーパスの 1/4
kgfm bench run --config xxlarge --max-epoch 0.5
```

```yaml
defaults:
  max_epoch: 1.0     # benchmarks/config_xxlarge.yaml はこれを使っています
```

**`--max-epoch` を指定すると `--max-steps` は無視されます。** step 数は

```
steps = ceil(max_epoch × epoch_examples / (batch_size × nproc × grad_accum))
```

で導出されるので、**バッチサイズや GPU 数を変えても 1.0 は 1 エポックのまま**
です。DDP では train ファイルが `files[rank::world_size]` で分割され各 rank が
互いに素な半分を読むため、`nproc` が分母に入るのは正しい挙動です
（これまで 2 GPU では max_steps を手で半分にする必要がありました）。

エポックのサイズは**実際に行数を数えて**求めます。ChEMBL のファイルは
1.81 GB〜39 KB と不均一なので、1 ファイル分から掛け算すると 26% ずれます。

- 約 1.3 GB/s で読むので 105 GiB の初回は約 5 分。以降は
  `(パス, サイズ, mtime)` をキーに `~/.cache/kgfm/rowcounts.json` に
  キャッシュされ即時になります（`KGFM_ROWCOUNT_CACHE` で変更可）。
  サイズか mtime が変われば再カウントします。
- `max_rows_per_file` と `row_keep_prob` はエポックサイズに反映されます
  （`row_keep_prob < 1` のときは確率的なので期待値です）。
- DDP では**シャードごとに測らず**エポックを均等割りします。シャードは
  不均一ですが、全 rank が同じ step 数で回らないと集団通信がデッドロック
  するためです。

### ファイルはインターリーブして読む

ChEMBL の TSV は activity ID で分割されているため、**1 ファイル = 1 つの
エンティティ集団**です。`StreamingTripleDataset` は既定で
`interleave_chunk`（64）行ずつ**全 train ファイルをラウンドロビンに**読みます
(`interleave_files=True`)。

これは 2026-08-25 に既定を変えた箇所です。それ以前は各ファイルを最後まで
読んでから次に移る実装でしたが、1 ファイルが 1000 万行あるので**どのファイルも
読み終わりません** — 25k step / B=512 の run が worker あたり 320 万行しか
引かないので、85 ファイル中 4 つの先頭 32% しか見ないことになります。学習
ストリームは 1 つの分布ではなく「分布の列」で、勾配は常に直近のファイルの
偏りを追いかけていました。

ラウンドロビンでも**行の多重集合は完全に同一**で、順序だけが変わります
（2 ファイル各 500 行で多重集合が一致することを検証済み）。効果は集団の
混合で、先頭 2000 行の平均 tail 長は連結読みで 26.6（1 ファイル目そのものの
値）、インターリーブで 39.7（2 ファイルの中間）でした。

**`--no-interleave-files` は 2026-08-25 より前の run を再現するためだけの
フラグ**です。新しく実験するときに渡す理由はありません。
validation loss も既定でシャッフルしたストリーム上で測ります
（`--no-valid-loss-shuffle` で無効化）— シャッフルしないと in-batch 負例が
「たまたま隣にいた行」になってしまうためです。どちらも `kgfm train` と
`kgfm bench cell` の両方にあり、config の `cells:` でセルごとに設定できます。

なお `kgfm eval` の候補プールと filter index (`eval.build_candidate_pool` /
`build_filter_index`) も同じデータセットを使うため、プールが 1 ファイルの
先頭からではなく全ファイルから引かれるようになります。より代表的になり
ますが、**候補プールの中身が変わるので、2026-08-25 より前に取った評価値
（`benchmarks/results/chembl/` の 20260816 以前の run）とは比較できません**。

---

## 学習済みモデルの公開 (`kgfm hf`)

学習済みチェックポイントを HuggingFace Hub に公開します。

```bash
pip install -e ".[hf]"

# ベンチマーク実行からセルを指定して
kgfm hf --out-dir latest --tag bge-large_linear_frozen --repo you/kgfm-bge

# 任意のチェックポイントを直接
kgfm hf --ckpt checkpoints/ngram/best.pt --repo you/kgfm-ngram

# 何がアップロードされるかだけ確認（ネットワークに触りません）
kgfm hf --out-dir latest --tag ngram_linear --repo you/x --dry-run --dry-run-dir /tmp/stage
```

アップロードされるのは 3 ファイルです。`kgfm_model.pt`（ペイロード）、
`kgfm_config.json`（`TrainConfig` と評価指標、`.pt` を開かずに中身が分かる）、
`README.md`（モデルカード。アーキテクチャ表・実行済みなら指標表・使い方）。
実行ディレクトリから指定した場合は `kgfm_<protocol>_<tag>.json` の指標が
自動でカードに入ります。

**サイズを 2 段階で削ります。**

| | 削るもの | 実測 |
|---|---|---|
| 既定 | optimizer state（推論に不要、モデルの約 2 倍） | ngram 3.1 GB → **1.0 GB** |
| `--mode head-only` | 凍結エンコーダの重み | bge-large 1.2 GB → **1.0 MB** |
| 同上 | 同上（7B） | e5-mistral-7b 13.2 GB → **68 MB** |

`freeze_encoder=True` はエンコーダのパラメータが全学習中 `requires_grad=False`
だったことを意味するので、その重みは公開モデルとビット単位で同一です。
再アップロードは他人のリリースの GB 単位のコピーにすぎず、**head だけが
学習された成果物**です。`kgfm.hf.load` がプリセットからエンコーダを再構築して
head を載せ直します。等価性は確認済みで、元のチェックポイントと再構築後の
スコアは**ビット単位で一致**します（head-only / full 両方）。

`--mode` の既定は `auto`（無損失なときだけ head-only）。fine-tune 済み
エンコーダや ngram（スクラッチ学習なので公開コピーが無い）に `head-only` を
指定すると、理由付きで拒否します。

```python
from kgfm.hf import load
scorer = load("you/kgfm-bge")        # head-only ならエンコーダを自動取得
h, r, t = scorer.encode_triple(["aspirin"], ["treats"], ["headache"])
score = scorer.score(h, r, t)
```

- **リポジトリは既定で private** です。公開は後戻りしにくいので `--public` を
  明示したときだけ公開になります。
- トークンは `--token` → `$HF_TOKEN` → `$HUGGINGFACE_HUB_TOKEN` →
  キャッシュ済みログインの順に解決します。
- `--which best|final|last` でスナップショットを選べます（既定 `best`）。
- `--with-optimizer` で optimizer state を残せます（Hub から学習を再開したい
  場合のみ）。
- head-only のペイロードは `kgfm eval --ckpt` では読めません（strict な
  state_dict ロードで、エンコーダのテンソルを意図的に含まないため）。
  モデルカードにもその旨が出ます。full なら `kgfm eval --ckpt` で直接使えます。

---

## 評価プロトコル

`kgfm.eval.evaluate` は `protocol` 引数で 2 種類のランキングプロトコル
を切り替えられます。CLI では `kgfm eval --protocol pooled|filtered`、
ベンチマークでは `kgfm bench run --protocols ...` でも同じです。

### `pooled` (デフォルト・高速)

1. test ファイルをストリーミングし、`pool_size` 個のユニークな tail テキストを候補プールとして集める。
2. プールを 1 度だけバッチエンコードして埋め込みを保持。
3. 各 test 三つ組について `(h, r)` をプール内 tail 全件 + 真の tail と内積し、真の tail の順位を取る。

`pool_size` でメモリ上限が決まるため、任意サイズの test セットに対して
スケールします。学習中の in-loop 検証は常にこの方式です。

### `filtered` (KG 標準・ULTRA / MOTIF と直接比較可能)

1. **filter ファイル**（既定では train + valid + test の和集合）をストリーミングし、
   全 tail 候補語彙と `(h_text, r_text) -> {tail_text}` の真値テーブルを作る。
2. 全 tail 語彙を 1 度だけエンコード。
3. 各 test 三つ組 `(h, r, t)` について全 tail に対してスコアを計算し、
   同じ `(h, r)` に対する `t` 以外の正解 tail のスコアを `-inf` でマスク
   してから `t` の順位を取る。

ULTRA / MOTIF など多くの公開数値はこのプロトコルです。候補語彙が
大きくなりがちなため、`--max-filter-tails` で語彙サイズ、
`--max-filter-rows` で読み取り行数を上限指定できます (multi-TB 級の
コーパスで実用的)。

```bash
# 例: filtered プロトコルで kgfm eval を実行
kgfm eval \
    --ckpt checkpoints/ngram/best.pt \
    --test-list list_chembl/test.txt \
    --protocol filtered \
    --filter-list list_chembl/train.txt \
    --filter-list list_chembl/valid.txt \
    --filter-list list_chembl/test.txt \
    --max-filter-tails 50000 \
    --max-filter-rows 1000000
```

---

## ベンチマーク (ULTRA / MOTIF との比較)

`benchmarks/` ディレクトリに、KG 基盤モデルである **ULTRA**
([DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA)) と
**MOTIF** ([HxyScotthuang/MOTIF](https://github.com/HxyScotthuang/MOTIF))
との比較スクリプト一式を置いています。詳細は `benchmarks/README.md` を
参照してください。ベンチマークは 1 コマンドで実行 + 自動記録できます。

```bash
# ChEMBL ベンチマークを丸ごと実行 (セットアップ→prep→各手法→集計)
bash benchmarks/run_chembl.sh

# 結果は benchmarks/results/chembl/<UTC タイムスタンプ>/ に記録
ls benchmarks/results/chembl/latest/
# meta.json kgfm_<protocol>_<encoder>.json ultra.json motif.json table.md report.html run.log ...
```

**3 手法が同じ run に揃っている記録**は
`benchmarks/results/chembl/20260808T073317Z_fulltest/table.md` です
（`config_small.yaml` 相当: prep 50k / 2k / 2k、|E| = 72,669、
inductive モード、kgfm は 200 step のみ）。

| 手法 | モード | Protocol | MRR | Hit@1 | Hit@3 | Hit@10 | n_eval |
|---|---|---|---|---|---|---|---|
| kgfm (ngram, 200 steps) | 学習 | pooled (5000) | 0.2632 | — | — | 0.2807 | 5120 |
| kgfm (ngram, 200 steps) | 学習 | filtered (50000) | 0.2698 | — | — | 0.2791 | 5120 |
| kgfm (transformer, 200 steps) | 学習 | pooled (5000) | 0.1966 | — | — | 0.3891 | 5120 |
| kgfm (transformer, 200 steps) | 学習 | filtered (50000) | 0.2851 | — | — | 0.3639 | 5120 |
| ULTRA (zero-shot, CPU) | 推論のみ | filtered | 0.1753 | 0.1532 | 0.1993 | 0.2100 | 2000 |
| MOTIF (zero-shot, GPU) | 推論のみ | filtered | 0.1740 | 0.1470 | 0.2003 | 0.2100 | 2000 |

> **これは「3 手法が同じディレクトリで動く」ことの確認で、kgfm の性能を
> 示す数字ではありません。** kgfm 側は 200 step しか学習していません。
> また `n_eval` は手法間で意味が違い（kgfm は tail 方向のみ、ULTRA / MOTIF は
> head+tail の平均なので分母が実質 2 倍）、直接横並びにはできません。
> 詳細は `benchmarks/README.md` の「test サンプル数 (`n_eval`) と評価指標への
> 影響」を参照してください。

**kgfm 本体の実力**は下記の 2 つの run です（どちらもベースライン行は
含まれていません）。

| run | 条件 | filtered MRR | pooled MRR |
|---|---|---|---|
| `20260811T230140Z_chembl_xlarge_compare` | 28 セル比較の最良（`gte-large` + `mlp`）、6,000 step | 0.4641 | 0.4929 |
| `20260816T053252Z_chembl_xxlarge` | 同じセルで**全 674M 行を 1 エポック**（1,316,925 step） | 0.4410 | **0.5164** |

filtered が下がっているのは候補語彙が 419k → 1,563,335 と 3.7 倍になって
分母が増えたためで、データ量の効果とは分離できていません。pooled は
両方 `pool_size: 12000` なので比較可能で、こちらは上がっています。

3 手法はそれぞれ独立したコマンドです。共有しているのは実行ディレクトリだけで、各手法がそこに JSON を落とし、`kgfm report` が拾って 1 枚の表にします。`benchmarks/*.sh` は repo root に `cd` してこれらを順に呼ぶだけの数行のラッパです。

```bash
kgfm bench run --config benchmarks/config_large.yaml   # kgfm 自身のベンチマーク
                                                       # small / middle / large / xlarge /
                                                       # xlarge_compare / xxlarge
kgfm-ultra --out-dir latest       # 別手法・別コマンド
kgfm-motif --out-dir latest
kgfm viz --ckpt <ckpt>            # h/t 埋め込みを 2 次元に射影 (PCA / UMAP)
kgfm report --out-dir latest      # 集計 → table.md + report.html (--list で run 一覧)
```

env は 2 つです: kgfm 自身は **`kgfm`**、ベースライン (ULTRA / MOTIF) は **`kgfm-ultra`** で動きます。後者は両者が `rspmm` CUDA 拡張を JIT ビルドする都合で torch と一致する nvcc が要るためで、`bash benchmarks/setup_baseline_env.sh` で構築します。どちらの env で動かしても指標は一致することを実測済みです。詳細とハマりどころは `benchmarks/README.md` の「conda 環境」節を参照してください。

---

## スケーリング則の測定 (`benchmark_scaling/`)

「どの構成が一番強いか」を問う `benchmarks/` とは別に、「**計算量を増やすと
損失はどう下がるか**」を問う実験を `benchmark_scaling/` に置いています。
sweep 軸はモデルサイズ（`scratch-tiny` 4.4M 〜 `scratch-large` 335M）で、
出力は比較表ではなく計算量-損失のプロットです。

```bash
bash benchmark_scaling/run_scaling_smoke.sh     # 数分の通し確認
bash benchmark_scaling/run_lr_probe.sh          # サイズごとの学習率を測る（先にこれ）
bash benchmark_scaling/run_scaling_scratch.sh   # 本番（ランダム初期化 7 サイズ）
```

**学習コードは `kgfm bench run` と共通**で、スケーリング専用の学習経路は
ありません。`kgfm scaling` が既存の `cell_*.log` を (計算量, 損失) 座標に
読み替えるだけなので、終わった run に後から適用できます。設計・実測値・
落とし穴（学習率をサイズごとに振らないと指数が平坦化する、early stopping を
入れてはいけない、など）は `benchmark_scaling/README.md` にまとめてあります。

---

## ライセンス

MIT.
