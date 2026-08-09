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
│   ├── encoders.py      # HashedNgram / Transformer エンコーダ
│   ├── model.py         # DistMultScorer
│   ├── losses.py        # 学習目的関数 (contrastive / softmax_ce / bce / ...)
│   ├── eval.py          # MRR / Hit@k / nDCG (kgfm eval)
│   ├── train.py         # 学習ループ (kgfm train)
│   ├── cli.py           # `kgfm` コマンド (train / eval / bench / report)
│   ├── report.py        # 実行結果の集計 (kgfm report)
│   ├── report_html.py   # report.html (比較表 + 学習曲線) の生成
│   ├── charts.py        # グラフ描画 (plotly / matplotlib / 内蔵 SVG)
│   ├── viz.py           # 埋め込みの 2 次元射影 (kgfm viz)
│   ├── runs.py          # 実行ディレクトリと tee ログ
│   ├── envs.py          # conda env 解決
│   ├── utils.py         # GPU セレクタ、ファイルハッシュバケット
│   ├── bench/           # kgfm 自身のベンチマーク (kgfm bench ...)
│   └── baselines/       # ULTRA / MOTIF (kgfm-ultra / kgfm-motif)
├── benchmarks/          # ベンチマーク用の薄いシェルラッパ (詳細は benchmarks/README.md)
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

`contrastive` から正規化と温度を除いたもの。本モジュール導入以前の挙動です。

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
> 既定を変更したため、`softmax_ce` 時代の結果とは直接比較できません。
> 従来挙動は `--loss softmax_ce` で再現できます。

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

ChEMBL 上でのスモーク実測例 (50k 学習 / 2k テスト・三つ組、kgfm は 200 steps のみの簡易学習):

| 手法 | モード | Protocol | MRR | Hit@1 | Hit@3 | Hit@10 |
|---|---|---|---|---|---|---|
| kgfm (ngram, 200 steps) | 学習 | pooled (200) | 0.111 | — | — | 0.328 |
| kgfm (ngram, 200 steps) | 学習 | filtered (5000) | 0.037 | — | — | 0.082 |
| ULTRA (zero-shot, CPU) | 推論のみ | filtered | 0.175 | 0.153 | 0.199 | 0.210 |
| MOTIF (zero-shot, GPU) | 推論のみ | filtered | 0.174 | 0.147 | 0.200 | 0.210 |

> kgfm は文字 n-gram の 200 step 学習という極めて軽い条件、ULTRA / MOTIF は事前学習済みモデルでのゼロショット推論です。filtered プロトコルでは候補集合が約 25 倍に増えるため数字が大きく下がります。kgfm を ULTRA / MOTIF と直接比較するときは `--protocols filtered` を使い、学習ステップ数も同じ計算予算で揃えてください。

3 手法はそれぞれ独立したコマンドです。共有しているのは実行ディレクトリだけで、各手法がそこに JSON を落とし、`kgfm report` が拾って 1 枚の表にします。`benchmarks/*.sh` は repo root に `cd` してこれらを順に呼ぶだけの数行のラッパです。

```bash
kgfm bench run --config benchmarks/config_large.yaml   # kgfm 自身のベンチマーク
                                                       # small/middle/large/xlarge
kgfm-ultra --out-dir latest       # 別手法・別コマンド
kgfm-motif --out-dir latest
kgfm viz --ckpt <ckpt>            # h/t 埋め込みを 2 次元に射影 (PCA / UMAP)
kgfm report --out-dir latest      # 集計 → table.md + report.html (--list で run 一覧)
```

env は 2 つです: kgfm 自身は **`kgfm`**、ベースライン (ULTRA / MOTIF) は **`kgfm-ultra`** で動きます。後者は両者が `rspmm` CUDA 拡張を JIT ビルドする都合で torch と一致する nvcc が要るためで、`bash benchmarks/setup_baseline_env.sh` で構築します。どちらの env で動かしても指標は一致することを実測済みです。詳細とハマりどころは `benchmarks/README.md` の「conda 環境」節を参照してください。

---

## ライセンス

MIT.
