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
```

PyTorch は CUDA / ドライバに依存します。`pip install torch` で適切な wheel が入らない場合は、<https://pytorch.org/get-started/locally/> から直接導入してください。

インストール後、2 つのコンソールスクリプトが利用可能になります。

```bash
kgfm-train --help
kgfm-eval  --help
```

モジュールとして直接呼ぶこともできます。

```bash
python -m kgfm.train --help
python -m kgfm.eval  --help
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
kgfm-train \
    --train-list list_small/train.txt \
    --valid-list list_small/valid.txt \
    --test-list  list_small/test.txt
```

**自動分割フォールバック** — リストが指定されない場合、`--data-root` 配下のファイルがハッシュバケットで分割されます。デフォルトは 80 / 10 / 10 です。

```bash
kgfm-train --data-root data --valid-buckets 1 --test-buckets 1 --n-buckets 10
```

混在も可能です。`--train-list` だけ与えれば、リストに含まれないファイルが valid / test に自動分割されます。

---

## クイックスタート

### 1. 同梱の小規模リストで n-gram モデルを学習

```bash
kgfm-train \
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
kgfm-train \
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
kgfm-eval \
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
│   ├── eval.py          # MRR / Hit@k / nDCG + kgfm-eval CLI
│   ├── train.py         # 学習ループ + kgfm-train CLI
│   └── utils.py         # GPU セレクタ、ファイルハッシュバケット
├── benchmarks/          # ULTRA / MOTIF との比較ベンチマーク (詳細は benchmarks/README.md)
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

## 評価プロトコル

`kgfm.eval.evaluate` は `protocol` 引数で 2 種類のランキングプロトコル
を切り替えられます。CLI では `kgfm-eval --protocol pooled|filtered`、
`run_kgfm.py --protocol ...` でも同じです。

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
# 例: filtered プロトコルで kgfm-eval を実行
kgfm-eval \
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
参照してください。各ベンチマークは `bootstrap_<name>.sh` で 1 コマンド
実行 + 自動記録できます。

```bash
# ChEMBL ベンチマークを丸ごと実行 (セットアップ→prep→各手法→集計)
bash benchmarks/bootstrap_chembl.sh

# 結果は benchmarks/results/chembl/<UTC タイムスタンプ>/ に記録
ls benchmarks/results/chembl/latest/
# meta.json kgfm.json ultra.json motif.json table.md run.log ...
```

ChEMBL 上でのスモーク実測例 (50k 学習 / 2k テスト・三つ組、kgfm は 200 steps のみの簡易学習):

| 手法 | モード | Protocol | MRR | Hit@1 | Hit@3 | Hit@10 |
|---|---|---|---|---|---|---|
| kgfm (ngram, 200 steps) | 学習 | pooled (200) | 0.111 | — | — | 0.328 |
| kgfm (ngram, 200 steps) | 学習 | filtered (5000) | 0.037 | — | — | 0.082 |
| ULTRA (zero-shot, CPU) | 推論のみ | filtered | 0.175 | 0.153 | 0.199 | 0.210 |
| MOTIF (zero-shot, GPU) | 推論のみ | filtered | 0.174 | 0.147 | 0.200 | 0.210 |

> kgfm は文字 n-gram の 200 step 学習という極めて軽い条件、ULTRA / MOTIF は事前学習済みモデルでのゼロショット推論です。filtered プロトコルでは候補集合が約 25 倍に増えるため数字が大きく下がります。kgfm を ULTRA / MOTIF と直接比較するときは `--protocol filtered` を使い、学習ステップ数も同じ計算予算で揃えてください。

ULTRA は CUDA 拡張の都合で torch のバージョンに敏感なため、専用の conda 環境 `kgfm-ultra` (Python 3.11 / torch 2.5.1+cu121) を `benchmarks/setup_ultra_env.sh` で作成し、`run_ultra.py` がその Python に subprocess で切り替えて実行します。詳細とハマりどころは `benchmarks/README.md` の「ULTRA 専用 conda 環境」節を参照してください。

---

## ライセンス

MIT.
