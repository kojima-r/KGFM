# ベンチマーク: kgfm vs ULTRA vs MOTIF on ChEMBL

このディレクトリは、本リポジトリの **kgfm** を、KG リンク予測の基盤モデル
2 種類と比較するためのスクリプト群です。

- **ULTRA** — <https://github.com/DeepGraphLearning/ULTRA>
- **MOTIF** — <https://github.com/HxyScotthuang/MOTIF>

ベンチマーク用データは `list_chembl/` で用意した ChEMBL のサブセット
（生成手順はトップレベルの README 参照）です。

## 方法論

3 手法を 1 つの test セットで比較します。

| 手法 | 学習 | 入力 |
| --- | --- | --- |
| `kgfm` | `list_chembl/train.txt` で学習・`valid.txt` で選択 | 生の `(h_text, r_text, t_text)` 文字列 |
| `ULTRA` | **ゼロショット**（pretrained ckpt を使用、デフォルト `ultra_50g.pth`） | `prepare_chembl_kg.py` が生成する entity-ID KG |
| `MOTIF` | **ゼロショット**（pretrained ckpt を使用） | 同上 |

`kgfm` はテキストベースで、ChEMBL に対する pretrained checkpoint を持た
ないため、純粋なゼロショット比較は原理的に不可能です。本ベンチマークは
「ChEMBL でゼロから学習した kgfm」vs「ChEMBL を学習せずに転移する
ULTRA / MOTIF」という非対称な設定で動かしており、kgfm に絶対値で有利な
土俵ですが、その代わりに「ドメイン特化 KG ではテキスト対応のスクラッチ
学習がグラフ基盤モデルを上回りうるか」という逆向きの仮説を検証していま
す。

### プロトコル差異と統一方法

kgfm はランキングプロトコルを 2 つ持っています。

- **pooled** (デフォルト) — `pool_size` 件の tail 候補プールに対して
  順位を取る軽量モード。学習中の in-loop 検証で使用。
- **filtered** — 全 tail 語彙に対してスコアを計算し、同じ `(h, r)` の
  他の正解 tail を `-inf` でマスクして順位を取る KG 標準モード。
  ULTRA / MOTIF と直接比較したい場合はこちらを使う。

`run_kgfm.py --protocol filtered` で切り替え可能です。filtered では
語彙サイズ・読み取り行数を `--max-filter-tails` と `--max-filter-rows`
で上限指定できるため、multi-TB の chembl 全量でも現実的に走らせられ
ます。

ULTRA / MOTIF はもともと filtered ranking です。`aggregate.py` は
出力テーブルで Protocol 列を表示するので、揃っているかどうかが
ひと目でわかります。

---

## ワンショット実行 (推奨)

`bootstrap_<benchmark>.sh` で「セットアップ → データ準備 → 各手法の実行
→ 集計」を 1 コマンドで通します。結果は実行のたびに UTC タイムスタンプ
ディレクトリに記録され、最新へのシンボリックリンク `latest` が更新され
ます。

```bash
# ChEMBL ベンチマークを丸ごと実行
bash benchmarks/bootstrap_chembl.sh

# 細かく制御 (代表例)
bash benchmarks/bootstrap_chembl.sh \
    --max-train 200000 --max-valid 5000 --max-test 5000 \
    --max-steps 5000 \
    --protocol filtered \
    --ultra-gpus null   # H200 では rspmm バグ回避で CPU 推奨
```

実行時に作られる `benchmarks/results/chembl/<UTC タイムスタンプ>/` には:

```
meta.json                # ホスト / git rev / torch ver / 与えたパラメータ
chembl_kg_stats.json     # 構築した KG の統計のコピー
kgfm.json                # kgfm の評価結果
ultra.json               # ULTRA の評価結果
motif.json               # MOTIF の評価結果
table.md                 # 集計テーブル
run.log                  # 統合ログ
<step>.log               # 各ステップ単体ログ
kgfm_ckpts/              # この実行の kgfm チェックポイント
```

`benchmarks/results/chembl/latest -> <最新タイムスタンプ>` のシンボリック
リンクが常に最新を指します。

主な flag:

| flag | デフォルト | 用途 |
| --- | --- | --- |
| `--max-train N` / `--max-valid N` / `--max-test N` | 50000 / 2000 / 2000 | ChEMBL prep の三つ組キャップ |
| `--max-steps N` | 200 | kgfm の学習ステップ数 |
| `--batch-size N` | 256 | kgfm の学習バッチサイズ |
| `--transformer-batch-size N` | （`--batch-size` と同じ） | transformer エンコーダ系のセルだけバッチサイズを上書き。BERT-base のフル fine-tune は B=1024 で OOM するため、こちらで個別に下げる |
| `--proj-dim N` | （未指定 = `None`） | DistMult 直前に学習可能な `Linear` 射影を挿入。`--kgfm-freezes on` を使う場合は必須（凍結 LM + `proj_dim=None` だと学習可能パラメータが 0 になる）。ngram に対しても同値であれば `nn.Identity` に縮退するため無害 |
| `--kgfm-protocols LIST` | `pooled,filtered` | 最終評価プロトコルのスイープ |
| `--kgfm-encoders LIST` | `ngram,transformer` | エンコーダのスイープ |
| `--kgfm-freezes LIST` | `off` | 凍結モードのスイープ。`off,on` を渡すと、各 transformer エンコーダについて「フル fine-tune」「凍結 + 射影頭のみ学習」の両方を回す。`ngram` の `on` 変種は no-op なので暗黙にスキップ |
| `--protocol pooled\|filtered` | pooled | （単体実行時の互換用）kgfm の最終評価プロトコル |
| `--max-filter-tails N` | 50000 | filtered 時の候補語彙上限 |
| `--max-filter-rows N`  | 1000000 | filtered 時の読み取り行上限 |
| `--ultra-gpus "<json>"` | `null` | ULTRA の GPU JSON (H200 では CPU 推奨) |
| `--motif-gpus "<json>"` | `[0]` | MOTIF の GPU JSON |
| `--ultra-ckpt PATH` | `benchmarks/ULTRA/ckpts/ultra_50g.pth` | ULTRA ckpt |
| `--motif-ckpt PATH` | `benchmarks/MOTIF/ckpts/motif_3g.pth` | MOTIF ckpt |
| `--skip-prep` / `--skip-kgfm` / `--skip-ultra` / `--skip-motif` / `--skip-aggregate` | off | 個別ステップのスキップ |
| `--help` | — | flag 一覧 |

`--skip-ultra` は対応する `setup_ultra_env.sh` もスキップします
（conda env 構築は時間がかかるため）。setup と env 構築自体は冪等
なので、複数回呼んでも OK です。

新しいベンチマークを追加するときは `bootstrap_<name>.sh` を `bootstrap_chembl.sh` の構造そのままにコピーし、prep スクリプトと結果ディレクトリ名を差し替えるのが想定パターンです。

### 全コーパスに対する `--max-*` のカバー率

`list_chembl/{train,valid,test}.txt` から参照される ChEMBL TSV の総量と、
そこから推定した triple 数に対する各キャップのカバー率は以下の通りです
（推定法: 全ファイルの合計バイト数 ÷ 158.2 bytes/行。先頭ファイルの
10,000,000 行 / 1.58 GB という実測値で校正）。

| split | 全ファイル合計 | 推定 triple 数 | `bootstrap_chembl.sh` 既定 | `bootstrap_chembl_large.sh` 既定 |
|---|---|---|---|---|
| train | 105.4 GB | ~715M | 50,000 ≈ **0.0070%** | 500,000 ≈ **0.070%** |
| valid |   4.0 GB | ~27.1M |  2,000 ≈ 0.0074% |  10,000 ≈ 0.037% |
| test  |   6.4 GB | ~43.2M |  2,000 ≈ 0.0046% |  10,000 ≈ 0.023% |

両既定値とも全コーパスの **0.1% 未満**で、`large` でも 1 epoch には程遠い
水準です。1% 超を狙う場合は `--max-train` を数百万オーダー（例: 7,000,000
で約 1%）まで上げてください。エンティティ語彙が ULTRA / MOTIF の推論時
メモリに収まる範囲を意識する必要があります。

### より大きいスケールで走らせる: `bootstrap_chembl_large.sh`

`bootstrap_chembl.sh` の薄いラッパで、より正確な評価のためにキャップ・
ステップ数・バッチサイズを大きくしただけのスクリプトです。本体の引数
パーサが後勝ちなので、追加で渡したフラグはこちらの拡張デフォルトを上書き
します。

```bash
bash benchmarks/bootstrap_chembl_large.sh
bash benchmarks/bootstrap_chembl_large.sh --kgfm-encoders ngram --skip-motif
bash benchmarks/bootstrap_chembl_large.sh --max-train 7000000   # ~1% 相当
```

| flag | `bootstrap_chembl.sh` | `bootstrap_chembl_large.sh` |
|---|---|---|
| `--max-train`              | 50,000    | **500,000** |
| `--max-valid`              | 2,000     | **10,000** |
| `--max-test`               | 2,000     | **10,000** |
| `--max-steps`              | 200       | **2,000** |
| `--batch-size`             | 256       | **1,024** |
| `--transformer-batch-size` | (= `--batch-size`) | **64** |
| `--proj-dim`               | (未指定) | **256** |
| `--kgfm-freezes`           | `off`     | **`off,on`** |
| `--max-filter-tails`       | 50,000    | **200,000** |
| `--max-filter-rows`        | 1,000,000 | **5,000,000** |

`bootstrap_chembl_large.sh` の transformer 関連デフォルトの意図:

- **`--transformer-batch-size 64`**: `kgfm.model.DistMultScorer.encode_triple` は
  `(h, r, t)` の 3 系列を 1 回の encoder forward にまとめるため、`B=1024` は
  実質 `3072×128` の BERT-base 入力になり H200 (140GB) でも OOM します。
  B=64 (実質 192 系列) であれば bf16 オートキャスト + AdamW 状態と合わせて
  おおむね 60GB 程度に収まり安定して回ります。
- **`--proj-dim 256`**: `--kgfm-freezes on` セルの学習に必須です。`proj_dim=None`
  かつ encoder 凍結だと `nn.Identity` が射影層になり、optimizer に渡る
  trainable パラメータが 0 になります。ngram (embedding_dim=256) では
  そのまま Identity に縮退するので、結果としてはフル fine-tune セルへの
  追加コストはありません。BERT の fine-tune セルは 768→256 の小さな射影を
  経由する形になります。
- **`--kgfm-freezes off,on`**: 同一 encoder を「フル fine-tune」と
  「frozen + 射影のみ学習」で対比評価します。集計テーブルでは
  `kgfm (transformer)` と `kgfm (transformer, frozen)` の 2 行に分かれます。

下記は手動で個別に実行する場合のレシピです。

## セットアップ

### 1. 上流リポジトリの取得

```bash
# benchmarks/{ULTRA, MOTIF} に git clone
bash benchmarks/setup.sh

# (任意) 上流の pip 依存をインストール
INSTALL_DEPS=1 bash benchmarks/setup.sh

# (任意) ULTRA のデフォルト pretrained ckpt をダウンロード
FETCH_CKPTS=1 bash benchmarks/setup.sh
```

> 注: ULTRA / MOTIF とも、最初から `ckpts/` 以下に複数のチェックポイントを同梱しているため、`FETCH_CKPTS=1` は必須ではありません。

MOTIF の追加チェックポイントが欲しい場合は upstream から手動で取得し、
`run_motif.py --ckpt ...` に渡してください。

### 2. ULTRA 専用 conda 環境の構築 (重要)

ULTRA は CUDA 拡張 `rspmm` を JIT ビルドする都合で、torch / nvcc / 
ホストコンパイラのバージョン整合に強く依存します。新しめの torch では
ビルド自体は通ってもランタイムで `cudaErrorIllegalAddress` を吐くなど、
他用途と同居しづらいため、本ベンチマークでは ULTRA を**独立した conda
環境**で動かします。MOTIF は kgfm と同じ環境で問題なく動作します。

```bash
# 専用 env (デフォルト名: kgfm-ultra) を構築
bash benchmarks/setup_ultra_env.sh

# デフォルトを上書きしたい場合
ENV_NAME=my-ultra TORCH_VERSION=2.4.1 \
    bash benchmarks/setup_ultra_env.sh
FORCE_RECREATE=1 bash benchmarks/setup_ultra_env.sh
```

このスクリプトは以下を行います。

- Python 3.11 + torch 2.5.1+cu121 + 対応する torch_geometric / torch_scatter を導入
- nvcc 12.1 ツールチェーンを `nvidia/label/cuda-12.1.1` 経由で **厳密にピン留め**（素の `nvidia` チャネルだと version pin が無視され 13.x が入ってしまうため）
- gcc 12 を導入（CUDA 12.1 の nvcc は gcc ≥ 13 を拒否する）
- `g++` / `gcc` / `cpp` を env 内に symlink 設置（conda は接頭辞付きで導入するが nvcc は素の名前で解決するため）
- `etc/conda/activate.d/cuda-home.sh` フックで `CUDA_HOME` / `CPATH` / `LIBRARY_PATH` を環境変数に注入

`run_ultra.py` はこの env を自動検出し、その env の Python を subprocess
として呼び出します。手元のシェルは kgfm 用の env のままで OK です。
`--env <name>` または `--python /abs/path/to/python` で上書きできます。

---

## ChEMBL KG の構築

```bash
python benchmarks/prepare_chembl_kg.py \
    --train-list list_chembl/train.txt \
    --valid-list list_chembl/valid.txt \
    --test-list  list_chembl/test.txt \
    --out-dir benchmarks/chembl_kg \
    --max-train 2000000 --max-valid 20000 --max-test 20000
```

kgfm の生 TSV をストリーミングし、`head\trel\ttail` 形式の三つ組ファイル
3 種 (`train.txt`, `valid.txt`, `test.txt`) と `entities.dict`、
`relations.dict`、`stats.json` を出力します。

`--max-*` のキャップは、ChEMBL を全量入れると ULTRA / MOTIF が推論時に
保持しきれないエンティティ語彙が出来てしまうために設定しています。
GPU/CPU メモリに余裕があれば増やしてください。

> ChEMBL のデフォルト分割は **inductive** です。ChEMBL ファイルは
> activity ID で分けられており train ファイルに含まれる ID は他の
> ファイルにほとんど現れないため、strict transductive で語彙を凍結
> すると test がほぼ空になります。`--strict-transductive` を渡すと
> 従来の transductive 設定になります。`stats.json` の `mode` フィールドで
> どちらだったかを確認できます。

---

## 各手法の実行

```bash
# kgfm — chembl train+valid で学習、test で評価 (pooled プロトコル)
python benchmarks/run_kgfm.py --encoder ngram --max-steps 5000

# kgfm — ULTRA / MOTIF と同じ filtered プロトコルで比較
python benchmarks/run_kgfm.py --encoder ngram --max-steps 5000 \
    --protocol filtered \
    --max-filter-tails 50000 \
    --max-filter-rows 2000000

# ULTRA — 構築済み KG でゼロショット推論
# 1) bash benchmarks/setup_ultra_env.sh を済ませてから
# 2) sm_90 (H200) では下記の既知問題のため --gpus null を推奨
python benchmarks/run_ultra.py \
    --ckpt benchmarks/ULTRA/ckpts/ultra_50g.pth

# MOTIF — 構築済み KG でゼロショット推論
python benchmarks/run_motif.py --ckpt benchmarks/MOTIF/ckpts/motif_3g.pth
```

各スクリプトは `benchmarks/results/<method>.json` に JSON で結果を残します。

## 集計

```bash
python benchmarks/aggregate.py --out benchmarks/results/table.md
```

MRR / Hit@1 / Hit@3 / Hit@10 / nDCG を比較する Markdown テーブルが
出力されます。

---

## 動作確認時の実測値 (簡易版)

このマシンでの動作確認時の数値です。あくまで小規模スモーク条件ですが、
パイプラインがエンドツーエンドで動作することの証明になります。

データ: `prepare_chembl_kg.py --max-train 50000 --max-valid 2000 --max-test 2000`
（|E| ≈ 72,669, |R| = 22, inductive モード）

| 手法 | モード | Protocol | MRR | Hit@1 | Hit@3 | Hit@10 | MR | 備考 |
|---|---|---|---|---|---|---|---|---|
| kgfm (ngram, 200 steps) | 学習 | pooled (200) | 0.1106 | — | — | 0.3281 | — | H200 GPU、10.6 秒 |
| kgfm (ngram, 200 steps) | 学習 | filtered (5000) | 0.0371 | — | — | 0.0820 | — | 同 ckpt、`--protocol filtered` |
| ULTRA (zero-shot) | 推論のみ | filtered | **0.1753** | 0.1532 | 0.1993 | 0.2100 | 5702.79 | `kgfm-ultra` env、CPU、約 19 分 |
| MOTIF (zero-shot) | 推論のみ | filtered | **0.1740** | 0.1470 | 0.2003 | 0.2100 | 7650.40 | gnn env、H200 GPU、約 96 秒 |

注意点:

- ULTRA と MOTIF はゼロショットでもほぼ同等の MRR / Hit@10 を出します。
- kgfm の 2 行 (pooled / filtered) は同じチェックポイントの評価です。
  filtered では候補集合が 25 倍 (200 → 5000) に増えるため数字が大きく
  下がります。pooled / filtered の数値を同列に比較しないでください。
- 上記 kgfm は学習 200 step だけの簡易設定です。実用比較には学習ステップ数を増やすとともに、ULTRA / MOTIF と同じ filtered プロトコルに揃えてください (`run_kgfm.py --protocol filtered`)。

---

## 上流リポジトリへの自動パッチ

`run_ultra.py` と `run_motif.py` は対象リポジトリにオンザフライでパッチを
当てます。

- `ChEMBLCustom` データセットクラスを `<repo>/<pkg>/datasets.py` に追記
  （冪等。センチネルコメントで囲み、再実行時に上書き）。
- `torch.load(self.processed_paths[0])` を `weights_only=False` 付きに
  書き換え（PyTorch 2.6+ で既定値が反転し、上流の pickle キャッシュが
  読めなくなったため）。

元に戻したい場合はセンチネルで囲まれた区間を削除し、`weights_only=False`
を消してください。

---

## 既知の問題: H200 (sm_90) での ULTRA `rspmm` カーネル

ULTRA の CUDA 拡張 `rspmm` は H200 (compute capability sm_90) で
`cudaErrorIllegalAddress` を出して落ちます。これは **kgfm-ultra env でも**
再現し、しかも ChEMBL 固有ではなく ULTRA に同梱されている `CoDExSmall`
でも同様です。upstream `rspmm` のカーネル実装と新しめの torch + sm_90 の
組み合わせが原因と見られます。

回避策: **`--gpus null` で CPU 実行**にする。CoDExSmall で公開値どおりの
MRR 0.498 を出すこと、ChEMBL でも上記の通り MRR 0.175 を出して完走
することを確認済みです。CPU でも本ベンチマーク程度のサイズなら 20 分
程度で終わります。

```bash
python benchmarks/run_ultra.py --gpus null
```

---

## ファイル構成

```
benchmarks/
├── README.md               # 本書
├── bootstrap_chembl.sh     # ChEMBL ベンチマークを丸ごと実行 + 記録
├── bootstrap_chembl_large.sh # 上記の薄いラッパ。キャップ等を拡大
├── setup.sh                # ULTRA + MOTIF を git clone
├── setup_ultra_env.sh      # kgfm-ultra conda env を構築
├── prepare_chembl_kg.py    # kgfm TSVs -> entity-ID KG
├── run_kgfm.py             # kgfm を学習・評価
├── run_ultra.py            # ULTRA ゼロショット (kgfm-ultra env を自動使用)
├── run_motif.py            # MOTIF ゼロショット
├── aggregate.py            # JSON 結果を Markdown テーブルに集計
├── ULTRA/                  # 上流クローン (gitignore)
├── MOTIF/                  # 上流クローン (gitignore)
├── chembl_kg/              # 生成 KG (gitignore)
└── results/                # gitignore
    └── chembl/
        ├── 20260507T1234Z/      # bootstrap で作られる実行ごとの記録
        │   ├── meta.json
        │   ├── kgfm.json / ultra.json / motif.json
        │   ├── table.md
        │   ├── run.log + 各ステップログ
        │   └── kgfm_ckpts/
        └── latest -> 20260507T1234Z
```

---

## 細かい注意事項

- **ULTRA dataset シム** — `run_ultra.py` は `ultra/datasets_chembl.py`
  ではなく `ultra/datasets.py` 末尾に直接追記する形で `ChEMBLCustom` を
  注入します（ULTRA は `getattr(ultra.datasets, <ClassName>)` で
  クラスを解決するため）。upstream で `TransductiveDataset` がリネーム
  されたら、`run_ultra.py` 冒頭のテンプレートを編集してください。
- **MOTIF CLI** — MOTIF は ULTRA から派生したコードベースで、
  `script/run.py` のフラグ規約 (`--dataset --gpus --epochs --bpe --ckpt`)
  も同一です。`run_motif.py` も `run_ultra.py` とほぼ並行な構造に
  なっています。
- **GPU メモリ** — ULTRA の NBFNet メッセージパッシングは `|E| · L`
  (L はレイヤ数) でスケールします。OOM になったら `--max-train` を
  下げるか、`run_ultra.py` 内の batch_size を縮めてください。
