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
| `ULTRA` | **ゼロショット**（pretrained ckpt を使用、デフォルト `ultra_50g.pth`） | `kgfm bench prep` が生成する entity-ID KG |
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

`--protocols filtered` で切り替え可能です。filtered では
語彙サイズ・読み取り行数を `--max-filter-tails` と `--max-filter-rows`
で上限指定できるため、multi-TB の chembl 全量でも現実的に走らせられ
ます。

ULTRA / MOTIF はもともと filtered ranking です。集計は
出力テーブルで Protocol 列を表示するので、揃っているかどうかが
ひと目でわかります。

---

## 構成: 手法ごとに独立したコマンド

3 手法はそれぞれ別のコマンドです。共有しているのは **実行ディレクトリ**
だけで、そこに各手法が JSON を落とし、`kgfm report` が拾って 1 枚の表に
します。

| コマンド | 中身 |
|---|---|
| `kgfm bench ...` | kgfm 自身のベンチマーク (prep + sweep) |
| `kgfm-ultra` | ULTRA ゼロショット推論 (別手法・別コマンド) |
| `kgfm-motif` | MOTIF ゼロショット推論 (別手法・別コマンド) |
| `kgfm report` | 実行ディレクトリの結果を拾って `table.md` に集計 |

```
kgfm bench run        prep + sweep を新しい実行ディレクトリに対して実行
kgfm bench prep       生 TSV → entity-ID の ChEMBL KG (ULTRA / MOTIF 用)
kgfm bench sweep      kgfm スイープ (encoders × freezes × protocols)
kgfm bench cell       kgfm セル 1 つ (sweep が内部で起動する)
kgfm bench configs    同梱の YAML 設定ファイルを一覧表示
kgfm viz              チェックポイントの h/t 埋め込みを 2 次元に射影
```

ULTRA / MOTIF は kgfm のモデルにも学習経路にも関与しないため、
`kgfm bench` からは外してあります。`kgfm/bench/` は kgfm 自身の数値を
出すことだけを知っており、ベースラインは `kgfm/baselines/` に分離
されています。集計も特定の手法に属さないので `kgfm report` として独立
させ、「後から ULTRA だけ足して集計し直す」ができるようにしています。

### 典型的な流れ

```bash
kgfm bench run --config benchmarks/config_large.yaml   # 実行ディレクトリを作り、学習・評価
kgfm-ultra --out-dir latest       # 同じディレクトリに ULTRA の結果を追加
kgfm-motif --out-dir latest
kgfm report --out-dir latest      # 全部まとめて 1 枚の表に
```

`--out-dir` は `latest` / タイムスタンプ / パスのいずれでも指定できます
(`kgfm bench run` が `latest` シンボリックリンクを張り替えます)。
`kgfm report --list` で、どの実行にどの手法の結果があるか一覧できます。

個別に再実行したいときも同じ形です。「ULTRA だけ落ちたので再実行」
「集計だけやり直す」に全体を回す必要はありません。

```bash
kgfm-ultra --out-dir latest --gpus null
kgfm bench sweep --out-dir latest --encoders ngram --protocols filtered \
    --max-steps 2000
kgfm report --out-dir latest
```

### モジュール構成

| 場所 | 役割 |
|---|---|
| `kgfm/bench/` | kgfm 側のベンチマーク (`prep` / `sweep` / `cell` / `pipeline` / `config`) |
| `kgfm/baselines/` | ULTRA / MOTIF (`ultra.py` / `motif.py` / 共通処理 `common.py`) |
| `kgfm/report.py` | 集計・学習ログ解析 |
| `kgfm/report_html.py` | HTML / SVG レンダリング |
| `kgfm/runs.py` | 実行ディレクトリと tee ログ (3 手法で共有) |
| `kgfm/envs.py` | conda env 解決 (子プロセス用) |

### シェルスクリプト

repo root に `cd` して 4 つのコマンドを順に呼ぶだけです。`kgfm` /
`kgfm-ultra` / `kgfm-motif` は普通に PATH から解決します。

```bash
# run_chembl.sh の中身はこれだけ
cd "$(dirname "${BASH_SOURCE[0]}")/.."
kgfm bench run --config benchmarks/config_small.yaml "$@"
kgfm-ultra  --out-dir latest
kgfm-motif  --out-dir latest
kgfm report --out-dir latest
```

| スクリプト | 中身 |
|---|---|
| `run_chembl.sh` / `_middle.sh` / `_large.sh` / `_xlarge.sh` | 上記を `--config benchmarks/config_*.yaml` で（`run_chembl.sh` は `config_small.yaml`） |
| `run_chembl_xlarge_compare.sh` | `config_xlarge_compare.yaml`（28 セルのアーキテクチャ比較） |
| `run_chembl_xxlarge.sh` | `config_xxlarge.yaml`（全 674M 行を 1 エポック） |
| `run_chembl_large_2gpu.sh` / `_xlarge_2gpu.sh` | 同じ config に `--nproc 2` を足すだけ |
| `resume_chembl.sh [target]` | `config_xlarge.yaml` + `--resume <target>`（既定 `latest`）。**config は中断した run と一致させる必要があるので、xlarge 以外を再開するときは `--config` を渡してください** |
| `setup_baselines.sh` | `kgfm-ultra --setup` と `kgfm-motif --setup` |
| `setup_baseline_env.sh` | ベースライン用 conda env (`kgfm-ultra`) を構築 |

**環境変数は `setup_baseline_env.sh` の `FORCE_RECREATE=1` だけ**です。
conda env の選択・GPU 指定・ステップのスキップはすべてコマンドの flag
側にあります (`--conda-env` / `--gpus` / `--skip`)。ラッパに渡した追加
flag は `kgfm bench run` (kgfm 側) に届きます。ベースライン側の設定を
変えたいときはコマンドを直接呼んでください。

## ワンショット実行 (推奨)

「セットアップ → データ準備 → 各手法の実行 → 集計」を 1 コマンドで通し
ます。結果は実行のたびに UTC タイムスタンプディレクトリに記録され、最新
へのシンボリックリンク `latest` が更新されます。

```bash
# ChEMBL ベンチマークを丸ごと実行 (smoke スケール)
bash benchmarks/run_chembl.sh

# 細かく制御 (代表例)。追加の flag は kgfm 側 (bench run) に渡ります
bash benchmarks/run_chembl.sh \
    --prep-max-train 200000 --prep-max-valid 5000 --prep-max-test 5000 \
    --max-steps 5000 \
    --protocols filtered

# 一部だけ回す / ベースラインを個別に
kgfm bench run --config small --skip sweep
kgfm-motif --out-dir latest --gpus null
```

実行時に作られる `benchmarks/results/chembl/<UTC タイムスタンプ>[_ラベル]/` には:

```
meta.json                # ホスト / git rev / torch ver / 与えたパラメータ
chembl_kg_stats.json     # 構築した KG の統計のコピー
kgfm_<protocol>_<encoder>[_frozen].json   # kgfm 各セルの評価結果
ultra.json               # ULTRA の評価結果
motif.json               # MOTIF の評価結果
table.md                 # 集計テーブル
report.html              # 比較テーブル + 学習曲線 + メタ情報 (自己完結 HTML)
run.log                  # 統合ログ
<step>.log               # 各ステップ単体ログ
kgfm_ckpts_<encoder>[_frozen]/   # セルごとの kgfm チェックポイント
```

`benchmarks/results/chembl/latest -> <最新タイムスタンプ>` のシンボリック
リンクが常に最新を指します。

### 設定ファイル (YAML)

スケール設定は `benchmarks/config_*.yaml` にあり、`--config` で選びます。

```bash
kgfm bench run --config benchmarks/config_large.yaml
kgfm bench run --config large            # benchmarks/config_large.yaml に解決
kgfm bench configs                       # 同梱の設定ファイル一覧と中身
```

| ファイル | 用途 | 実測ベースの見積 (H200 x1) | 完走済みの run |
|---|---|---|---|
| `config_small.yaml` | スモークテスト | 約 0.5 時間 | `20260808T073317Z_fulltest`（3 手法揃い） |
| `config_middle.yaml` | 中規模 | 約 4.3 時間（6 時間枠） | — |
| `config_large.yaml` | 論文掲載レベル | 約 10.8 時間（12 時間枠） | `20260809T051610Z_chembl_large` |
| `config_xlarge.yaml` | 論文掲載レベル・最大 | 約 21.4 時間（24 時間枠） | `20260808T114948Z_chembl_xlarge` |
| `config_xlarge_compare.yaml` | アーキテクチャ比較（28 セル） | 約 19 時間（学習 15.9h + 評価 3h） | `20260811T230140Z_chembl_xlarge_compare` |
| `config_xxlarge.yaml` | **全 674M 行を 1 エポック**（gte-large + mlp） | 約 10.4 日（2 GPU で 5.2 日） | `20260816T053252Z_chembl_xxlarge` |

「完走済みの run」は `benchmarks/results/chembl/<名前>/` です。それぞれ
`table.md`（結果）・`meta.json`（実行条件）・`commands.jsonl`（実行した
コマンド列）・`cell_*.log`（学習ログ）が揃っているので、数字の出どころは
ここから追えます。`kgfm report --list` でも一覧できます。
**世代をまたぐ比較には 2 つ注意があります**（どの run がどちらかは
`meta.json` の `git_rev` で判別できます）。

- **損失関数の既定**: `kgfm/losses.py` が入ったのはコミット `3e23a11`
  (2026-08-09) です。それ以前の `git_rev` を持つ run
  （`_fulltest` / `_chembl_xlarge` / `_chembl_large` = いずれも `0fabff5`）は
  唯一の目的関数だった **`softmax_ce`** で学習されており、`_xlarge_compare`
  以降は **`contrastive`** です。損失の絶対値はまたいで比較できません
  （MRR / Hit@k は行ごとの単調変換に不変なので比較できます）。
- **データ読み出し順**: ファイルのラウンドロビン読み (`interleave_files`)
  を既定にしたのは 2026-08-25 で、**上表のどの run よりも後**です。つまり
  既存の結果はすべて「1 ファイルを読み切ってから次へ」の順序で学習され、
  評価の候補プールも先頭ファイル寄りに偏っています。今から回す run とは
  評価値を直接比較できません。

```bash
bash benchmarks/run_chembl.sh          # small
bash benchmarks/run_chembl_middle.sh   # middle
bash benchmarks/run_chembl_large.sh    # large
bash benchmarks/run_chembl_xlarge.sh   # xlarge
bash benchmarks/run_chembl_xlarge_compare.sh   # エンコーダ/ヘッド比較
bash benchmarks/run_chembl_xxlarge.sh          # 全行 1 エポック
```

#### 全コーパス 1 エポック (`xxlarge`)

他の config は**スライス**を学習します（各 TSV を順に読み切る実装なので、
25k step × B=512 は 1 worker あたり 3.2M 行しか消費せず、85 ファイル中
約 4 個しか触りません）。`config_xxlarge.yaml` はその反対側で、
`list_chembl/train.txt` の**全行を 1 回**読みます。

実測したコーパスサイズ（`wc -l`、推定ではありません。`_iter_tsv_rows` の
drop は 0 で keep rate = 1.0000）:

| split | ファイル数 | 行数 | サイズ |
|---|---|---|---|
| train | 85 | **674,265,105** | 105.35 GiB |
| valid | 6 | 25,764,240 | 3.99 GiB |
| test | 4 | 40,000,000 | 6.36 GiB |

ファイルは均一ではありません（最大 1.81 GB / 最小 39 KB）。「85 × 1000 万行」
という概算は 26% ずれるので、上の実測値を使っています。

**既定は 28 セル比較で 1 位だった `gte-large` + `mlp` + fine-tune** です
（filtered MRR 0.4641。ngram + linear は 5 位の 0.4331）。実行時間より性能を
優先した設定なので、1 エポックは **10.4 日**（B=256、実測 751 ex/s）です。

B=256 は選択ではなく制約です — `gte-large` の fine-tune は B=384 でも 512 でも
OOM します。0.4641 を出したセルと同じ B なので、この実行はデータ量だけが
違うことになります。

**長さは step 数ではなく `max_epoch: 1.0` で宣言しています**（後述）。
バッチサイズと GPU 数から step 数が自動で決まるので、B=256 / 1 GPU なら
2,633,849 step、`--nproc 2` なら 1,316,925 step で、どちらもちょうど 1 エポックです。
2 GPU はデータを倍にするのではなく**所要時間を半分にします**（train ファイルは
`files[rank::world_size]` で分割され、各 rank が互いに素な半分を読むため）。

```bash
bash benchmarks/run_chembl_xxlarge.sh              # 約 10.4 日
bash benchmarks/run_chembl_xxlarge.sh --nproc 2    # 約 5.2 日

# 当日中に全行を回したい場合（5 位・MRR 0.4331・約 11 h）
# step 数の再計算は不要 — max_epoch が B=512 を吸収します
bash benchmarks/run_chembl_xxlarge.sh \
    --encoders ngram --heads linear --batch-size 512
```

ngram のベースライン行を同じレポートに入れたい場合は、**2 回目のパス**として
同じ run ディレクトリに追加してください（+11 h、全体の約 4%）。`heads` は
グローバルな軸なので `--encoders gte-large,ngram` では ngram に mlp が付き、
**ngram + mlp は 28 セル中の最下位（0.2539）**になってしまい 0.4331 は得られません。

```bash
bash benchmarks/run_chembl_xxlarge.sh --resume latest \
    --encoders ngram --heads linear --batch-size 512
```

#### `max_epoch` — step 数の代わりにエポック数で指定する

`max_steps` の代わりに **`max_epoch`（小数可）** でも学習量を指定できます。
cell 設定なので `defaults:` / `cells:` / `--max-epoch` のいずれでも使えます。

```yaml
defaults:
  max_epoch: 1.0      # ちょうど 1 周
  # max_epoch: 0.25   # コーパスの 1/4
```

```bash
kgfm train --max-epoch 1 --train-list list_chembl/train.txt ...
kgfm bench run --config xxlarge --max-epoch 0.5
```

step 数は次式で導出されます。**`max_epoch` を指定すると `max_steps` は無視されます。**

```
steps = ceil(max_epoch × epoch_examples / (batch_size × nproc × grad_accum))
```

`nproc` が分母に入るので、**GPU 数やバッチサイズを変えても 1.0 は 1 エポックの
まま**です。これまで「2 GPU では max_steps を手で半分にする」必要があったのが
不要になります。

エポックのサイズはファイルの**実行数**から求めます（ChEMBL のファイルは
1.81 GB〜39 KB と不均一なので、1 ファイル分から掛け算すると 26% ずれます）。
105 GiB の読み取りで初回は約 5 分かかりますが、`(パス, サイズ, mtime)` を
キーに `~/.cache/kgfm/rowcounts.json` へキャッシュされるので 2 回目以降は
即時です（`KGFM_ROWCOUNT_CACHE` で場所を変更可）。`max_rows_per_file` と
`row_keep_prob` はエポックサイズに反映されます。

エンコーダごとの 1 エポック所要時間（28 セル比較で実測したスループットから
単純計算）:

| エンコーダ | mode | ex/s | 1 エポック | 2 GPU |
|---|---|---|---|---|
| `ngram` | fine-tune | 17,000 | **11.0 h** | 5.5 h |
| `bert-multilingual` | frozen | 2,434 | 3.2 日 | 38.5 h |
| `bert-multilingual` | fine-tune | 1,498 | 5.2 日 | 2.6 日 |
| `xlm-roberta-large` | fine-tune | 890 | 8.8 日 | 4.4 日 |
| `bge` / `e5` / `gte-large` | fine-tune | 756 | 10.3 日 | 5.2 日 |
| `e5-mistral-7b` | frozen | 192 | **40.6 日** | 20.3 日 |

既定の `gte-large` は 10.4 日、`ngram` なら 11 時間です。どちらを選ぶかは
「性能優先か時間優先か」で、既定は**性能優先**にしてあります。

なお 25k step の large 実行
（`benchmarks/results/chembl/20260809T051610Z_chembl_large/`）では fine-tune
した transformer が過学習しました（train loss は下がり続けるのに valid loss が
step 12.5k で底を打って上昇）。これは行の反復による暗記ではありません
（0.05 エポック未満なので同じ行は二度と現れない）— 到達できた数ファイルへの
適合、つまり母集団シフトでした。全エポックは 85 ファイルすべてに到達するので、
**この失敗モードに対する回避策ではなく直接的な対処**になります。

##### 実行結果（完走済み）

`benchmarks/results/chembl/20260816T053252Z_chembl_xxlarge/` が、この config を
`--nproc 2` で**実際に 1 エポック完走させた**記録です。

| | 値 |
|---|---|
| セル | `gte-large` + `mlp` + fine-tune、B=256 |
| step 数 | 1,316,925（= `max_epoch: 1.0` を 2 GPU で解決した値。337M examples/rank） |
| スループット | 785 ex/s（最終付近の実測） |
| in-loop valid MRR (pooled) | 0.7772 @ step 1,316,920 |
| **最終 test MRR (pooled, 12k)** | **0.5164** (Hit@1 0.4982 / Hit@10 0.5508 / nDCG 0.5734) |
| **最終 test MRR (filtered)** | **0.4410** (Hit@1 0.4173 / Hit@10 0.4869 / nDCG 0.4906) |

**同じセルを 6,000 step だけ回した 28 セル比較では filtered MRR 0.4641** です。
つまり 220 倍のデータを入れても filtered の数字は上がっていません（0.4641 →
0.4410）。ただし**この 2 つは直接比較できません** — 28 セル比較は
`max_filter_tails: 460000`（≈ その prep の \|E\| = 419,252）、xxlarge は
2,000,000（実際に構築された語彙 1,563,335）で、候補語彙の大きさが 3.7 倍
違います。分母が増えれば MRR は下がるので、この差の内訳は不明です。
**同じ語彙で比べたいなら `kgfm eval` で再スコアしてください**（その際は
`--filter-list` を train / valid / test の 3 つとも渡すこと。後述の
「filtered の落とし穴 2 つ」参照）。
pooled 側は両方 `pool_size: 12000` なので比較可能で、0.4929 → 0.5164 と
上がっています。

> **filtered の値は他の config と比較できません。** 全コーパスの |E| は未計測です
> （知るには 674M 行の prep が必要で、それを filtered プロトコルで全部
> 埋め込むのはそもそも不可能）。したがって `max_filter_tails: 2000000` は
> |E| ではなく単に「評価が現実的な時間で終わる上限」です。実際に構築された
> 語彙は 1,563,335 件でこの上限に届いていません — 効いていたのは
> `max_filter_rows: 10000000` の方で、`filter_pairs` が 9,213,329 なので
> ちょうど 1000 万行を読んだところで打ち切られています。真の |E| より
> 候補が少ないぶん **楽観的な値**になります。large / xlarge は
> `max_filter_tails ≈ |E|` に合わせてあるので、xxlarge の filtered は
> この config 内でのみ比較してください。

#### アーキテクチャ比較

`config_xlarge_compare.yaml` はデータ規模を xlarge に固定したまま
`encoders × heads × freezes` を振ります（**28 セル**）。確認済みプリセット
全 8 種 × `heads: [linear, mlp]` × `freezes: [off, on]` で、両方の freeze が
可能なものはすべて両方を実行します（ngram の frozen と 7B の fine-tune は
`cell_specs()` が自動的に落とします）。エンコーダとヘッドの
一覧・プリセット名はトップレベルの
[README.md「エンコーダとヘッド」](../README.md#エンコーダとヘッド)を参照。

```bash
bash benchmarks/run_chembl_xlarge_compare.sh
bash benchmarks/run_chembl_xlarge_compare.sh --encoders ngram,bge-large --heads linear
```

全セルで **`batch_size: 256`** と `proj_dim: 256` を固定しています。B-1 は
負例数そのもの、`proj_dim` はスコア計算の次元なので、セルごとに変えると
アーキテクチャの差と交絡するためです。**256 は選択ではなく制約**で、一番重い
`bge-large` + `mlp` の fine-tune が B=384 でも B=512 でも OOM するために
決まった値です（実際の `kgfm bench cell` で確認。詳細は
`config_xlarge_compare.yaml` 冒頭のコメント）。`e5-mistral-7b` は 14 GB の
初回ダウンロードが発生します。

**結果は
`benchmarks/results/chembl/20260811T230140Z_chembl_xlarge_compare/table.md`
にあります。** filtered MRR の 1 位は `gte-large` + `mlp` + fine-tune で
0.4641、最下位は `ngram` + `mlp` の 0.2539、`ngram` + `linear` は 0.4331 で
5 位です。`config_xxlarge.yaml` が `gte-large` + `mlp` を既定にしているのは
この表が根拠で、ngram のベースライン行を足すときに `--heads linear` を
明示しないといけないのも同じ理由です（`ngram` + `mlp` は 28 セル中最下位）。

#### 2 GPU で動かす

`_2gpu.sh` は large / xlarge と**同じ config**（＝同じスケール）を
`--nproc 2` で回すだけです。config はスケールを記述するもので GPU 台数は
フラグ、という区分に従っているため `config_large_2gpu.yaml` は作りません。

```bash
bash benchmarks/run_chembl_large_2gpu.sh    # config_large.yaml  + --nproc 2
bash benchmarks/run_chembl_xlarge_2gpu.sh   # config_xlarge.yaml + --nproc 2
```

> **2026-08-16 より前の 2 GPU 実行結果は信用できません。** それ以前の
> `--nproc 2` 実行は **DDP の勾配同期が全く効いておらず**、2 つの独立した
> モデルを学習して rank 1 の結果を捨てていました。原因は
> `in_batch_negative_loss` が `DistributedDataParallel` のラッパではなく
> `.module` を呼んでいたことで、`reducer.prepare_for_backward()` を呼ぶのは
> `DDP.forward` だけなので、**勾配の all-reduce が 1 度も起きていません**
> でした（実測: 異なるデータを与えた 2 rank が 5 step 後に全く違う重みに
> なる。`DDP.forward` 経由なら同じ条件でビット単位一致）。副作用として
> rank が同期せず自由走行し、最初の eval barrier で NCCL の watchdog を
> 超えて 9 時間の run が step 131k で落ちる、という形でも現れました。
> 現在は `DistMultScorer.forward` に `return_embeddings=True` を通して
> ラッパ側を呼んでいます。**1 GPU の結果は影響を受けません**（`benchmarks/
> results/chembl/` の run のうち影響があるのは `--nproc 2` で回したものだけ
> で、`meta.json` の `nproc` で判別できます）。以下の記述は修正後の挙動です。

**負例数は GPU 台数で変わりません。** `in_batch_negative_loss` の `[B,B]`
行列は各 rank のローカルなマイクロバッチから作られ、kgfm には埋め込みの
all_gather がどこにもありません。`batch_size: 512` は
`train._resolve_per_device_batch_size` により**1 GPU あたり** 512 になるので、
負例数は 1 GPU 実行と同じ 511 のままです。変わるのはグローバルバッチ
（512 × 2 = 1024）で、1 step あたり 2 倍のサンプルで勾配を平均します。

したがって 2 GPU 実行は 1 GPU 実行の step 単位の再現ではありません。
**負例数と所要時間はそのままで、見るデータ量が 2 倍**になります
（large なら 25,000 step で 12.8M → 25.6M サンプル）。1 GPU と同じ学習量に
したい場合は step を半分にしてください:

```bash
bash benchmarks/run_chembl_large_2gpu.sh --max-steps 12500
```

**2 台目の GPU が増やすのは「データ量」より「ファイル多様性」です。**
ChEMBL の TSV は 1 ファイル 10,000,000 行で、ローダは各ファイルを順に
読み切る実装です。large の 25,000 step × B=512 は 1 worker あたり 3.2M 行
しか消費しないので、**最初の 1 ファイルすら読み終わりません**。
`num_workers=4` なので 1 GPU 実行が触るのは 85 ファイル中およそ **4 個**
（各 32% まで）、2 GPU なら worker が 8 個で **8 個**です。ChEMBL は
activity ID でファイル分割されており 1 ファイル＝1 つのエンティティ集団、
そして train/valid のギャップはまさにその集団シフトなので、2 GPU 目は
スループットではなく**エンティティ多様性**を買っていることになります。

1 GPU のまま多様性を上げたい場合は `max_rows_per_file`（cell 設定）で
深さと広さを交換できます。

```yaml
defaults:
  max_rows_per_file: 500000   # 1 ファイル 50 万行で次へ = 触るファイル数が約 20 倍
```

**プロセスグループのタイムアウトは 2 時間**です（`TrainConfig.dist_timeout_min
= 120`、NCCL の既定は 10 分）。評価とチェックポイント書き出しは rank 0 だけで
走り、その間ほかの rank は `barrier()` で待つので、大きな tail 語彙に対する
filtered 評価は 10 分を普通に超えます。実際に既定の 10 分だと
`20260813T093403Z_chembl_xxlarge` が step 131k で
`Watchdog caught collective operation timeout ... ran for 600068 milliseconds`
を出して死にました（`cell_pooled_gte-large.log` に残っています）。**遅い評価を
クラッシュに変えないための設定**で、性能とは無関係です。

**DDP では HF モデルの `pooler` を凍結しています**（`kgfm/encoders.py`）。
このエンコーダは `last_hidden_state` を自分で pooling するので BERT 系
チェックポイントに付いてくる `pooler`（[CLS] に対する Linear）を一度も
呼びません。すると勾配を受け取らないパラメータが残り、
`find_unused_parameters=False`（既定）の DDP では 2 step 目で
`Expected to have finished reduction in the prior iteration` になって
**落ちます**（実例: `20260816T052044Z_chembl_xxlarge` が
`Parameter indices which did not receive grad: 389 390` で終了）。
`requires_grad=False` にするのが正確な対処で、`add_pooling_layer=False` と
違って重みは state_dict に残るのでチェックポイントの互換性も保てます。

**学習率は変えていません。** 実効バッチが 2 倍なら lr を上げるのが定石ですが、
実測（ngram, 1000 step, global_bs=1024, valid loss / pooled MRR）では

| lr | valid loss | MRR |
|---|---|---|
| 1e-3（既定） | **4.656** | 0.2013 |
| 1.4e-3 | 4.754 | 0.1941 |
| 2e-3 | 4.697 | 0.2070 |

と単調でなく、差も 0.1 nats 未満です（同一設定 3 回の再実行はビット単位で
一致したので、この差はノイズではなく本当に小さいということです）。
そもそも `--lr` は全セルを 1 つの値で上書きしてしまい、既定がエンコーダごと
（ngram 1e-3 / transformer 3e-5）である以上それ自体が不適切です。どうしても
変えるなら `cells: <tag>: lr:` を使ってください。

なお `_2gpu.sh` に `--per-device-train-batch-size` は**渡さないでください**。
`sweep.py` はセルごとに `--batch-size` を送りますが
`--per-device-train-batch-size` は run 全体に 1 回しか送らないので、これを
渡すと `cells: <tag>: batch_size:` で付けたセルごとの差が全部潰れます
（`_2gpu.sh` が `--nproc 2` だけを足す薄いラッパである理由もこれです）。
`--nproc` を変えても per-device のバッチサイズは `batch_size` のままなので、
そもそも渡す必要がありません。

### 所要時間の内訳

見積は推測ではなく、このマシンでの**実測値**から計算しています。

| 実測項目 | 値 |
|---|---|
| ngram 学習 (B=512) | 約 21,000 examples/s / 5.1 GiB |
| transformer 学習 (B=512) | 約 3,014 examples/s / 22.2 GiB |
| ULTRA (CPU) | 1,134 秒 @ n_test=2,000, n_train=50,000 |
| MOTIF (GPU) | 70 秒 @ 同上 |
| entity 数 \|E\| | 50k→72.7k / 200k→231k / **400k→419.3k** / 1M→685k（実測、劣線形） |

| config | ULTRA | MOTIF | kgfm | 合計 | 学習量 |
|---|---|---|---|---|---|
| middle | 2.8 h | 0.2 h | 1.3 h | **4.3 h** | 10.2M |
| large | 7.9 h | 0.5 h | 2.4 h | **10.8 h** | 12.8M |
| xlarge | 15.1 h | 0.9 h | 5.3 h | **21.4 h** | 30.7M |

> **ULTRA が支配的です。** ULTRA / MOTIF のコストは `n_test × n_train` に
> 比例します（クエリごとに全エッジをメッセージパッシングするため）。ULTRA
> は sm_90 の rspmm バグで **CPU 実行**なので、これが全 config で最大の項に
> なります。ULTRA を外す（`kgfm bench run` + `kgfm-motif` だけ回す）と壁
> 時計時間はおおむね半分以下になり、その分を kgfm の学習に回せます。
>
> 見積には枠に対して 20〜30% の余裕を持たせています。ULTRA のスケーリング
> は**測定点 1 つからの線形外挿**なので、超線形だった場合の保険です。

### 論文掲載レベルの設定 (large / xlarge)

README の「test サンプル数と評価指標への影響」節のガイドラインに従って
います。

- `prep_max_test >= 10,000` — MRR の標準誤差が約 ±0.004 に収まる水準
- `n_eval_triples = prep_max_test` — kgfm もベースラインと同じ件数を評価する
  （既定の 5,000 打ち切りを無効化）
- `max_filter_tails ≈ |E|` — filtered の候補語彙を切り詰めない
- `freezes: ["off", "on"]` — フル fine-tune と凍結エンコーダの ablation

**バッチサイズは全セル共通の 512** です（`config_large.yaml` /
`config_xlarge.yaml` の `defaults: batch_size: 512` で、`cells:` 側で
上書きしていません）。`max_steps` が ngram と transformer で同じデータ量を
意味するので、曲線を step 単位でそのまま比較できます。B は同時に
**モデル側のハイパーパラメータ**でもあります — `in_batch_negative_loss` は
バッチ内の他の tail すべてを負例に使うため、B-1 が負例数です。

（`config_xlarge_compare.yaml` だけは B=256 です。1024 次元の大きい
エンコーダを 8 種並べるので、一番重いセルが通る値まで下げる必要があり
ました。前述の「アーキテクチャ比較」節を参照。）

512 は H200 上での実測から決めています（1 学習ステップあたり）:

| encoder | B=256 | B=384 | B=512 | B=768 |
|---|---|---|---|---|
| transformer | 1,127 ex/s / 12.3 GiB | 2,610 ex/s / 17.2 GiB | **3,014 ex/s / 22.2 GiB** | 2,919 ex/s / 32.2 GiB |
| ngram | 17,690 ex/s / 5.1 GiB | — | 約 21,000 ex/s | 21,904 ex/s (B=4096) / 5.2 GiB |

transformer は 512 が最良で、768 は**遅くなるうえメモリを 45% 多く使います**。
ngram も 512 で頭打ち (B=4096 の 5% 以内) です。22 GiB / 143 GiB なので、
tail 語彙全体をエンコードする評価パスにも十分な余裕が残ります。

> B=256 から 512 に上げると、同じ `max_steps` で**学習データが 2 倍**に
> なり、しかも**所要時間は約 25% 減ります**（スループットが 2.7 倍のため）。
> 負例も 255 → 511 に増えます。

優先順位は **デフォルト → `defaults:` → `cells:` → コマンドラインの flag** です。
ファイルに書いた値をその場で上書きできます。

```bash
bash benchmarks/run_chembl.sh --max-steps 1000   # config_small.yaml の max_steps: 200 を上書き
```

#### 設定ファイルは 2 階層

設定には性質の違う 2 種類があるので、ファイルもその形にしています。

| 階層 | 何を書くか | 例 |
|---|---|---|
| トップレベル | **run 全体**の設定。セルごとに変えられない | `prep_max_train`, `encoders`, `heads`, `freezes`, `protocols`, `nproc` |
| `defaults:` | **全セル共通**のセル設定 | `max_steps`, `batch_size`, `proj_dim` |
| `cells: <tag>:` | **そのセルだけ**の設定 | `transformer: {batch_size: 64}` |

どちらの階層に属するかは `kgfm/bench/config.py` の `CELL_FIELDS` が唯一の
定義で、間違った階層に書くと「どこに書くべきか」を示すエラーになります。

`<tag>` は `<encoder>[_<head>][_frozen]`（`sweep.cell_tag` と同じ。`_<head>` は
`heads` が 2 つ以上のときだけ付く）で、`encoders × heads × freezes` が実際に
生成するセルでなければエラーになります。

> **`prep_max_*` は kgfm の学習量ではありません。** これは `kgfm bench prep`
> が作る entity-ID KG（ULTRA / MOTIF が読むもの）の行数上限です。kgfm 自身は
> `train_list` の生 TSV を直接ストリームし、この KG を読みません。旧名の
> `max_train` は run 全体の上限のように見えて紛らわしかったため、接頭辞を
> 付けました（旧名を書いた config は「代わりにこれを使え」というエラーに
> なります: `config._RENAMED`）。2 つのベースラインは**同じ** KG を読むので、
> キャップは手法ごとではなく 1 組です。**kgfm の学習量を決めるのは
> `max_steps` / `max_epoch`** です。

```yaml
prep_max_train: 250000                 # run 全体
encoders: [ngram, transformer]
freezes: ["off", "on"]

defaults:                         # 全セル
  max_steps: 25000
  batch_size: 512
  proj_dim: 256

cells:                            # このセルだけ
  transformer:
    encoder_weight_decay: 0.01    # fine-tune 側だけ過学習するので
    head_weight_decay: 0.0
  transformer_frozen:
    head_dropout: 0.1             # head しか学習しないセル
```

**コマンドラインの flag は「全セル共通の単一オーバーライド」**で、`cells:` の
後に適用されるため per-cell 設定にも勝ちます。セルごとに変えたいものは
必ず設定ファイル側に書いてください。

```bash
kgfm bench run --config large --batch-size 64   # 全セルが 64 になる
```

実際に解決された値は run.log の先頭にセルごとに 1 行ずつ出ます:

```
cell ngram               max_steps=25000 batch_size=512 eval_every=2500 proj_dim=256
cell transformer         max_steps=25000 batch_size=512 eval_every=2500 proj_dim=256 encoder_weight_decay=0.01 head_weight_decay=0.0   <- cells:
cell transformer_frozen  max_steps=25000 batch_size=512 eval_every=2500 proj_dim=256 head_dropout=0.1   <- cells:
```

**未知のキーは黙って無視せずエラー**にして、有効なキー一覧を表示します。
セル設定をトップレベルに書いた場合・run 設定を `defaults:` に書いた場合・
`cells:` のタグを打ち間違えた場合も、それぞれ「どこに書くべきか」を示す
エラーになります。使った設定ファイルのパスは `meta.json` に記録されます。

> エンコーダごとにバッチサイズを変えていた `transformer_batch_size` /
> `--transformer-batch-size` は廃止しました。`cells: transformer:
> batch_size:` が一般形です。

> **YAML の落とし穴**: `freezes: [off, on]` の `off` / `on` は YAML では
> **真偽値** (False / True) として解釈されます。同梱ファイルでは
> `["off", "on"]` とクォートしていますが、ローダ側でも真偽値を文字列に
> 戻すので、どちらの書き方でも正しく動きます。

新しいスケールを足すときは YAML を 1 枚置くだけです。コードの変更は
要りません。

主な flag (`kgfm bench run --help` に全一覧):

| flag | デフォルト (small) | 用途 |
| --- | --- | --- |
| `--config PATH` | なし | YAML 設定ファイル (`benchmarks/*.yaml`) |
| `--conda-env NAME` | `kgfm` | 子プロセスを動かす conda env |
| `--prep-max-train N` / `--prep-max-valid N` / `--prep-max-test N` | 50000 / 2000 / 2000 | ChEMBL prep の三つ組キャップ |
| `--max-steps N` | 200 | kgfm の学習ステップ数 |
| `--batch-size N` | 256 | kgfm の学習バッチサイズ (per-device・**全セル**)。セルごとに変えたいときは YAML の `cells: <tag>: batch_size:` を使う（専用フラグ `--transformer-batch-size` は廃止） |
| `--max-epoch F` | なし | step 数の代わりにエポック数で指定（小数可）。指定すると `--max-steps` は無視 |
| `--heads LIST` | `auto` | 射影ヘッドのスイープ (`auto,identity,linear,mlp,residual_mlp` から選択)。2 つ以上でセルタグに `_<head>` が付く |
| `--loss NAME` / `--loss-temperature F` | なし (= `contrastive` / 0.1) | 学習目的関数。詳細はトップレベル README |
| `--proj-dim N` | （未指定 = `None`） | DistMult 直前に学習可能な `Linear` 射影を挿入。`--freezes on` を使う場合は必須（凍結 LM + `proj_dim=None` だと学習可能パラメータが 0 になる）。ngram に対しても同値であれば `nn.Identity` に縮退するため無害 |
| `--protocols LIST` | `pooled,filtered` | 最終評価プロトコルのスイープ |
| `--encoders LIST` | `ngram,transformer` | エンコーダのスイープ |
| `--freezes LIST` | `off` | 凍結モードのスイープ。`off,on` を渡すと、各 transformer エンコーダについて「フル fine-tune」「凍結 + 射影頭のみ学習」の両方を回す。`ngram` の `on` 変種は no-op なので暗黙にスキップ |
| `--nproc N` | 1 | kgfm セル 1 つあたりの GPU 数。>1 で torchrun 起動 |
| `--max-filter-tails N` | 50000 | filtered 時の候補語彙上限 |
| `--max-filter-rows N`  | 1000000 | filtered 時の読み取り行上限 |
| `--skip STEP` | なし | ステップのスキップ (`prep` / `sweep`)。`prep` は「既存 KG を再利用」の意味 |
| `--resume [TARGET]` | なし | 中断した run の再開 (既定 `latest`) |

ベースラインの flag (`--gpus` / `--ckpt` / `--config` など) は
`kgfm-ultra --help` / `kgfm-motif --help` を参照してください。

手法を 1 つ足すときは、kgfm 系なら `kgfm/bench/` に、外部手法なら
`kgfm/baselines/` に `Baseline` を 1 つ定義してコマンドを生やします。
`kgfm report` は実行ディレクトリを走査するだけなので、新しい JSON を
置けば自動的に表に載ります。

### 全コーパスに対する `--max-*` のカバー率

`list_chembl/{train,valid,test}.txt` から参照される ChEMBL TSV の総量
（`wc -l` による**実測値**。前節「全コーパス 1 エポック」の表と同じもの）に
対する、`prep_max_*` のカバー率です。

| split | 全ファイル合計 | 行数（実測） | `config_small.yaml` | `config_large.yaml` |
|---|---|---|---|---|
| train | 105.35 GiB | 674,265,105 | 50,000 ≈ **0.0074%** | 250,000 ≈ **0.037%** |
| valid |   3.99 GiB |  25,764,240 |  2,000 ≈ 0.0078% |  10,000 ≈ 0.039% |
| test  |   6.36 GiB |  40,000,000 |  2,000 ≈ 0.0050% |  10,000 ≈ 0.025% |

**繰り返しますがこれは `kgfm bench prep` が作る entity-ID KG のカバー率で、
kgfm 自身の学習量ではありません**（kgfm は `train_list` の生 TSV を直接
ストリームします）。つまりこの表が言っているのは「**ULTRA / MOTIF が見ている
グラフは全コーパスの 0.1% 未満**」ということです。ベースライン側で 1% 超を
狙う場合は `--prep-max-train` を数百万オーダー（例: 6,700,000 で約 1%）まで
上げてください。ただしエンティティ語彙が ULTRA / MOTIF の推論時メモリに
収まる範囲を意識する必要があります（|E| はほぼ劣線形で伸びます —
50k→72.7k / 200k→231k / 400k→419.3k / 1M→685k）。

kgfm 側の学習量は `max_steps` × `batch_size` で、`config_large.yaml` の
25,000 × 512 = 12.8M examples ≈ **0.0019 エポック**です。全 1 エポックを
回すのが `config_xxlarge.yaml` です。

### `config_large.yaml` の狙い

```bash
bash benchmarks/run_chembl_large.sh
bash benchmarks/run_chembl_large.sh --encoders ngram --skip motif
bash benchmarks/run_chembl_large.sh --prep-max-train 6700000   # ベースラインの KG を全体の ~1% に
```

実ファイル（`config_small.yaml` / `config_large.yaml`）の値です。

| 設定 | `config_small.yaml` | `config_large.yaml` |
|---|---|---|
| `prep_max_train`   | 50,000    | **250,000** |
| `prep_max_valid`   | 2,000     | **10,000** |
| `prep_max_test`    | 2,000     | **10,000** |
| `max_steps`        | 200       | **25,000** |
| `batch_size`       | 256       | **512**（全セル共通） |
| `proj_dim`         | (未指定)  | **256** |
| `freezes`          | `["off"]` | **`["off", "on"]`** |
| `n_eval_triples` / `pool_size` | (既定 5,000) | **10,000** |
| `max_filter_tails` | 50,000    | **310,000** |
| `max_filter_rows`  | 1,000,000 | **10,000,000** |

各値の意図:

- **`batch_size: 512` は全セル共通**です。エンコーダごとにバッチサイズを
  変える専用フィールド `transformer_batch_size` は廃止済みで、必要なら
  `cells: transformer: batch_size:` と書きます。`config_large.yaml` は
  あえて分けていません — `max_steps` が ngram と transformer で同じデータ量を
  意味し、学習曲線を step 軸でそのまま重ねられるようにするためです。512 が
  H200 上の実測最良値である根拠は前節の表にあります。
  なお `kgfm.model.DistMultScorer.encode_triple` は `(h, r, t)` を 1 回の
  encoder forward にまとめるため、**B=512 は実質 1,536 系列**です。これが
  transformer セルのメモリを決めているので、B を上げるときはこの 3 倍を
  意識してください。
- **`proj_dim: 256`**: `freezes` に `"on"` が入るとき**必須**です。
  `proj_dim=None` かつ encoder 凍結だと `DistMultScorer.proj` が
  `nn.Identity` に縮退し、optimizer に渡る trainable パラメータが 0 に
  なります（`train()` はこれを検出して DDP のラップを丸ごと飛ばします）。
  ngram (embedding_dim=256) では 256 を渡してもそのまま Identity に縮退
  するので、fine-tune セルへの追加コストはありません。transformer の
  fine-tune セルは 768→256 の小さな射影を経由する形になります。
- **`freezes: ["off", "on"]`**: 同一 encoder を「フル fine-tune」と
  「frozen + 射影のみ学習」で対比評価します。集計テーブルでは
  `kgfm (transformer)` と `kgfm (transformer, frozen)` の 2 行に分かれます。
  `ngram` の `"on"` は「凍結できる LM が無い」ので `cell_specs()` が
  自動的に落とします。
- **`cells:` で fine-tune セルだけに `encoder_weight_decay: 0.01`** を
  かけています。この run（`benchmarks/results/chembl/20260809T051610Z_chembl_large/`）
  で、fine-tune した transformer は train loss が 2.25 → 1.97 と下がり続ける
  一方 valid loss が step 12.5k を底に上昇したのに対し、head しか学習しない
  frozen セルでは同じ現象が起きなかったためです。frozen セルの方は
  `head_dropout: 0.1` を持ちます。

## セットアップ

### 1. 上流リポジトリの取得

各ベースラインが自分の clone を持ちます（冪等）。

```bash
# benchmarks/{ULTRA, MOTIF} に git clone + conda env の依存チェック
bash benchmarks/setup_baselines.sh

# 片方だけ / ckpt もダウンロード
kgfm-ultra --setup
kgfm-ultra --fetch-ckpt
```

> 注: ULTRA / MOTIF とも、最初から `ckpts/` 以下に複数のチェックポイントを同梱しているため、`--fetch-ckpt` は必須ではありません。

MOTIF の追加チェックポイントが欲しい場合は upstream から手動で取得し、
`--motif-ckpt <path>` に渡してください。

### 2. conda 環境 (`kgfm` に一本化)

env は **2 つ**です（以前は 3 つ: `kgfm` / `gnn` / `kgfm-ultra`）。

| env | 用途 | 構築 |
|---|---|---|
| `kgfm` | kgfm 自身の学習・評価 | `pip install -e '.[transformer]'` |
| `kgfm-ultra` | ULTRA / MOTIF (ベースライン) | `bash benchmarks/setup_baseline_env.sh` |

`gnn` env はもう使いません。ベースラインを別 env に置くのは、両者が
`rspmm` CUDA 拡張を env の torch に対して JIT ビルドするため
`torch.version.cuda` に一致する nvcc が必要だからです。`kgfm` env は最新
torch を追いかけていて CUDA ツールチェーンを持たず、そこにピン留めすると
学習側の torch を巻き添えにしてしまいます。ベースラインは torch 2.5.1 +
CUDA 12.1 で自己完結した env に置く方が安全です。

**どちらの env で動かしても指標は完全に一致することを実測済み**です
(ULTRA: MRR 0.175287 / MOTIF: 0.174047 が `kgfm`・`gnn`・`kgfm-ultra` で
一致)。`kgfm-ultra` を既定にしているのは、`kgfm` env をベースライン依存
(とりわけ最新 torch に対する `torch_scatter` のソースビルド) から
解放できるためです。`kgfm/envs.py` が
`<conda base>/envs/<env 名>/bin/python` を明示的に解決するため、シェルで
どの env を activate しているかに依存しません（cron から起動しても
activate 済みシェルから起動しても同じ挙動になります。env 名は
`--conda-env` で上書き可能）。副次的な効果として、`meta.json` に
記録される torch バージョンが 1 つに定まり、実行条件が一意になります。

ベースライン用 env は次のコマンドで構築します（冪等）。

```bash
bash benchmarks/setup_baseline_env.sh
FORCE_RECREATE=1 bash benchmarks/setup_baseline_env.sh   # 作り直す
```

torch 2.5.1+cu121 / nvcc 12.1 / gcc 12 などのバージョンは**選択肢ではなく
ピン**（動作が確認された組み合わせ）なので、スクリプト冒頭の定数に直書き
しています。変えるときはそこを編集してください。

このスクリプトは Python 3.11 + torch 2.5.1+cu121 + nvcc 12.1 (厳密ピン)
+ gcc 12 + torch_geometric / torch_scatter を一式そろえます。**`kgfm` env
には一切触れません。** `kgfm-ultra --setup` / `kgfm-motif --setup` は実行前に
import 可否をチェックし、足りなければ上記コマンドを案内します
（チェックするだけで、勝手にインストールはしません）。

> **GPU / env についての注意**: ULTRA / MOTIF の `rspmm` は env の torch に
> 対して JIT ビルドされるため、`torch.version.cuda` に一致する nvcc と、
> その nvcc が受け付ける gcc が env 内に必要です。`kgfm` env には CUDA
> ツールチェーンが無いため、現状の役割分担は次の通りです。
>
> | 手法 | env (既定) | GPU |
> |---|---|---|
> | ULTRA | `kgfm-ultra` | CPU (`--gpus null`、既定)。後述の sm_90 バグのため |
> | MOTIF | `kgfm-ultra` | GPU 必須 |
>
> **MOTIF は CPU では動きません。** `models.py` が `HypergraphLayer` に
> config の `use_triton` を渡しておらず、この層は常に Triton カーネル
> (`torch.cuda.set_device` を呼ぶ) を使うためです。`--gpus null` を渡すと
> `kgfm-motif` が実行前に理由を表示して停止します。
>
> MOTIF は **torch と一致する nvcc を持つ env** が必須なので、
> `kgfm-ultra` (torch 2.5.1+cu121 + nvcc 12.1 + CUDA 版 torch_scatter) で
> 動かします。ULTRA も同じ env が既定です (CPU 実行なので必須ではありま
> せんが、`kgfm` env をベースライン依存から解放できるため)。
> `--conda-env` で上書きできます。

---

## ChEMBL KG の構築

```bash
kgfm bench prep --out-dir latest \
    --train-list list_chembl/train.txt \
    --valid-list list_chembl/valid.txt \
    --test-list  list_chembl/test.txt \
    --kg-dir benchmarks/chembl_kg \
    --prep-max-train 2000000 --prep-max-valid 20000 --prep-max-test 20000
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
> 語彙を凍結する transductive 設定になります。`stats.json` の `mode` フィールドで
> どちらだったかを確認できます。

---

## 各手法の実行

既存の実行ディレクトリ (`--out-dir`) を指定して、手法ごとに実行します。

```bash
# kgfm — chembl train+valid で学習、test で評価 (pooled プロトコル)
kgfm bench sweep --out-dir latest --encoders ngram --protocols pooled \
    --max-steps 5000

# kgfm — ULTRA / MOTIF と同じ filtered プロトコルで比較
kgfm bench sweep --out-dir latest --encoders ngram --protocols filtered \
    --max-steps 5000 --max-filter-tails 50000 --max-filter-rows 2000000

# ULTRA — 構築済み KG でゼロショット推論
# 1) bash benchmarks/setup_baseline_env.sh を済ませてから
# 2) sm_90 (H200) では下記の既知問題のため既定の --gpus null のままで
kgfm-ultra --out-dir latest --ckpt benchmarks/ULTRA/ckpts/ultra_50g.pth

# MOTIF — 構築済み KG でゼロショット推論
kgfm-motif --out-dir latest --ckpt benchmarks/MOTIF/ckpts/motif_3g.pth
```

結果は実行ディレクトリ内に `kgfm_<protocol>_<encoder>.json` /
`ultra.json` / `motif.json` として残ります。

## 集計

```bash
kgfm report --out-dir latest      # table.md と report.html を書き出し
kgfm report --list                # どの run にどの手法の結果があるか一覧
kgfm report --out-dir latest --out -        # 表は標準出力のみ (HTML は書く)
kgfm report --out-dir latest --no-html      # Markdown 表だけ
kgfm report --out-dir latest --html-out /path/to/x.html
```

出力は 2 種類です。

| ファイル | 内容 |
|---|---|
| `table.md` | 比較テーブルのみ (メモに貼る用) |
| `report.html` | 同じ比較テーブル + **学習曲線** + 手法ごとのパラメータ + 実行メタ情報を 1 ページにまとめた自己完結 HTML |

`report.html` のプロットは、実行ディレクトリ内の `cell_*.log` と各手法の
JSON を**後から解析**して描いています（学習ループのログ形式に依存）。
そのため過去に完了した run に対しても後から生成できます。

レポートは**手法ごとにセクション**が分かれ、各セクション内に
**プロトコルごとのサブセクション**（最終指標と設定）と、**学習曲線**が
1 つ入ります。学習曲線がプロトコル別ではなくセクションに 1 つなのは、
セルが学習するのは 1 回だけで各プロトコルは同じチェックポイントを
再スコアするだけだからです。**手法をまたぐ比較は最後**にまとめてあります。

```
（実行メタ情報 + この run を作ったコマンド一覧）
kgfm (ngram)              ← 手法ごと
  （この手法に紐づくコマンド）
  protocol: filtered        ← プロトコルごとの最終指標・設定
  protocol: pooled
  embedding space           ← h/t 埋め込みの 2 次元射影
  training                  ← 学習曲線 (プロトコル共通)
kgfm (transformer)
  ...
ULTRA / MOTIF
  protocol: filtered
Cross-method comparison   ← 手法横断は最後
  比較テーブル / 最終指標の棒グラフ / セル横断の学習曲線
Prepared KG / Run parameters
```

### 実行コマンドの記録

実行ディレクトリに書き込むコマンドは、すべて `commands.jsonl` に追記され
ます（時刻・コマンド・cwd、シェルラッパ経由なら親コマンドも）。レポートは
これを 2 か所に表示します。

- 冒頭: **この run を作ったトップレベルのコマンド**一覧
  (`kgfm bench run ...` / `kgfm-ultra ...` / `kgfm viz ...` / `kgfm report ...`)
- 各手法セクション: **その手法が起動した子プロセスのコマンド**
  (`kgfm bench cell ...`、ULTRA/MOTIF の `script/run.py ...`、`kgfm viz ...`)

`kgfm report` 自身も記録されますが、**描画後**に追記するため、そのレポート
自身を生成したコマンドは次回の描画から出ます。`--resume` でスキップだけ
して何もしなかった呼び出しは記録しません（同じ行が並ぶだけで情報がない
ため）。

コマンド記録はこの機能の導入後に実行されたものだけが対象です。それ以前に
作られた run では該当セクションは出ません。

**各手法セクション内の埋め込み空間 (`embedding space`)**

学習済みチェックポイントで entity 文字列をエンコードし、2 次元に射影した
散布図です。`kgfm viz` が生成し (`embeddings_<tag>.json`)、`kgfm bench run`
の `viz` ステップが各セルに対して自動実行します。

```bash
kgfm viz --ckpt <run>/kgfm_ckpts_ngram/best.pt --reducer umap
kgfm bench run --viz-reducer pca --viz-max-points 2000
kgfm bench run --skip viz          # 射影を飛ばす
```

- **点は文字列で重複排除**します。コーパスは同じ entity を何度も含むので、
  triple 単位で打つと頻出 entity の重みが増えるだけで情報は増えません。
- **head と tail を均等にサンプリング**します。`DistMultScorer` は h/t を
  正規化しますが r はしないうえ、h と t はスコア中で役割が違うため、
  両者の雲が分離するかどうか自体が結果です。
- 既定は **4000 点** (h 2000 + t 2000)。`--viz-max-points` で変更できます。
  レポートを軽く保つため数千点に抑えています。
- 色分けは **role (h/t) / relation (`rel_text`) / RDF node type** の 3 通り。
  いずれも**学習時に一切見ていないラベル**（kgfm はテキスト列しか読まない）
  なので、まとまりが見えたらそれは学習された構造です。値が 1 種類しかない
  ラベルは自動的にスキップします。
- 次元削減は `--reducer {auto,pca,umap}`。`auto` は umap-learn があれば
  UMAP、無ければ PCA (torch の SVD、追加依存なし)。PCA のときは寄与率も
  レポートに出ます（実測 43.5% + 34.7% = 78.2%）。

**各手法セクション内の学習曲線**

| プロット | 何を見るか |
|---|---|
| loss — train vs valid | 学習の進行と過学習。同一軸で直接比較 |
| validation metrics (MRR / Hit@10 / nDCG) | 損失ではなく実際の順位性能の推移 |
| generalisation gap (valid − train) | 過学習の開始点。上昇し始めたら学習データに適合し始めた合図 |
| gradient norm (pre-clip) | 勾配の爆発・消失。`--grad-clip` が毎ステップ効いていないかの確認 |
| throughput (examples/s) | dataloader の詰まり・GPU 利用効率 |

**Cross-method comparison セクション内 (セル横断)**

| プロット | 何を見るか |
|---|---|
| train loss by **examples seen** | バッチサイズが違うセル同士の公平な比較 |
| valid loss by step | 汎化性能の比較 |
| validation MRR by step | 最終指標での比較 |

> セル横断の学習曲線は x 軸を **examples seen**（= step × バッチサイズ）に
> しています。同梱の config はどれも全セル同じ `batch_size` なので現状は
> step 軸と比例しますが、`cells: <tag>: batch_size:` でセルごとに変えた
> run では step 軸に重ねると見たデータ量が違うものを並べることになるため、
> examples 軸を既定にしています。

**Cross-method comparison セクション内 (最終指標)**

MRR / Hit@1 / Hit@3 / Hit@10 / nDCG の
グループ棒グラフを、**protocol ごとに別チャート**として出します。pooled と
filtered を 1 枚に混ぜると、表であえて避けているクロスプロトコル比較を
誘発するためです（同一 protocol 内でも kgfm は tail 方向のみ、
ULTRA / MOTIF は head+tail 平均である点は残ります）。

### train / valid の損失比較

学習ループは元々 valid 側では MRR などの指標しか計算しておらず、train の
loss と単位が違うため同じ軸に載せられませんでした。そこで in-loop 評価の
たびに **valid の loss** も計算してログに出すようにしています
(`--valid-loss-batches`、既定 10 バッチ、0 で無効)。

> **バッチサイズを train と揃えて計測**しています。この loss は in-batch
> negative に対する softmax なので、値のスケールが負例数 = バッチサイズで
> 決まります。eval 用の小さいバッチ (既定 `max(64, B//2)`) で測ると valid
> が理由もなく良く見えてしまうため、train と同じバッチサイズで測ります。

これにより両者を同一軸で比較でき、差 (generalisation gap) がそのまま
読めます。report では最終ステップの `valid − train` も併記します。

> **検証の頻度は自動で決まります。** `kgfm.train` 単体の既定 `--eval-every`
> は 1000 で、200 step の smoke run では検証が一度も走らず曲線が空でした。
> ベンチマーク側では未指定なら `max_steps // 10`（loss は `max_steps // 20`
> 上限）を使い、run の長さによらず 10 点程度の曲線が残るようにしています。
> 明示的に `--eval-every` / `--log-every` を渡せばそちらが優先されます。

### グラフのバックエンド

`--charts {auto,plotly,matplotlib,svg}` で切り替えます (既定 `auto`)。

| backend | 出力 | サイズの目安 | 備考 |
|---|---|---|---|
| `plotly` | インタラクティブ (hover / zoom / 系列の表示切替) | 約 4.9 MB | plotly.js を**ページに 1 回だけ**インライン展開 |
| `matplotlib` | 静的 SVG (ベクタ) | 約 100 KB | 印刷・貼り付け向け |
| `svg` | 内蔵の手書き SVG | 約 17 KB | 追加依存なしのフォールバック |

`auto` は plotly → matplotlib → 内蔵 SVG の順に利用可能なものを選びます。
plotly / matplotlib は optional extra なので、未インストールでも report は
内蔵 SVG で生成されます。

```bash
pip install -e '.[report]'          # matplotlib + plotly
kgfm report --out-dir latest --charts matplotlib   # 軽量な静的版
```

いずれのバックエンドでも**外部アセット参照はゼロ**で、ファイル 1 つを
そのままブラウザで開けます (plotly 版もオフラインで動作します)。

集計は実行ディレクトリ内の `kgfm_*.json` / `ultra.json` / `motif.json` を
走査するだけなので、あとから手法を足して再実行しても構いません。

なお `report.html` の比較テーブルでは**最良値の強調表示をしていません**。
kgfm の pooled / filtered、ULTRA / MOTIF の head+tail 平均、`n_eval` の
違いから行同士は直接比較できないためで、列の最大値を太字にすると
まさにその誤読を誘発します。

MRR / Hit@1 / Hit@3 / Hit@10 / nDCG を比較する Markdown テーブルが
出力されます。

---

## test サンプル数 (`n_eval`) と評価指標への影響

`kgfm report` が出力するテーブルにある `n_eval` 列は、各手法が実際に評価に
使った test 三つ組数です。**手法ごとに収集方法が異なるので、数字を
そのまま横並びで比較しないでください。**

### `n_eval` の中身 (手法ごと)

| 手法 | `n_eval` の定義 | 上限 |
|---|---|---|
| `kgfm` | **tail 方向のみ**にストリーミング評価したカウント (`kgfm/eval.py` の `n`) | `--n-eval-triples` (既定 **5000**) で打ち切り |
| `ULTRA` | `script/run.py` の `test()` に渡る `len(test_triplets)` = `test.txt` の全行数 | `--prep-max-test` (`kgfm bench prep`) |
| `MOTIF` | 同上 | 同上 |

ULTRA / MOTIF は head 方向と tail 方向の両方で順位付けして平均するため、
MRR / Hit@K の **分母は実質 `2 × n_eval`** です。kgfm は tail 片方向のみ
なので分母は `n_eval` そのものになります。

### サンプリングが指標に与えるバイアス

- **標本誤差 (共通)** — `n_eval` が 2,000 オーダーだと信頼区間はかなり
  広く、MRR ≈ 0.10〜0.20 のレンジで標準誤差は ±0.005〜±0.010 (95%CI
  でおおよそ ±0.01〜±0.02) 程度です。**手法間の差が 1〜2 ポイント以内
  なら有意とは主張できません。** `large` プリセット
  (`--prep-max-test 10000`) ならおおむね √5 倍狭まります。
- **kgfm の打ち切り (`--n-eval-triples`)** — 既定は
  **5,000** です。`large` プリセットのように `--prep-max-test 10000`
  で test ファイルを大きくしても、kgfm 側は 5,000 で評価を打ち切ります
  (ULTRA / MOTIF は test 全件を評価)。test 全件で kgfm を回したい場合は
  `--n-eval-triples >= --prep-max-test` を明示してください。なお
  `small` プリセットの既定 (`--prep-max-test 2000 < 5000`) では test
  ファイルが先に尽きるため、この打ち切りは発動しません。
- **kgfm 評価対象のシャッフル** — `kgfm/eval.py` の
  `StreamingTripleDataset` はファイル順をシャッフルしバッファ内も
  shuffle しますが、ChEMBL の TSV はファイル境界で activity ID
  (したがって関係タイプ) が偏ります。`--n-eval-triples` を `--prep-max-test`
  より小さくすると **関係タイプ分布が test 全体と乖離** することが
  あるので注意してください。
- **filtered プロトコルの `--max-filter-tails` / `--max-filter-rows`** —
  これらを絞ると `kgfm` の filtered 評価には 2 つの相反する効果が出ます。
  - (a) tail 候補語彙が小さくなる → ランキング分母が縮むので **MRR は
    楽観方向**。
  - (b) 同じ `(h, r)` の他の正解 tail を `-inf` でマスクしきれない →
    real positive が distractor に混ざるので **MRR は悲観方向**。

  ULTRA / MOTIF は全エンティティに対する filtered ranking なので
  これらの上限を持ちません。手法間比較を厳密にやる場合は
  `--max-filter-tails ≈ |E|`、`--max-filter-rows ≈ |train+valid+test 合計
  行数|` 程度を取り、(a)(b) いずれも実質発動しない領域で走らせるか、
  もしくは `--protocols pooled` で揃えてください (pooled なら
  `pool_size` が 3 手法とも同じ意味になります)。
- **`kgfm bench prep --prep-max-test`** — これは 3 手法共通の上限です
  (test.txt のサイズそのもの)。ここを大きくすれば 3 手法とも信頼区間が
  同じだけ狭まります。kgfm の `--n-eval-triples` だけ大きくしても、
  `--prep-max-test` で先に切られた test ファイルより多くは評価できません。

### 実務上のガイドライン

- 公開できる数値を出すときは少なくとも `large` プリセット
  水準 (`--prep-max-test 10000`、可能なら `--prep-max-test 50000` 以上) で取り、
  `--n-eval-triples` を `--prep-max-test` 以上に設定してください。
- 手法間で差を主張する場合は、`table.md` の `n_eval` と Protocol が
  揃っていることを確認した上で、差が標本誤差を上回るか併記してください。
- kgfm を filtered で評価する場合、`--max-filter-tails` ≈ `|E|`、
  `--max-filter-rows` ≈ `|train+valid+test 合計行数|` を目安に、
  「フィルタが完全」「候補集合が全エンティティ」の状態に持っていくのが
  ULTRA / MOTIF と最もフェアな比較条件です。

### filtered の落とし穴 2 つ

**(1) `kgfm eval --protocol filtered` と `kgfm bench cell` は既定の filter
集合が違います。** `kgfm bench cell` は KG 完成タスクの標準どおり
**train + valid + test** の和集合から filter index を作りますが、
`kgfm eval` の既定は **test だけ**です。したがって**ベンチマークの
チェックポイントを `kgfm eval` で再スコアすると、何も間違えていないのに
違う（＝甘い）数字が出ます**。実測: xxlarge のチェックポイントを同じ
`max_filter_tails` で再スコアすると、test のみの既定で MRR 0.4886、
train+valid+test で 0.4555 でした。再スコアするときは `--filter-list` を
3 つとも明示してください（トップレベル README の「評価プロトコル」節の
コマンド例がその形です）。

比較したい 2 つの JSON がほんとうに同じ条件かは、`metrics` の中の
**`filter_pairs`（filter index に入った `(h,r)` の数）と
`tails_outside_vocab`** を見れば分かります。プロトコルが一致していれば
この 2 つは完全に一致します。なお `max_filter_rows` は語彙の大きさを
決めているにもかかわらず**指標はほとんど動きません** — xxlarge の
チェックポイントで無制限と 10M がどちらも MRR 0.48858357 でした。理由は
(2) です。逆に **filter 集合の選び方 (train+valid+test か test だけか) は
0.03 以上動く**ので、効くのはそちらだと覚えてください。

**(2) 評価対象の tail の約 93% は filter 語彙の外にあります。** これは
どの run でも一貫していて（92.8〜94.2%。xxlarge では 12,032 件中 11,162 件
= 92.8%）、ファイル単位の inductive 分割の帰結です — test のエンティティは
filter ファイル側とは別の集団だからです。語彙に無い真の tail は候補列に
追加してから順位を取るので**指標の計算自体は一貫していて、同じ語彙で
取った run どうしは比較できます**。ただし**教科書的な filtered MRR とは
別物なので、外部の公開値と並べて引用しないでください。**

---

## 動作確認時の実測値（3 手法が同じ run で揃っている唯一の記録）

出典は `benchmarks/results/chembl/20260808T073317Z_fulltest/table.md` です。
データ: `kgfm bench prep --prep-max-train 50000 --prep-max-valid 2000
--prep-max-test 2000`（|E| = 72,669, |R| = 22, inductive モード）、kgfm は
200 steps。**これは「3 手法が同じ実行ディレクトリで動くこと」を確認するための
スモークで、kgfm の性能を示す数字ではありません** — kgfm 側は 200 step
（`config_small.yaml`）しか学習していません。kgfm の実力は
`20260811T230140Z_chembl_xlarge_compare`（6,000 step、filtered MRR 最良
0.4641）と `20260816T053252Z_chembl_xxlarge`（1 エポック、filtered 0.4410 /
pooled 0.5164）を見てください。ただし**それらの run には ULTRA / MOTIF の行が
ありません**（ベースラインは prep された KG を読むので、`kgfm-ultra` /
`kgfm-motif` を同じディレクトリに後から足すことは可能です）。

| 手法 | Protocol | MRR | Hit@1 | Hit@3 | Hit@10 | MR | n_eval |
|---|---|---|---|---|---|---|---|
| kgfm (ngram) | pooled (5000) | 0.2632 | — | — | 0.2807 | — | 5120 |
| kgfm (ngram) | filtered (50000) | 0.2698 | — | — | 0.2791 | — | 5120 |
| kgfm (transformer) | pooled (5000) | 0.1966 | — | — | 0.3891 | — | 5120 |
| kgfm (transformer) | filtered (50000) | 0.2851 | — | — | 0.3639 | — | 5120 |
| ULTRA (zero-shot) | filtered | **0.1753** | 0.1532 | 0.1993 | 0.2100 | 5702.79 | 2000 |
| MOTIF (zero-shot) | filtered | **0.1740** | 0.1470 | 0.2003 | 0.2100 | 7650.40 | 2000 |

> transformer の行は**エンコーダごとの学習率**（`train.default_lr()`、
> transformer の fine-tune は `lr=3e-5`）を入れたあとの値です。それ以前は
> ngram 用の `1e-3` が transformer にもかかっていて完全に崩壊しており、
> しかも当時の同順位判定（許容誤差なし）と組み合わさって
> pooled 0.9330 / filtered 0.0004 という無意味な値を出していました。
> 両方の詳細は後述の「学習率はエンコーダごとに決まります」と
> 「評価指標の注意点: 同順位の扱い」にあります。
> **`benchmarks/results/chembl/` の 2026-06 以前の run（`20260522T101455Z` の
> transformer 行 0.8686 など）はこの崩壊状態の値**なので、参照しないで
> ください。

- ULTRA / MOTIF の値は、この 2 つを `kgfm/baselines/` に切り出すリファクタの
  前後で完全に一致しています（同じ結果が再現されることの確認）。
  `kgfm` / `gnn` / `kgfm-ultra` のどの env で実行しても指標が一致することも
  確認済みです（`gnn` env は現在使っていません）。
- **kgfm の最終評価は `best.pt`（valid MRR 最良）を読みます**。
  `final.pt`（最終ステップ）ではありません。in-loop validation を必ず
  走らせるようにしたことで、`train.py` の本来の設計どおりの挙動に
  なりました。ただし ChEMBL の file-level inductive 分割では valid と test の
  傾向がかなり違い、200 step のスモークでは valid MRR が step 20〜40 で
  頭打ちなのに test MRR は step 200 まで伸び続けます（**valid が test の
  良い代理になっていない**）。best.pt を選ぶことが必ずしも test 最良を
  選ぶことにならない、という含みがあります。
- **学習は run 間で再現しません。** `StreamingTripleDataset` は
  `IterableDataset` で、`num_workers>1` だとワーカー間のバッチ到着順が
  非決定的なため、同じ seed でも別の run になります。

### 損失関数は切り替え可能（既定は contrastive）

`--loss` で選びます。いずれも in-batch negative を使うため、**バッチサイズが
そのまま負例数**です。

| `--loss` | 内容 |
|---|---|
| `contrastive` **(既定)** | InfoNCE / NT-Xent（正規化 cosine を温度で割る） |
| `softmax_ce` | 生スコアの softmax 交差エントロピー（`kgfm/losses.py` を入れる前の唯一の目的関数。2026-08-09 より前の run はすべてこれ） |
| `bce` | 候補ごとの二値交差エントロピー（ConvE の 1-N scoring） |
| `margin` | max-margin ヒンジ（TransE 系） |
| `self_adversarial` | RotatE の self-adversarial negative sampling |

**数式と、既定を `contrastive` にした理由（実測付き）はトップレベルの
[README.md「学習目的関数（損失）」](../README.md#学習目的関数損失)に
まとめてあります。** ベンチマークからは `--loss` / `--loss-temperature`、
または YAML の `loss:` / `loss_temperature:` で指定できます。

```bash
kgfm bench run --config large --loss softmax_ce   # 2026-08-09 より前の挙動で回す
```

### 学習率はエンコーダごとに決まります

単一の学習率では両エンコーダを賄えません。`kgfm.train.default_lr()` が
選びます（`--lr` を明示すればそちらが優先）。

| encoder | freeze | lr | 理由 |
|---|---|---|---|
| ngram | — | `1e-3` | スクラッチ学習の `EmbeddingBag`。大きく動かす |
| transformer | off | `3e-5` | 事前学習済み LM の fine-tune (標準は 2e-5〜5e-5) |
| transformer | on | `1e-3` | 学習するのは小さな射影ヘッドだけなので大きい方 |

`default_lr()` が入る前は両方 ngram 用の `1e-3` 固定で、
**BERT は完全に崩壊していました**: 200 step で
loss が 5.5498 → 5.5466 とほぼ動かず、無関係なテキスト同士の埋め込みの
cos 類似度が **1.000000**。`3e-5` では 120 step で loss 3.2551 → 1.7588、
cos 類似度 0.95 (BERT 本来の異方性の範囲) と正常に学習します。勾配ノルムも
0.01 → 6〜10 と実質的な値になります。

### 評価指標の注意点: 同順位の扱い

`eval._ranks_with_ties` の話です。素朴な実装は順位を「真の tail より
**厳密に大きい**スコアの数 + 1」と定義しますが、この定義だと**全候補の
スコアが同じモデル（= 崩壊したモデル）が全候補について rank 1、つまり
MRR = 1.0 を取ってしまいます**。仮定ではなく実際に起きていて、
`benchmarks/results/chembl/20260606T090031Z/table.md` の transformer 行が
filtered で `1.00` を報告しています。

現在は KG 完成タスクの標準である**同順位の平均**（楽観順位と悲観順位の
平均 = `#greater + (#tied + 1) / 2`）を使っています。同順位が無ければ
厳密比較の定義と完全に同値になるので、実際 ngram セルの数値は小数 6 桁まで
不変でした。

同順位判定には**許容誤差**を使います (`torch.isclose`, `TIE_RTOL=1e-5` /
`TIE_ATOL=1e-8`)。厳密な等値だけでは不十分だからです — 崩壊モデルでは
スコアが 1e-7 程度ずれ、しかも真の tail だけ `(hr*t).sum(-1)`、候補側は
`hr @ pool.t()` で計算されるため**浮動小数の累積順序が系統的に違い**、
真の tail が毎回わずかに有利になります。これが厳密判定でも pooled MRR
0.93 が残った原因でした。

閾値の根拠: スコアは数百次元の float32 内積で、相対誤差 1e-6 程度は
累積ノイズであって信号ではありません。1e-5 はその上、かつ実際に学習した
モデルが作るスコア差よりはるかに下です。同一チェックポイントでの実測:

| チェックポイント | 厳密判定 | 許容誤差つき |
|---|---|---|
| ngram (正常) | 0.26323267 | 0.26323609（実質不変） |
| transformer (崩壊) | 0.9330 | **0.00040** |

崩壊モデルの 0.00040 は候補 2500 件相当のランダム順位で、filtered
プロトコルが独立に出していた 0.0004 と一致します。

---

## 上流リポジトリへの自動パッチ

`kgfm/baselines/common.py` は対象リポジトリにオンザフライでパッチを
当てます。

- `ChEMBLCustom` データセットクラスを `<repo>/<pkg>/datasets.py` に追記
  （冪等。センチネルコメントで囲み、再実行時に上書き）。
- `torch.load(self.processed_paths[0])` を `weights_only=False` 付きに
  書き換え（PyTorch 2.6+ で既定値が反転し、上流の pickle キャッシュが
  読めなくなったため）。
- `<pkg>/layers.py` の torch_geometric バージョン判定
  `[int(i) for i in torch_geometric.__version__.split(".")]` に
  `if i.isdigit()` を追加。PEP 440 のサフィックス付きバージョン
  (`2.8.0.post1` など) で `ValueError: invalid literal for int()` が
  出るためです。参照されるのは minor 成分だけなので挙動は変わりません。
- `script/run.py` に `#test_triplets: N` のログ行を追加（評価件数
  `n_eval` を取得する唯一の手段）。

元に戻したい場合はセンチネルで囲まれた区間を削除し、`weights_only=False`
と `if i.isdigit()` を消してください。

### CPU 実行時の `rspmm`

`--gpus null` を渡すだけでは CPU 実行になりません。`rspmm` のローダは
`torch.cuda.is_available()` が真であれば `.cu` ソースをコンパイルしよう
とするため、GPU はあるが torch と一致する nvcc が無いマシンでは、推論に
入る前にビルドが失敗します。そこで `--gpus null`（`[]` / `none` も同様）
のときは子プロセスに `CUDA_VISIBLE_DEVICES=""` を渡し、CPU 専用の
ソース一覧でビルドさせています。

同様に、シェルに残った `CUDA_HOME`（別 env を `conda activate` した名残
など）は torch の CUDA と一致するときだけ子プロセスに渡します。一致
しない値をそのまま渡すと、拡張のロード時に
`libcudart.so.12: cannot open shared object file` のような分かりにくい
エラーになります。

---

## 既知の問題: H200 (sm_90) での ULTRA `rspmm` カーネル

ULTRA の CUDA 拡張 `rspmm` は H200 (compute capability sm_90) で
`cudaErrorIllegalAddress` を出して落ちます。これは torch 2.5.1+cu121 を
ピン留めした専用 env (旧 `kgfm-ultra`) **でも**再現し、しかも ChEMBL 固有
ではなく ULTRA に同梱されている `CoDExSmall` でも同様でした。upstream
`rspmm` のカーネル実装と新しめの torch + sm_90 の組み合わせが原因と
見られます。ULTRA を専用 env から `kgfm` env に移したのは、この env を
維持しても ULTRA の GPU 実行の役には立たなかったためです（env 自体は
MOTIF が使うので残っています）。

回避策: **`--gpus null` で CPU 実行**にする。CoDExSmall で公開値どおりの
MRR 0.498 を出すこと、ChEMBL でも上記の通り MRR 0.175 を出して完走
することを確認済みです。CPU でも本ベンチマーク程度のサイズなら 20 分
程度で終わります。

```bash
kgfm-ultra --out-dir latest --gpus null   # --gpus null が既定なので省略可
```

---

## ファイル構成

```
kgfm/                       # 処理の本体はすべてここ
├── cli.py                  # `kgfm` コマンド (train / eval / bench / report)
├── report.py               # `kgfm report` — 集計 + 学習ログ解析
├── report_html.py          # report.html の HTML / SVG 生成
├── runs.py                 # 実行ディレクトリと tee ログ (3 手法で共有)
├── envs.py                 # conda env 解決 (子プロセス用)
├── bench/                  # kgfm 自身のベンチマーク
│   ├── cli.py              # `kgfm bench ...` の引数定義
│   ├── pipeline.py         # prep → sweep (run dir / meta.json)
│   ├── config.py           # パラメータ + スケールプリセット
│   ├── prep.py             # kgfm TSVs -> entity-ID KG
│   ├── sweep.py            # kgfm スイープ
│   └── cell.py             # kgfm セル 1 つ (学習 + 評価 + JSON)
└── baselines/              # 外部手法 — 別コマンド
    ├── common.py           # 上流パッチ / 実行 / メトリクス抽出
    ├── ultra.py            # `kgfm-ultra`
    └── motif.py            # `kgfm-motif`

benchmarks/                 # シェルは薄いラッパのみ
├── README.md               # 本書
├── run_chembl.sh           # bench run → ultra → motif → report (config_small.yaml)
├── run_chembl_middle.sh    # --config benchmarks/config_middle.yaml
├── run_chembl_large.sh     # --config benchmarks/config_large.yaml
├── run_chembl_xlarge.sh    # --config benchmarks/config_xlarge.yaml
├── run_chembl_xlarge_compare.sh  # --config benchmarks/config_xlarge_compare.yaml (28 セル)
├── run_chembl_xxlarge.sh   # --config benchmarks/config_xxlarge.yaml (全 674M 行 1 エポック)
├── run_chembl_large_2gpu.sh   # config_large.yaml  + --nproc 2
├── run_chembl_xlarge_2gpu.sh  # config_xlarge.yaml + --nproc 2
├── config_small.yaml / _middle / _large / _xlarge /
│   _xlarge_compare / _xxlarge.yaml         # スケール設定
├── resume_chembl.sh        # config_large.yaml + --resume <target>
├── setup_baselines.sh      # ULTRA / MOTIF の clone + 依存チェック
├── setup_baseline_env.sh   # ベースライン用 env (kgfm-ultra) を構築
├── ULTRA/                  # 上流クローン (gitignore)
├── MOTIF/                  # 上流クローン (gitignore)
├── chembl_kg/              # 生成 KG (gitignore)
└── results/                # gitignore
    └── chembl/
        ├── 20260507T1234Z_chembl_large/   # 実行ごとの記録
        │   ├── meta.json
        │   ├── kgfm_<protocol>_<encoder>.json / ultra.json / motif.json
        │   ├── table.md / report.html
        │   ├── run.log + 各ステップログ
        │   └── kgfm_ckpts_<encoder>/
        └── latest -> 20260507T1234Z_chembl_large
```

---

## 細かい注意事項

実装は `kgfm/baselines/` にあります（`common.py` に共通処理、`ultra.py` /
`motif.py` は `Baseline` インスタンス + `main()` だけ）。以前の
`benchmarks/run_ultra.py` / `run_motif.py` はこの 2 コマンドに置き換わって
おり、リポジトリには残っていません。

- **ULTRA dataset シム** — `kgfm/baselines/common.py` は
  `ultra/datasets_chembl.py` のような別モジュールを作るのではなく、
  `ultra/datasets.py` の末尾に直接 `ChEMBLCustom` を追記します（ULTRA が
  `getattr(ultra.datasets, <ClassName>)` でクラスを解決するため、別モジュール
  では見つけてもらえません）。upstream で `TransductiveDataset` が
  リネームされたら、`common.py` 内のクラステンプレート文字列を編集して
  ください。
- **MOTIF CLI** — MOTIF は ULTRA から派生したコードベースで、
  `script/run.py` のフラグ規約 (`--dataset --gpus --epochs --bpe --ckpt`)
  も同一です。そのため `motif.py` は `ultra.py` とほぼ同じ `Baseline`
  定義になっています。手法を 1 つ足すときも、`Baseline` を 1 個定義して
  コンソールスクリプトを生やすだけです。
- **GPU メモリ** — ULTRA の NBFNet メッセージパッシングは `|E| · L`
  (L はレイヤ数) でスケールします。OOM になったら `--prep-max-train` を
  下げるか、`kgfm-ultra --bpe` / upstream の設定 YAML 側で batch size を
  縮めてください。
