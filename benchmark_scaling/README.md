# benchmark_scaling — スケーリング則の測定

`benchmarks/` と同じ構成・同じパイプラインで、**軸だけが違う**実験です。

| | benchmarks/ | benchmark_scaling/ |
|---|---|---|
| 問い | どの構成が一番強いか | 計算量を増やすと損失はどう下がるか |
| sweep 軸 | encoders × heads × freezes | **モデルサイズ**（他は固定） |
| 主な出力 | `table.md` / `report.html` | `scaling_table.md` / `scaling_report.html` |
| 出力先 | `benchmarks/results/chembl/` | `benchmark_scaling/results/chembl/` |

学習は `kgfm bench run` がそのまま担当します。**スケーリング用の学習コードは
ありません** — `kgfm scaling` が既存の `cell_*.log` を読み直して
(計算量, 損失) 座標に置き換えるだけです。数週間前に終わった run に対しても
後から実行できます。

## 使い方

```bash
bash benchmark_scaling/run_scaling_smoke.sh     # 数分。まずこれで通しを確認
bash benchmark_scaling/run_lr_probe.sh          # サイズごとの学習率を測る
bash benchmark_scaling/run_lr_probe_extend.sh   # 上の格子を下側に拡張
bash benchmark_scaling/run_scaling_scratch.sh   # ★ 本番（ランダム初期化 7 サイズ）
bash benchmark_scaling/run_scaling.sh           # 事前学習済み 9 エンコーダ
bash benchmark_scaling/run_scaling_frozen.sh    # 凍結エンコーダ版（7B を含む）
bash benchmark_scaling/resume_scaling.sh latest # 中断した run の再開
```

**中心になるのは `run_scaling_scratch.sh`** です。サイズ軸から事前学習量を
外したかったので、ランダム初期化の `scratch-*` 7 サイズを回します
（4.4M / 11.2M / 28.8M / 41.4M / 109.5M / 234.6M / 335.4M、48,000 step、
合計約 26 時間）。**先に `run_lr_probe.sh` + `run_lr_probe_extend.sh` を
回して、サイズごとの学習率を決める必要があります** — 固定の学習率で回すと
サイズ軸が壊れます（後述「4. LR がサイズに合っていない」）。すでに測った値は
`config_scaling_scratch.yaml` の `cells:` に入っているので、条件を変えない
なら再測は不要です。

`run_scaling.sh`（`config_scaling.yaml`）は**事前学習済みモデルの側**で、
`bert-tiny/mini/small/medium` + `ngram` + `mpnet` + `bert-multilingual` +
`bge-large` + `xlm-roberta-large` の 9 セル × 4,000 step です。
scratch 側の指数との差が「事前学習が何を寄与しているか」に対応します。

各スクリプトは 3 コマンドです（`kgfm bench run` が学習、`kgfm report` が
通常のベンチマークレポート、`kgfm scaling` がスケーリングレポート）。

```bash
kgfm bench run --config benchmark_scaling/config_scaling.yaml
kgfm report  --out-dir latest --results-root benchmark_scaling/results/chembl
kgfm scaling --out-dir latest --results-root benchmark_scaling/results/chembl
```

`kgfm report` に `--results-root` を渡しているのは、その既定が
`benchmarks/results/chembl`（手法比較側）で、そのままだと別のツリーの
`latest` を見てしまうためです（`kgfm scaling` の既定は
`benchmark_scaling/results/chembl` なので不要ですが、対称性のために
明示しています）。`kgfm bench run` は config の `results_root:` を使います。

## 設計: 1 本の学習曲線が 1 本のスケーリング曲線

モデルサイズごとに 1 セルを回し、**そのセルの in-loop validation の履歴を
そのまま計算量軸に載せます**。1 つのモデルについて計算量ごとに別々の run を
回す必要はありません — 1 回の学習が、自分より小さい全予算を通過するからです。

そのため `eval_every` は通常のベンチマークよりずっと細かく設定しています
（**validation 点がそのままデータ点**なので）。`valid_loss_batches > 0` が
必須です。0 だと y 軸が存在しません。

固定するもの（いずれも動かすと交絡します）:

| 設定 | 理由 |
|---|---|
| `batch_size` | B-1 が in-batch 負例数そのもの。変えると課題自体が変わる |
| `proj_dim` | スコアを計算する次元。「大きいエンコーダ」と「広いスコア」が混ざる |
| `heads` | 1 つだけ。セルタグが `<encoder>` のままになり、変数が 1 つに保たれる |
| `max_steps` | 全セル同じ。各線が同じ**データ量**を張り、計算量/step だけが違う |

## 計算量の数え方 (`kgfm/scaling/compute.py`)

標準的な会計（Kaplan et al. 2020）を使います。

```
forward  ~ 2·N·T      backward ~ 4·N·T      合計 ~ 6·N·T
1 PF-day = 1e15 × 86400 = 8.64e19 FLOPs
```

この実装で重要な 3 点:

- **T は実測値で、推定値ではありません。** `encode_triple` は 1 step で 3B 本の
  系列を通し、tokenizer はバッチ内最長にパディングします。学習側が実際の
  パディング済みトークン数を数えて `tokens=` としてログに出し、レポートは
  それを読むだけです。`3 · B · max_length` から導出すると ChEMBL では
  約 3.5 倍の過大評価になります（実際は 128 のうち約 36 までしか使わない）。
- **凍結エンコーダは 6NT ではなく 2NT です。** `requires_grad=False` なので LM
  への backward が存在しません。6NT 扱いすると 7B モデルの計算量を 3 倍に
  見積もり、全プロットで位置を間違えます。
- **ngram は計算量軸に載せません。** `HashedNgramEncoder` は EmbeddingBag で、
  268M のパラメータは参照表です。gather は N に比例した積和をしないので、
  6ND を当てはめると sweep 中**最も高価なモデル**として描かれてしまいます
  （実際は最も安い）。レポートは行を残したうえで「計算量軸に載せない」と
  明示します。

## 評価時メモリ — サイズ軸の上端を決めていたのは grad モードだった

`scratch-large` (24x1024, 335M) は「B=256 で学習は通るが最初の in-loop
evaluation で OOM する」ため、サイズ軸は 234M (`scratch-xl`) で止めていました。
これは**バグで、モデルサイズの限界ではありませんでした**。

`TransformerEncoder.forward` は LM 呼び出しを

```python
ctx = torch.no_grad() if self.freeze else torch.enable_grad()
```

で包んでいました。`torch.enable_grad()` は**外側の `no_grad()` を上書きします**
（`nullcontext` のように委ねるのではなく、強制的に ON にする）。したがって
リポジトリ中のすべての `@torch.no_grad()` — `evaluate`、`evaluate_loss`、
`build_candidate_pool`、`build_filter_index`、`viz` — が、**唯一メモリを
確保する呼び出しのところで無効化されていました**。評価は毎回、学習と同じ
サイズの計算グラフ（3B 系列分）を作って捨てていたことになります。

凍結エンコーダは no_grad を強制、それ以外は**外側のモードを継承**
（`contextlib.nullcontext()`）が正しい形です。学習は外側 no_grad の下で走る
ことがないので、学習側は何も変わりません。

B=256 の評価側 `encode_triple` 1 回の peak allocated（実測）:

| | 修正前 | 修正後 |
|---|---|---|
| `scratch-base` 109M | 5.299 GiB | 0.293 GiB |
| `scratch-xl` 234M | 9.170 GiB | 0.387 GiB |
| `scratch-large` 335M | 13.684 GiB | 0.387 GiB |

修正後が深さに対してほぼ平坦なのは、層を抜けた後に残るのが pooling 済み出力
だけになるからです。

**数値は一切変わりません。** 実チェックポイントに対して実際の評価経路
(`kgfm.eval.evaluate` + `train.evaluate_loss`) を新旧両方で回して、metrics
dict が完全一致・validation loss も一致 (3.298191547393799) することを
確認済みです。**過去の run を回し直す必要はありません** — 変わるのは
メモリと速度だけです。

### `nvidia-smi` も `memory_reserved()` も、単独ではセルのサイズ決めに使えない

各 in-loop eval は自分用の行を出します:

```
[mem @ step 500] train_peak=..GiB train_alloc=..GiB eval_peak=..GiB reserved=..GiB driver=../..GiB
```

- `train_peak` — 前回の eval からの peak allocated。学習の forward+backward の
  活性で、eval 到達時にはもう解放されています。**セルのサイズを決めるのは
  これ**で、`train_alloc`（optimizer step 後の残り = パラメータ + Adam の
  モーメント）ではありません。
- `eval_peak` — 評価中の peak。修正前はここが binding constraint でした。
- `driver` — `torch.cuda.mem_get_info()` による実使用 / 総容量。**実際に OOM
  するのはこの数字だけです。**

3 つを並べるのは、**1 つだけ見ると必ず読み違えるから**です。修正後の
`scratch-large` (335M, B=256) の実測は

```
[mem @ step 75] train_peak=101.81GiB train_alloc=3.80GiB eval_peak=7.98GiB \
                reserved=9.61GiB driver=10.34/139.80GiB
```

で、**同じ瞬間の値が 3.80 と 101.81 と 10.34 GiB**です。どれも間違いでは
ありません:

- `train_alloc` 3.80 GiB は optimizer step 後の残り（パラメータ 1.34 GiB +
  Adam の 2 モーメント 2.68 GiB）。活性はもう解放済みなので、**これでセルを
  サイズするとほぼ 30 倍過小に見積もります**。
- `train_peak` 101.81 GiB が本当の要求量。24 層 × 768 系列分の活性です。
  139.8 GiB のカードに対して、**律速しているのは今や学習側**です。
- `driver`/`nvidia-smi` はアロケータのキャッシュ済みプールを見ています。
  同じ run の**連続する 2 回の eval** が `driver=10.34GiB` と `115.94GiB`
  を返しました（間でアロケータがキャッシュを吐いたため）。`train_peak` は
  どちらも 101.81 GiB です。**つまり `nvidia-smi` の答えはいつ叩いたかで
  11 倍変わり、単独では pass/fail を予測できません。**

修正前の OOM はこの構図で起きていました: 学習ピークでカードがほぼ埋まった
ところに、eval が**さらに 13.7 GiB の不要な計算グラフ**を積んで超過。今は
eval の上乗せが 0.4 GiB なので、アロケータがキャッシュを吐いて再試行する
警告 (`memory allocation failed with OOM ... CUDACachingAllocator`) が 1 回
出ることはあっても、回復して先に進みます。**その警告行は致命ではありません**
— `torch.OutOfMemoryError` の traceback が出たときだけが失敗です。

## レポート冒頭の「Experiment settings」

レポートの先頭に、**何をどう回したか**を必ず出します。数値を読む前に設定が
分かるようにするためで、run-level の設定一覧と、セルごとの表を出します。

セルごとの表には **step 数・見たサンプル数・それが何エポックにあたるか**を
並べます。step 数は単体では解釈できないからです — 48,000 step は大きな数に
見えますが、このコーパスでは **0.0182 エポック**（2% 未満）でしかありません。

| Cell | Params | Steps | Examples seen | Epochs | Tokens | PF-days |
|---|---|---|---|---|---|---|
| scratch-base | 109,679,104 | 48,000 | 12,288,000 | 0.0182 | 3.66e9 | 0.0279 |

エポック数は**実測の行数**から計算します（`data.epoch_examples`、
`(パス, サイズ, mtime)` でキャッシュ）。ChEMBL のファイルは 1.81 GB〜39 KB と
不均一なので、1 ファイル分から掛け算すると 26% ずれます。

この数字が小さいこと自体が重要な情報です。**1 パスに全く届かない**ので行の
再出現がなく、train/valid のギャップは記憶ではなく**ファイル間の集団シフト**
だと分かります。

## プロット

| プロット | 見方 |
|---|---|
| **validation loss vs 計算量（色 = モデルサイズ）** | 主役。線が交差する点が「大きいモデルがコストに見合い始める」ところ |
| 計算効率フロンティア（log-log）+ 冪則フィット | 各セルの軌跡を薄く重ねた上に下側包絡線（赤）とフィット（黒）。`L = a·C^b` の **b** が求めたい指数 |
| 予算固定での loss vs パラメータ数 | 「その予算で止めたらどのサイズが勝つか」 |
| **系統ごとのフロンティア（別々にフィット）** | プールしたフィットはサイズと事前学習レシピを同時に動かしている。系統内フィットはレシピを固定するので、指数を「容量の話」として読める |
| フロンティア上のモデルサイズ vs 計算量 | 右上がり = 予算が増えるほど大きいモデルが有利。平坦 = データに使う方が良い |
| loss vs 学習データ量 | 「計算量が増えた」と「データが増えた」を分離するための軸 |
| **ranking error (1 − MRR) vs 計算量** | 第 2 の y 軸。loss 軸と同じセル・同じ色・同じ x 軸で、y 軸だけが違う |
| **ranking error フロンティア + 冪則フィット** | loss 軸のフロンティアと同じものを 1 − MRR で |

**色は viridis で、log10(パラメータ数) に比例**します。順位ではなく量に対応
させているので、4.4M と 41M が隣り合った色にならず、色をそのまま「大きさ」と
して読めます。カテゴリカルな 6 色パレットだと 9 セルで色が一周してしまい、
何も意味しなくなります。viridis は知覚的に均等で明度が単調なので、白黒印刷
でも一般的な色覚特性でも順序が保たれます。matplotlib が無くても使えるよう
制御点を `kgfm/scaling/palette.py` に埋め込んでいます。同じ色は全チャートで
同じモデルサイズを指します。

軸は log スケールではなく **log10 に変換した値**です。`kgfm/charts.py` は
3 つのバックエンドを 1 つの呼び出しの裏に持っていて log 軸を扱えないため、
データ側を変換して 3 つとも同じ絵が出るようにしています。目盛りの `-2` は
0.01 PF-days の意味です。

フィットはフロンティア（各予算の最良点）に対して行います。全点で回帰すると
「どれだけ長く回したか」を測ることになり、「その予算での最良性能」になりません。

## 4 つの交絡と、その外し方

このスタディを素朴に回すと「BERT 系統に絞っても scaling law が見えない」
という結果になります — 指数が −0.02 前後、R² が 0.4 程度。**それは法則が
無いからではなく、指数を潰す要因が重なっているから**です。以下の 4 つが
それで、対処はいずれもコード側に入っています（1〜3 は `kgfm/scaling/` の
レポート側、4 は config 側）。4 つを外すと同じ 9 セルが R²=0.871 まで
上がります（後述）。

1. y 軸（validation loss）に天井と床がある → `1 − MRR` の軸を併記
2. 系統をまたいでフィットするとレシピとサイズが混ざる → 系統内でも別々にフィット
3. `L = a·C^b` は log-log で曲がれない → 床つき `L = L∞ + a·C^b` も同時にフィット
4. 学習率がサイズに合っていない → サイズごとに振る（`lr_probe.py`）

途中に挟まっている「おまけ: early stopping は入れてはいけない」は
5 つ目の交絡ではなく、**これらの対処を入れたあとに新しく入れてはいけない
もの**の話です。

### 4 番目（学習率）が最大の原因です

1〜3 はレポート側の対処ですが、**4 番目は config 側で、しかも見落としやすい**
ところです。この事前学習済みスイープは長らく config に
セルごとの LR 表が存在せず、全セルが `train.default_lr` の 3e-5 で
回っていたのです。既定のヘッド結線変更にともなって初めてプローブしたところ、
**transformer 8 セル中 6 セルで 3e-5 が 1 桁低すぎる**と分かりました。

| エンコーダ | 旧（既定） | 実測最適 | 差 |
|---|---|---|---|
| bert-tiny | 3e-5 | **1e-3** | 0.266 nats |
| bert-mini | 3e-5 | **3e-4** | 0.267 |
| bert-small | 3e-5 | **1e-4** | 0.181 |
| bert-medium | 3e-5 | **1e-4** | 0.121 |
| mpnet | 3e-5 | **1e-4** | 0.246 |
| bert-multilingual | 3e-5 | **1e-4** | 0.128 |
| bge-large | 3e-5 | 3e-5 | — |
| xlm-roberta-large | 1e-5 | 1e-5 | — |

実測レートで回した結果（`scaling_pretrained`、9 セル × 4,000 step）:

| フィット | b | R² |
|---|---|---|
| プール フロンティア | −0.0309 | **0.871** |
| 床つき（L∞=2.325） | −0.1348 | 0.906 |
| **BERT 系統（Turc）** | **−0.0343** | **0.964** |
| retrieval 系 | −0.0469 | 0.732 |

best valid loss は BERT 系統で単調です（3.0822 / 2.9881 / 2.9235 / 2.7979 for
4.4M → 41.4M）。**同じ 9 セルを既定の 3e-5 一律で回すと R² は 0.396 まで落ち、
「法則が無い」ように見えます** — 傾き自体はほとんど変わらず（系統内 −0.0343
対 −0.0368）、**変わるのは傾きを隠すばらつきの量**だけです。誤差が最大なのは
小さいモデル側（tiny 0.266 / mini 0.267 / mpnet 0.246 nats）で、サイズ軸を
平坦化させるのはまさにそこです。

**留保**: 実測レートでの run は `head_mode=separate` でもあります。
この 1 本では LR とヘッド結線の寄与を分離できません。系統内指数がほぼ不変で
R² だけ動くことが、主因は LR だと示唆する根拠です。

### 凍結エンコーダ軸 — こちらは本当に平坦です

`config_scaling_frozen.yaml`（5 エンコーダ、110M → 7.1B、ヘッドのみ学習、
`scaling_frozen`）:

| セル | エンコーダ params | ヘッド params | best valid loss |
|---|---|---|---|
| `mpnet_frozen` | 110M | 590,592 | 3.3447 |
| `bert-multilingual_frozen` | 178M | 590,592 | 3.2033 |
| **`bge-large_frozen`** | 335M | 787,200 | **3.1844** |
| `xlm-roberta-large_frozen` | 561M | 787,200 | 3.2103 |
| `e5-mistral-7b_frozen` | **7.1B** | 3,146,496 | 3.2621 |

b = −0.0068（R²=0.151）。**7.1B の凍結エンコーダは 335M より悪い**のに
計算量は 35 倍です。凍結時にはエンコーダサイズがこのタスクの性能を
ほとんど決めていません。これはこのスイープの存在理由そのもので、
7.1B セルが無ければ答えられない問いでした（そして `frozen_only` プリセットは
原理的に fine-tune できないので、7.1B はここにしか現れません）。

**凍結系は LR にほぼ無感**でもありました（2 桁振って崩壊ゼロ、幅
0.03〜0.05 nats、5 セル中 2 セルが既定 1e-3 ちょうど、残り 3 つも 0.013〜0.027
しか上回らない = 分解能以下）。そのため `config_scaling_frozen.yaml` には
**あえてセルごとの LR を書いていません** — ノイズへの当てはめになるためです。
fine-tune 側で 8 セル中 6 セルが 1 桁ずれ、3 レートが崩壊したのと対照的で、
これは構造的な違いです（ヘッドのみ学習は小さく条件の良い問題で、壊すべき
事前学習済み重みが無い）。

### 1. y 軸に天井がある — だから 1 − MRR も並べて出す

学習損失は in-batch 負例に対する softmax です。したがって

* 上は `ln(B)` で頭打ち（B=256 なら 5.545）、
* 下はバッチ内で tail がどれだけ重複するかで決まる床がある

という二重の制約を持ちます。モデルサイズを 1 桁動かしても、この帯の中でしか
動けません。実測でも、loss 軸では 1 桁の N に対して 0.169 nats しか動かない
一方、1 本の学習軌跡は 0.24〜1.28 nats 動きます — つまり loss は
「容量」より「どれだけ回したか」を測っています。

そこで **同じ点を `1 − MRR` でも描きます**。MRR は全セル同じ大きさの候補
プールに対する順位なので、目的関数側の天井を持ちません。冪則は 0 に向かって
減衰する量に対する主張なので、MRR そのものではなく誤差 `1 − MRR` を使います。

実測（`results/chembl/scaling_pretrained/` の run、`bert-tiny/mini/small/medium` の
1 系統だけを取り出したフィット）:

| y 軸 | 指数 b | R² |
|---|---|---|
| validation loss | −0.0368 | 0.662 |
| **1 − MRR** | **−0.3318** | **0.836** |

同じ学習・同じ計算量軸で、y 軸を変えただけです。**loss 軸が法則を隠していました。**

さらに直接的な例として、LR プローブ（2000 step, B=256, 各サイズ最良の LR）では
**同じチェックポイントに対して 2 つの軸がサイズの順位を逆に付けます**。

| | loss（小さいほど良い） | MRR（大きいほど良い） |
|---|---|---|
| `scratch-tiny` (4.4M) | **3.189** | 0.5182 |
| `scratch-mini` (11.2M) | 3.217 | **0.5266** |

loss で見ると 4.4M が 11.2M に勝ち、順位で見ると負けています。同じ予算・同じ
チェックポイントなので、これは学習の違いではなく y 軸の違いそのものです。

### 2. プールしたフィットはレシピとサイズを同時に動かしている — だから系統内で切る

`bert-medium` と `bge-large` はサイズも違いますが、事前学習コーパスも目的関数も
違います。両者をまたいで引いた指数は「容量の効果」ではありません。
`kgfm/scaling/points.py` の `FAMILIES` が同じレシピのモデルを 1 系統にまとめ、
レポートは**系統ごとのフィットとプールしたフィットを並べて**表に出します。
系統内 −0.0368 に対しプール −0.0186 で、プールする方が指数を半分に潰していました。

### 3. 2 パラメータの冪則は曲がれない — だから床つきの形も同時にフィットする

`L = a·C^b` は log-log で直線しか描けません。フロンティアが寝てきている場合
（「法則が見えない」の実態はたいていこれです）、直線フィットは寝た部分に
引きずられて浅い指数を返します。そこで

```
L = L∞ + a·C^b
```

も同時にフィットします。`L∞` を固定すれば log-log で線形なので、`L∞` の
1 次元スキャンの中で OLS を解くだけです（scipy 不要）。スキャンの評価は
**log L ではなく linear L の残差**で行います — log 空間で評価すると `L∞` を
`min(L)` に押し付けるほど見かけの R² が上がってしまい、何もフィットせずに
高得点を取れてしまうためです。

合成データ `L = 2.0 + 5.0·C^−0.4` で検証すると、床つきフィットは
`L∞=1.9996, a=5.0008, b=−0.3999 (R²=1.00000)` を復元する一方、
同じデータへの 2 パラメータフィットは `b=−0.2454` — **指数を 4 割浅く
報告します**。

3 パラメータは 6 点未満では同定できません（`L∞` と `a` がほぼ完全に
トレードオフします）。`MIN_SATURATING_POINTS = 6` 未満では表に
「needs 6 pts」と出し、フィットしません。スキャンの最適点が端に張り付いた
場合は表に `(unconstrained)` と付記します — その場合 `|b|` は推定値ではなく
下限として読んでください。

### おまけ: early stopping は入れてはいけない

検証損失は**二相性**を持ちます。多くのセルは「下がる → 数千 step 悪化する →
再び下がって、はるかに良い最小値に達する」という形です。実測
（`results/chembl/scaling_chembl/`、7 サイズ × 48,000 step、
`eval_every: 250` なので各セル 192 点）:

| セル | 前半25%の最良 | それ以降の最良 | 差 |
|---|---|---|---|
| `scratch-tiny` 4.4M | 3.1500 @1,750 | 3.2253 @42,250 | −0.0753 |
| **`scratch-mini` 11.4M** | 2.9293 @9,000 | **2.7727 @31,500** | **+0.1566** |
| **`scratch-small` 29.2M** | 2.8010 @12,000 | **2.5010 @46,500** | **+0.3000** |
| **`scratch-medium` 41.8M** | 2.9560 @12,000 | **2.8274 @42,250** | **+0.1285** |
| `scratch-base` 110.1M | 2.9054 @9,000 | 2.9058 @17,500 | −0.0004 |
| **`scratch-xl` 235.2M** | 2.9347 @12,000 | **2.7496 @41,750** | **+0.1850** |
| **`scratch-large` 335.9M** | 3.0071 @11,000 | **2.6975 @42,250** | **+0.3095** |

**7 セル中 5 セルが後半で 0.13〜0.31 nats 良い最小値に達します。**
早期に底を打って戻らないのは `tiny` と `base` の 2 つだけで、**それらで
調整した patience は「安全そう」に見えますが、安全ではありません** — 実際の
曲線上で patience = 4 / 8 / 16 評価点すべてをシミュレートすると、いずれも
悪化の途中で発火し、残り 5 セルをその差のぶん手前で打ち切ります。最大で
0.31 nats、**サイズ軸の全体幅（0.65 nats）のおよそ半分**にあたり、
どのサイズが勝つかが入れ替わります — 学習率を間違えたときとまったく同じ
壊れ方です。

さらに悪いことに、**loss が悪化している区間でも MRR は改善を続けます**
（`scratch-large` で 0.4640 → 0.5105）。loss の悪化を停止条件にすると、
順位性能がまだ伸びている最中に止めることになります。

レポートの「train and validation loss vs step」の図がこの形を示し、その下の
表は**その run の曲線から毎回再計算**されます（上の数値の焼き付けではありません）。
計算量軸の図では見えないので、この図だけが step 軸です。

### 4. LR がサイズに合っていない — だからサイズごとに振る

`train.default_lr` は全 random-init エンコーダに同じ 1e-4 を渡します。
4M のモデルに最適な値と 110M のモデルに最適な値は違うので、そのままだと
サイズ軸は「容量」と「その LR がそのサイズにたまたま合っていたか」の
混合になります。

```bash
bash benchmark_scaling/run_lr_probe.sh             # scratch-*、格子 1e-4 … 3e-3
bash benchmark_scaling/run_lr_probe_extend.sh      # 同じ系統を 1e-5 / 3e-5 に拡張
bash benchmark_scaling/run_lr_probe_pretrained.sh  # 事前学習済み系統
```

`run_lr_probe.sh` の格子（1e-4 … 3e-3）では小さいサイズがすべて**下端の
1e-4 で勝ってしまった**ので、`run_lr_probe_extend.sh` で格子を下に伸ばして
います。**端で勝った値は最適値ではなく上限**なので、内点になるまで下げる
必要があります。`summarize_lr_probe.py` に複数のディレクトリを渡すと 1 枚の
表にまとまるので、両者の結果は合成できます。

実測結果は `benchmark_scaling/results/lr_probe_<experiment>/` に 1 スタディ
1 ディレクトリで残っています（`lr_probe_chembl` / `lr_probe_random` /
`lr_probe_pretrained` / `lr_probe_frozen` / `lr_probe_data_ngram` /
`lr_probe_xxlarge`）。各ディレクトリに
`lr_probe.json`（全 (encoder, lr) の validation 履歴）と `summary.txt`
（表 + 貼れる `cells:` ブロック）があります）。

各 (encoder, lr) を短く回し、**validation loss と MRR の両方**の表を出して、
そのまま `cells: <tag>: lr:` に貼れるブロックを出力します。

**採用基準は validation loss です**（`--select` の既定）。LR は optimizer が
実際に最小化している量に対して調整するものなので、その量で選びます。
レポートがフィットする軸は 1 − MRR ですが、**そこは合わせません** — MRR 列は
クロスチェックとして出力に残してあり、選択には使いません。実測では
`scratch-mini` で両者が食い違いました（loss は 3e-5、MRR は 1e-4 を選択。
0.5260 対 0.5266 でノイズの範囲内）。両者がノイズを超えて食い違う場合は、
基準を切り替えるのではなく食い違いそのものを報告してください。

**セルは GPU ではなく CPU/IO で律速します** — GPU 使用率は 4% 程度なので、
同時実行数を増やすのはほぼタダです。`--gpus 0,0,1,1` のように id を重ねると
1 GPU あたり 2 セル走ります。ただし `encode_triple` は 1 step で 3B 本の
系列を通すため、110M 級では 1 GPU に 2 セルは載りません。スクリプトは
そこで 2 フェーズに分け（41M までは 4 並列、bert-base 級は 2 並列）、
`summarize_lr_probe.py` が両フェーズの JSON を 1 枚の表にまとめます。

事前学習済み系統のレート格子は 1 桁低い側（1e-5〜3e-4）です。1e-3 は
**BERT を確実に崩壊させることが測定済み**（200 step で loss 5.5498 →
5.5466、無関係なテキスト間の cosine 類似度 1.000000）なので入れていません。

### 実測結果（2000 step, B=256, 最良 validation loss）

`coll` は ln(256)=5.5452 の 0.5% 以内、つまり崩壊固定点から出られなかった
セルです。角括弧が採用値で、**すべて格子の内点**です（端で勝った値は最適値
ではなく上限なので、内点になるまで格子を下に広げました）。

| | 3e-6 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|---|---|---|
| `scratch-tiny` 4.4M | — | 3.7627 | 3.3230 | **[3.1892]** | 3.2286 | 3.6682 | coll |
| `scratch-mini` 11.2M | — | 3.4016 | **[3.1677]** | 3.2167 | 3.3257 | coll | coll |
| `scratch-small` 28.8M | 3.5940 | **[3.2068]** | 3.2117 | 3.5207 | coll | coll | coll |
| `scratch-medium` 41.4M | — | 3.1837 | **[3.1714]** | 3.2959 | coll | coll | coll |
| `scratch-base` 109.5M | 3.2942 | **[3.1200]** | 3.3400 | coll | coll | coll | coll |
| `scratch-xl` 234.6M | 3.2386 | **[3.1995]** | coll | coll | coll | coll | coll |
| `scratch-large` 335.4M | **[3.2652]**¹ | 3.3188 | coll | coll | coll | coll | coll |

¹ `scratch-large` は格子の外側 1e-6 も測っており (3.5462)、採用値 3e-6 は
その意味で内点です。**loss と MRR が同じ 3e-6 を選びます**
（MRR: 1e-6 0.3331 / 3e-6 0.4204 / 1e-5 0.4019 / 3e-5 崩壊）。

この 335M のセルは**そもそも回せませんでした** — 3 レートすべてが最初の
in-loop evaluation で OOM して rc=1 で落ちていました（そのときのログが
`results/lr_probe_scratch_large_prefix_oom/` に残っています）。前述の
「評価時メモリ — サイズ軸の上端を決めていたのは grad モードだった」の修正
（`TransformerEncoder.forward` が外側の `no_grad` を継承するようにした）で
4 セルとも完走します。

なお `scratch-large` の 2000 step 時点の最良値 (3.2652) は `scratch-base`
(3.1200) や `scratch-xl` (3.1995) より**悪い**です。ランダム初期化では
大きいモデルほど 2000 step の多くを立ち上がりに使うので当然で、これが
**プローブは学習率のブラケットであってサイズの順位付けではない**という
ことの具体例です。順位は 48k step の本番 run が決めます。

**これは通常のハイパーパラメータ表以上の意味を持ちます。** 既定の
`SCRATCH_LR = 1e-4` では `scratch-base` は**崩壊**し、`scratch-small` は
自身の最適値から 0.31 nats 離れます。固定 LR のままこの実験を回すと、
最大セルがランダム推測線に乗ったまま「容量を増やしても効かない」という
結論になります — **平坦な指数が学習率の産物として作られる**わけです。
サイズごとの LR にすると `scratch-base` は逆に sweep 中の**最良セル**
(3.1200、どの小さいサイズよりも良い) になります。これがサイズ軸が本来
示すべきものです。

許容できる LR はサイズとともに下がりますが、滑らかではありません。small と
medium は崩壊閾値が同じで、しかも medium の方が良い (3.1714 対 3.2068) —
この 2 つはパラメータ数よりも**深さ**が違います (8x512 対 4x512)。

**注意**: 2000 step で勝つ LR が 16000 step でも勝つとは限りません。ここでは
ランダム初期化なので大きい LR は「寝る前に終わる」のではなく発散しますが、
逆に小さい LR は長く回せば追いつく可能性があります。結果は「この辺りに最適が
ある」というブラケットとして読んでください。

## 事前学習を変数から外す — `scratch-*` プリセット

`bert-tiny … bert-base` は同じ系統ですが、**サイズごとに事前学習量が違います**
（大きいモデルほど長く学習されている）。サイズ軸に事前学習量が相乗りしている
わけです。これを外すために、同じアーキテクチャを**ランダム初期化**で使う
プリセットを用意しています。

| プリセット | パラメータ数 | 層 × 幅 | config の出どころ |
|---|---|---|---|
| `scratch-tiny` | 4.4M | 2 × 128 | `prajjwal1/bert-tiny` |
| `scratch-mini` | 11.2M | 4 × 256 | `prajjwal1/bert-mini` |
| `scratch-small` | 28.8M | 4 × 512 | `prajjwal1/bert-small` |
| `scratch-medium` | 41.4M | 8 × 512 | `prajjwal1/bert-medium` |
| `scratch-base` | 109.5M | 12 × 768 | `bert-base-uncased` |
| `scratch-xl` | 234.6M | 16 × 1024 | `bert-large-uncased` + `num_hidden_layers: 16` |
| `scratch-large` | 335.4M | 24 × 1024 | `bert-large-uncased` |

1.9 decade（約 76 倍）の N をカバーします。`scratch-xl` は
「`bert-large` の 24×1024 が載らないように見えた」ときに間の点として
足したものですが、実際は載る（前述の「評価時メモリ」の節）ので、代替ではなく
**サイズ軸の内点**として残っています。`scratch-large` の 335M が H200 1 枚の
上端で、律速しているのは学習側です（B=256 で train_peak 101.81 GiB /
139.8 GiB）。B は in-batch 負例数そのものなので下げられず、その次のサイズは
今の構成では回せません。

`random_init: True` は `AutoModel.from_config()` を通り、重みを一切
ダウンロードしません（tokenizer だけ取ります）。学習率の既定値も
`train.SCRATCH_LR = 1e-4` に分かれています（`encoders.is_random_init()` で
振り分け）— 事前学習済みモデル用の 3e-5 は「既存の重みを壊さない」ための
値でここでの制約ではなく、ngram 用の 1e-3 は BERT を確実に崩壊させるためです。
**ただしこの 1e-4 が適切なのは最小サイズだけ**なので、実際の実験では
`cells:` でサイズごとに設定します（前述の
「4. LR がサイズに合っていない」参照）。

## データ順序: ファイルはインターリーブして読む

ChEMBL の TSV は activity ID で分割されているので、**1 ファイル = 1 つの
エンティティ集団**です。`StreamingTripleDataset` は既定を変える前は各
ファイルを最後まで読んでから次に移っていました。1 ファイルが 1000 万行ある
ので、25k step の run はどのファイルも読み終わらず、**学習ストリームは
「分布の列」であって 1 つの分布ではありませんでした**。これは非 i.i.d. なので、
勾配は常に直近のファイルの偏りを追いかけることになります。

**この変更はスケーリング則の測定に直接効きます。** 学習ストリームが「分布の列」
だと validation loss の推移に集団の切り替わりが乗り、それを容量の効果と
区別できません。`eval.build_candidate_pool` / `build_filter_index` も同じ
データセットを作るので、**評価のプールもこの変更の前後で中身が変わります**。
現在 `results/` にある run はすべて変更後のものです。

現在は既定でラウンドロビンに読みます（`interleave_files: True`、
`interleave_chunk` 行ずつ）。行の多重集合は完全に同一で、順序だけが変わります
（検証済み: 2 ファイル各 500 行で同一の多重集合）。効果は集団の混合で、
先頭 2000 行の平均 tail 長は連結読みで 26.6（1 ファイル目の値そのもの）、
インターリーブで 39.7（2 ファイルの中間）でした。

`--no-interleave-files` は **その旧挙動を再現するためだけの
フラグ**です。新しく実験するときに渡す理由はありません。
validation loss 側も既定でシャッフルします（`--no-valid-loss-shuffle` で無効化）
— シャッフルしないと in-batch 負例が「たまたま隣にいた行」になり、
本来の課題より簡単にも難しくもなります。

## サイズ軸の結果（`scaling_chembl`、48k step、7 サイズ）

出どころは `benchmark_scaling/results/chembl/scaling_chembl/`。
`scaling_table.md` に params / best valid loss / PF-days、`table.md` に最終
test 指標、`scaling_points.json` に全データ点とフィット結果、
`cell_scratch-*.log` に学習ログがあります。各セルは自身の実測 LR
（`config_scaling_scratch.yaml` の `cells:`）、B=256、`max_steps=48000`
（= 0.0182 エポック）です。

| セル | params | best valid loss | @step | best MRR (in-loop) | 最終 test MRR | 学習時間 |
|---|---|---|---|---|---|---|
| `scratch-tiny` | 4.4M | 3.1500 | 1,750 | 0.4610 | 0.4420 | 1.8 h |
| `scratch-mini` | 11.4M | 2.7727 | 31,500 | 0.5043 | 0.4944 | 1.9 h |
| **`scratch-small`** | 29.2M | **2.5010** | 46,500 | 0.5056 | 0.4890 | 2.1 h |
| `scratch-medium` | 41.8M | 2.8274 | 42,250 | 0.5174 | 0.4876 | 2.5 h |
| **`scratch-base`** | 110.1M | 2.9054 | 9,000 | 0.5015 | **0.5047** | 3.9 h |
| `scratch-xl` | 235.2M | 2.7496 | 41,750 | 0.5102 | 0.5000 | 6.2 h |
| `scratch-large` | 335.9M | 2.6975 | 42,250 | 0.5168 | 0.4948 | 8.6 h |

フィット（全 7 セル、フロンティア 23 点）:

| y 軸 | 単純冪則 b | R² | 床つき L∞ | 床つき b | R² |
|---|---|---|---|---|---|
| valid loss | −0.0286 | 0.671 | 2.436 | −0.1823 | 0.770 |
| 1 − MRR | −0.0374 | 0.647 | 0.4822 *(端)* | −0.4549 | 0.899 |

**読み方 — サイズ軸は単調ではありません。** loss は 29.2M
(`scratch-small`) が最良で、41.8M と 110M はそれより悪く、235M・336M で
また戻ります。MRR は 110M (`scratch-base`) が最良。**つまり 2 つの y 軸が
別々のサイズを勝者に選び、どちらの軸でも最大モデルは勝っていません。**
各セルは自身のプローブ最適 LR で回しているので、これは学習率の
アーティファクトではありません。

### 2 つの留保があり、どちらも「まだ結論ではない」側に効きます

**1. `best valid loss` は 192 点の最小値で、各点は 8 バッチしか見ていません。**
`valid_loss_batches: 8` は B=256 で 2,048 行です。終盤の推移を見ると

```
scratch-small: 2.5883 2.6095 2.5215 2.5922 2.5990   (報告値 2.5010)
scratch-large: 2.8117 2.8770 2.7888 2.8001 2.7750   (報告値 2.6975)
```

で、報告値は**プラトー上のノイズの谷**であって傾向の続きではありません。
min-of-192 はばらつきの大きいセルを系統的に有利にするので、**0.05 nats 程度の
差はこの統計量では区別できません**。「29.2M が最良」を 0.3 nats の主張として
読まないでください — プラトーどうしなら 2.59 対 2.78 で、差はもっと小さい。
順位を詰める段階に入るなら `valid_loss_batches` を上げるか、最小値ではなく
終盤の平均で比較する必要があります。

**2. 最適点の位置はセルごとにばらばらで、予算律速とは言い切れません。**
best loss の到達 step は 1,750 / 31,500 / 46,500 / 42,250 / 9,000 / 41,750 /
42,250（48,000 のうち）。48,000 に迫るのは `small` だけ、`tiny` と `base` は
極端に早く底を打ってその後 0.1〜0.3 nats 悪化します。**「予算が律速する」は
run ごとに測り直す性質のもの**で、固定の注意書きにはできません。

### 二相性（early stopping を入れてはいけない理由の実測）

validation loss は最初の 4 分の 1 で底を打ったあと上昇し、その後さらに
良い最小値まで落ちます。現 run での前半 25% の最良と、それ以降の最良:

| セル | 前半 25% の最良 | それ以降の最良 | 差 |
|---|---|---|---|
| `scratch-small` | 2.8010 @12,000 | **2.5010** @46,500 | **+0.3000** |
| `scratch-large` | 3.0071 @11,000 | **2.6975** @42,250 | **+0.3095** |
| `scratch-xl` | 2.9347 @12,000 | **2.7496** @41,750 | **+0.1850** |

patience ベースの early stopping はこの上昇中に確実に発火し、
**0.19〜0.31 nats 手前で打ち切ります**。しかも `scratch-large` では
同じ区間で **MRR は 0.4640 → 0.5105 と改善し続けて**いるので、
loss の悪化を停止条件にすると順位性能がまだ伸びている最中に止めることに
なります。詳細な根拠は後述の「early stopping を入れてはいけない」節。

## 別コーパスでの確認 — ランダム多ソースリスト

**ここまでの指数はすべて ChEMBL で測ったものです。** ChEMBL は単一ドメイン
（全行が活性アサーション、1 ファイル = 1 activity ID 集団）なので、得られた
指数がタスクの性質なのかそのドメインの性質なのかは、それだけでは分かりません。
そこで**別のコーパスで同じ軸を測る**ための一式を用意しています。

```bash
# 1. data/ 全体からランダムに抽出したリスト（ChEMBL 以上の規模）
python preprocess_list/make_lists.py --source all --sample-bytes 150GiB \
    --min-bytes 4096 --random-seed 0 --validate --out-dir list_random

# 2. そのコーパスで学習率を測り直す（ChEMBL の値は流用しません。後述）
bash benchmark_scaling/run_lr_probe_random.sh         # 3e-6〜1e-4、小さい 5 サイズ
bash benchmark_scaling/run_lr_probe_random_extend.sh  # 3e-4 / 1e-3 へ上方拡張
bash benchmark_scaling/run_lr_probe_random_large.sh   # xl + large、1 GPU 1 セル

# 3. 同じサイズ軸を回す（7 サイズ 4.4M〜335.4M、約 8 時間 / 2 GPU）
bash benchmark_scaling/run_scaling_random.sh

# 4. 2 つの run を同じ土俵に載せて指数を比較
python benchmark_scaling/compare_corpora.py \
    --run benchmark_scaling/results/chembl/scaling_chembl \
    --run benchmark_scaling/results/random/scaling_random
```

`list_random` は 22,886 本の TSV からファイル一様に引いた 196 本で、
**11 ソース**（uniprot / wikidata / biosampleplus / expressionatlas /
bioportal / biomodels / clinvar / oma / pubtator / chembl / pubmed）に
またがります。6 列の契約はどのソースも同じですが、内容はまったく違います
（タンパク質の文献参照、Wikidata の一般知識、シーケンシング run …）。
規模は ChEMBL 以上です。

### 学習率の差はサイズ依存で、一律ではありません

7 サイズすべてを内点で測った結果:

| size | random | ChEMBL | 比 |
|---|---|---|---|
| tiny 4.4M | 1e-3 | 1e-4 | **10x** |
| mini 11.4M | 3e-4 | 3e-4 | 1.0x |
| small 29.2M | 1e-4 | 1e-4 | 1.0x |
| medium 41.8M | 1e-4 | 3e-5 | 3.3x |
| base 110.1M | 3e-5 | 3e-5 | 1.0x |
| xl 235.2M | 1e-5 | 1e-5 | 1.0x |
| large 335.9M | 1e-5 | 1e-5 | 1.0x |

**2 コーパスは 7 サイズ中 5 つで同じレートを要求し、違うのは tiny（10 倍）と
medium（3.3 倍）だけ**です。差は小さい側に集中していて、上位 3 サイズは
完全に一致します。

それでも無視はできません。ChEMBL で崩壊するレート（small と medium の 3e-4）が
ここでは正常に学習するので、ChEMBL の値を機械的に流用すれば一部のセルを
最適から外して回し、その平坦化を「コーパスの差」と誤読します。

なお端で勝ったセルはすべて格子を広げて内点であることを確認しています
（ChEMBL は mini / base / large、random は tiny / large）。
`summarize_lr_probe.py` は端で勝ったセルを検出して警告を出すので、
それが消えるまで格子を広げてください。

| | ChEMBL train | random train |
|---|---|---|
| ファイル数 | 85 | 150 |
| サイズ | 105.35 GiB | **119.80 GiB** |
| 行数（実測） | 674,265,105 | **720,653,921** |

valid / test も含めた全体（いずれも `data.count_rows` の実測）:

| split | random | ChEMBL |
|---|---|---|
| train | 150 files / 720,653,921 | 85 / 674,265,105 |
| valid | 18 files / 33,116,566 | 6 / 25,764,240 |
| test | 28 files / 151,169,502 | 4 / 40,000,000 |

どちらの尺度でも ChEMBL 以上です。もっとも、24k step × B=256 = 614 万例は
どちらのコーパスでも **0.01 エポック未満**なので、サイズ軸の実験にとっては
両者とも「実質無限のデータ」であり、行数の差は結果に効きません。効くのは
**中身**です。

### 学習率は必ずコーパスごとに測り直します

許容される学習率は**(アーキテクチャ, コーパス) の性質**であって
アーキテクチャだけの性質ではありません。ChEMBL で測った値を別コーパスに
流用すると、「このコーパスはスケールの仕方が違う」と「ChEMBL の学習率が
このコーパスには合っていない」が交絡します — サイズ軸について
このファイルが延々と記録しているのと同じ失敗です。`run_lr_probe_random.sh`
がそのための probe です。**ヘッド結線を変えたときも同じ理由で測り直しが要ります**
— 許容レートは (アーキテクチャ, コーパス, ヘッド結線) の性質で、どれか 1 つを
動かして表を流用すれば同じ交絡が戻ります。

### 2 つの run は「そのまま」比較できません

`compare_corpora.py` が既定で 2 つの補正を入れます。**どちらも再学習なしで
できます** — セルの全 validation 点が `cell_*.log` に残っているからです。

- **共通のサイズだけに絞る。** 現状はどちらも 7 サイズなので実質素通りですが、
  N の範囲が違うフィットは別の推定量なので既定で揃えます。
- **短い方の step 予算で切る。** ChEMBL 側は 48,000 step、random 側は
  24,000 step。フロンティアは回すほど動くので、揃えないと予算差を
  コーパス差と読み違えます。

補正前（`as-run`）も併記されるので、補正が何を動かしたかが見えます。
参考値として、ChEMBL の run を 5 サイズ・step ≤ 24,000 に制限し直した
ベースラインは

| y 軸 | b | R² |
|---|---|---|
| valid loss | −0.0193 | 0.642 |
| 1 − MRR | −0.0331 | 0.626 |

です（同じ run を 7 サイズ・48k のまま読むと −0.0214 / −0.0307）。

### 実測結果（`scaling_random`、24,000 step × B=256、7 サイズ）

| セル | params | best valid loss |
|---|---|---|
| `scratch-tiny` | 4.4M | 2.8293 |
| **`scratch-mini`** | 11.4M | **2.7142** |
| `scratch-small` | 29.2M | 2.7162 |
| `scratch-medium` | 41.8M | 2.7597 |
| `scratch-base` | 110.1M | 2.8871 |
| `scratch-xl` | 235.2M | 3.0464 |
| `scratch-large` | 335.9M | 3.0075 |

予算をそろえた比較（`compare_corpora.py`、共通 7 サイズ・step ≤ 24,000）:

| 軸 | ChEMBL b | R² | random b | R² |
|---|---|---|---|---|
| valid loss | −0.0244 | 0.666 | **−0.0097** | 0.158 |
| 1 − MRR | −0.0361 | 0.561 | **−0.0117** | 0.390 |

**結論: このコーパスではサイズ軸がほとんど効きません。** 指数は loss で
2.5 分の 1、`1 − MRR` で 3 分の 1 です。**random は 11.4M で底を打ち**
（2.7142）、そこから 335.9M まで単調に悪化します。ChEMBL は 29.2M で
底を打つので、**最適サイズが 2.5 分の 1 に小さい**。

**機序**: uniprot / wikidata の URI は構造が非常に規則的
（`http://purl.uniprot.org/uniprot/<ID>` の反復）なので小さいモデルでも
取り切れてしまい、追加容量に買うものが残りません。ChEMBL の活性
アサーションはより難しく、容量が効き続けます。**つまりスケーリング指数は
タスクとアーキテクチャだけの性質ではなく、コーパスの性質でもあります。**

#### この数値を引用するときの留保

- **random 側の R² は 0.16〜0.39 と低い**です。R² が低いこと自体が
  「この軸に法則が乗っていない」という結果ですが、**b の値そのものを
  精密な推定量として引用はできません。** 頑健なのは (1) **最適サイズが
  11.4M と小さい**、(2) **そこから単調に悪化する**、の 2 点です。
- `best valid loss` は `valid_loss_batches: 8`（B=256 で 2,048 行）の測定の
  最小値なので、0.05 nats 程度の差は分離できません。
- 学習率は**両コーパスで別々に測ってあり、7 サイズすべて内点**です。
  したがってこの差は「ChEMBL のレートが転移しなかった」ではありません。
- 2 つの run は step 予算が違う（48k 対 24k）ので、表を並べるのではなく
  `compare_corpora.py` の揃えた数字を使ってください。

## 出力

```
benchmark_scaling/results/chembl/<experiment-name>/
  cell_<protocol>_<tag>.log     学習ログ（scaling はここから読む）
  run.log, commands.jsonl       実行の記録
  table.md, report.html         通常のベンチマークレポート
  scaling_table.md              セルごとの params / 最良 loss / PF-days
  scaling_report.html           上記 8 種のプロット + 系統別フィット表
  scaling_points.json           全データ点とフィット結果（機械可読）
                                `axes.<loss|mrr_error>.{power_law,saturating,families}`
```

LR プローブは別ディレクトリに出ます。

```
benchmark_scaling/results/lr_probe_<name>/
  lr_probe.json                 全 (encoder, lr) の validation 履歴
  summary.txt                   表 + そのまま貼れる cells: ブロック
  logs/<encoder>_lr<rate>.log   各セルの学習ログ
```
