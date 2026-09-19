# preprocess_list — `list_*/` ファイルリストの生成

`kgfm train` / `kgfm bench` が読む **ファイルリスト**（`train.txt` /
`valid.txt` / `test.txt`、1 行 1 TSV パス）を作るスクリプトです。

| ファイル | 役割 |
|---|---|
| `make_lists.py` | 本体。`data/<source>` を走査して 3 分割し、リストを書く |
| `make_all.sh` | リポジトリに入っている `list_chembl` / `list_large` の検証と再生成 |

```bash
# list_chembl を作り直す（data/chembl が対象。既存のものと完全に一致します）
python preprocess_list/make_lists.py --source chembl --out-dir list_chembl

# 別のデータソースに切り替える
python preprocess_list/make_lists.py --source uniprot --out-dir list_uniprot
python preprocess_list/make_lists.py --source chebi,rhea --out-dir list_chem
python preprocess_list/make_lists.py --source all --out-dir list_everything

# 書く前に中身を見る / 既存リストが今もこのルールで作れるか検証する
python preprocess_list/make_lists.py --source chembl --dry-run --count-rows
python preprocess_list/make_lists.py --source chembl --check list_chembl

# 使えるソース一覧（data/ の直下ディレクトリ）
python preprocess_list/make_lists.py --list-sources
```

## なぜリストが必要か（`kgfm train` は自分で分割できるのに）

`kgfm train` はリストを渡さなければ `--data-root` / `--pattern` で見つけた
ファイルをハッシュバケットで自動分割します。それでもベンチマークの config が
すべて明示的なリスト（`list_chembl/train.txt` など）を指しているのは、
**`data/` が生きたミラーだから**です。ファイルが 1 つ増えたり消えたりすると
train と test の境界が黙って動き、2 つの run が比較できなくなります。
リストはその境界を**凍結**するためのもので、このスクリプトはそれを作る手段、
`--check` は「リポジトリに入っているリストが今もそのルールで再現できるか」を
確認する手段です。

## 分割モード

**このモードの選択は交換可能ではありません。** 既存のリストがどちらで
作られたかは `make_all.sh` を見てください。

### `--split hash`（既定）

`kgfm.data.split_files_three_way` を**そのまま呼びます**。つまり
「リストを渡さなかったときに trainer 自身が作る分割」と完全に一致します。
既定の `--n-buckets 10 --valid-buckets 1 --test-buckets 1` が 80/10/10 です。

- **ファイルが増減しても既存ファイルの割り当てが動きません**（パス文字列の
  ハッシュだけで決まるため）。だから既定にしています。
- **ハッシュ対象はパス文字列**です。したがってこのスクリプトはリポジトリ
  相対のパス（`data/chembl/latest/x.tsv`）を書き、hash モードでは絶対パスの
  `--data-root` を**拒否します** — `/data1/.../data/chembl/latest/x.tsv` は
  別のバケットに落ち、既存のどのリストとも、trainer 自身の自動分割とも
  一致しない分割になってしまうためです。

### `--split sequential`

ソート順に先頭から `--train-files` / `--valid-files` / `--test-files` 個ずつ
取ります（`--ratios 0.8,0.1,0.1` でも指定可）。「特定の少数ファイルを
狙って使いたい」スモークセット向けで、`list_large` はこれで作られています。

### `--split random`

`--random-seed` でシャッフルしてから sequential と同じに切ります。
**(ファイル集合, seed) の組では決定的ですが、seed だけでは決定的ではありません**
— 上流でファイルが 1 つ増減すると全ファイルの割り当てが動きます。凍結された
リストを作る用途では hash の方が安全です。

### `--sample-bytes` — サイズ指定のランダム抽出

**ファイル一様のランダム抽出で、指定バイト数に達するまで引きます。**
`--source all` と組み合わせると `data/` 全体からのランダム標本、つまり
**多ソース混合コーパス**になります。

```bash
# ChEMBL 以上の規模の、ランダム多ソースリスト（list_random/ がこれ）
python preprocess_list/make_lists.py --source all --sample-bytes 150GiB \
    --min-bytes 4096 --random-seed 0 --validate --out-dir list_random
```

- サイズは `150GiB` / `105G` / `2TB` / 生のバイト数で指定できます。
- 抽出は**バイト一様ではなくファイル一様**です。全ファイルが等確率で引かれる
  ので、結果は「コーパスのランダム標本」になります（バイト一様にすると
  大きいファイルだけが引かれます）。この環境のファイルは 0 B〜2.1 GiB と幅が
  あるので、達成サイズは目標を最大 1 ファイル分超過し、ファイル数は成り行き
  です。それが抽出のばらつきそのものです。
- **`--validate` は抽出中に評価されます。** プール 2 万件を全部読むのではなく
  引いた 200 件程度だけを読むので、100 倍速い（実測 3 秒）うえに
  「目標サイズを使えるファイルで満たす」という意味も保てます。
- `--random-seed` で再現できます。`--sample-bytes` を渡すと
  `--max-files` は無視されます。

`--show-sources` を付けると split ごとのソース内訳が出ます。**多ソース
リストではこれを必ず確認してください** — ファイル単位分割なので、あるソースが
まるごと valid や test に入ることがあり、そうなると評価は「学習していない
ドメインへの転移」を測っていることになります（kgfm は inductive 設計なので
それが目的の場合もありますが、事故で起きてはいけません）。valid や test が
単一ソースになった場合は警告が出ます。

### `--train-source` / `--valid-source` / `--test-source`

「**X で学習して Y で評価する**」形。どれか 1 つでも指定すると `--split` は
無効になり、各 split が自分のソースから `--<name>-files` 個を取ります
（未指定なら全部）。train → valid → test の順に埋め、**先に取られた
ファイルは飛ばす**ので、同じソースを 2 つの split で共有した場合は
リークではなく分割になります。

```bash
# 学習は大きい amrportal、検証/テストは小さい biomodels（スモーク用に速い）
python preprocess_list/make_lists.py \
    --train-source amrportal --train-files 3 \
    --valid-source biomodels --valid-files 1 \
    --test-source  biomodels --test-files 1 \
    --out-dir list_smoke
```

同じソースを共有して個数を指定しないと、先の split が全部取ってしまって
後の split が空になります。その場合はエラーで止まります（黙って空のリストを
書きません）。

## ファイルの絞り込み

| フラグ | 用途 |
|---|---|
| `--max-files N` | ソート・絞り込み後、分割前に先頭 N 件に制限 |
| `--exclude GLOB` | パスまたはベース名がマッチしたら捨てる（複数指定可） |
| `--include GLOB` | マッチしたものだけ残す（複数指定可） |
| `--min-bytes N` | N バイト未満のファイルを捨てる |
| `--validate` | 各ファイルの先頭を**ローダ自身のパーサ**で読み、6 列の有効行が 1 つも取れないファイルを捨てる |

`--validate` は `kgfm.data._iter_tsv_rows` を使うので、「ここで 0 行」は
「学習中も 0 行」を意味します。`--probe-rows`（既定 5）行しか読まないので
安価です。

`data/chembl` には活性データ以外に `void.tsv`（VoID のデータセット記述）と
`cco.tsv`（オントロジー定義）が入っており、**現在の `list_chembl` はこれらも
含んでいます**（どちらも 6 列の形式は満たしています）。外したい場合は
`--exclude void.tsv --exclude cco.tsv` を渡してください。ただし
**除外するとハッシュ分割の中身が変わる**ので、既存の結果とは比較できなく
なります。

## 安全側の作り

- **既存の `--out-dir` は `--force` なしでは上書きしません。** 代わりに
  `--check` を案内します。
- **split 間でファイルが重複したらエラー**にします。kgfm はファイル単位で
  分割するので、1 ファイルの共有はその中の全行が両側に乗ることを意味します
  （行単位のリークではなく、エンティティ集団まるごとのリーク）。
- 書き出したリストの先頭には**生成コマンドがコメントで入ります**
  （`kgfm.data.read_file_list` は `#` 行を飛ばすので実行には影響しません）。
  手書きリストとバイト一致させたいときだけ `--no-header` を渡してください。
- `--count-rows` は `kgfm.data.count_rows` のキャッシュ
  （`~/.cache/kgfm/rowcounts.json`、キーは `(パス, サイズ, mtime)`）を使うので、
  一度数えたコーパスでは即時、初回でも約 1.3 GB/s です。

## リポジトリに入っているリスト

```bash
bash preprocess_list/make_all.sh          # 検証のみ（既定・何も書かない）
bash preprocess_list/make_all.sh --write  # 実際に作り直す
```

検証が既定なのは、これらのリストが `benchmarks/results/` にある全結果の
train/test 境界を定義しているからです。作り直すと黙って再分割されるので、
明示的なフラグを要求しています。

| リスト | 作り方 | 中身 |
|---|---|---|
| `list_chembl/` | `--source chembl`（hash 既定） | 95 ファイルを 85 / 6 / 4。674,265,105 / 25,764,240 / 40,000,000 行 |
| `list_large/` | `--source amrportal,bacdive,biomodels --split sequential --max-files 60 --train-files 40 --valid-files 10 --test-files 10` | 60 ファイルを 40 / 10 / 10 |
| `list_small/` | **手作り。ルールで再現できません** | 5 ファイル。train は amrportal (8 KB + 1.1 GB + 1.1 GB)、valid/test は biomodels (140 KB / 904 KB) |

`list_small` はスモーク用に「学習は流し続けられる大きいファイル、検証/テストは
一瞬で終わる小さいファイル」という意図で選ばれたもので、単一のルールでは
出てきません（valid が biomodels の 30 番目、test が 49 番目、という選び方に
規則性がありません）。`make_all.sh` も触りません。同じ意図の**ルールベースの
代替**は上の `--train-source` / `--valid-source` の例で、こちらは
`--check` で検証できる形になります。

## ChEMBL の分割は inductive です

ChEMBL の TSV は activity ID で分割されているため、train ファイルに現れる ID は
他のファイルにほとんど現れません。したがって `list_chembl` の分割は
**ファイル単位であると同時に実質 inductive** で、`kgfm bench prep
--strict-transductive` を使うと test がほぼ空になります。これは分割の作り方の
問題ではなくコーパスの性質です（`benchmarks/README.md` の「ChEMBL KG の構築」
節を参照）。

同じ理由で、**1 ファイル = 1 つのエンティティ集団**でもあります。
学習ストリームがファイルを跨いで混ざるかどうかは
`--interleave-files`（`kgfm train` 側の既定は有効）が決めます。詳細は
トップレベル README の「ファイルはインターリーブして読む」節にあります。
