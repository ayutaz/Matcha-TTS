# vast.ai セットアップ・前処理トラブルシューティング

2026-07-06 のフルスクラッチ環境構築（Phase 0-1）で実際に遭遇した問題と恒久対策の記録。
新しいインスタンスを立てる際は、まず本ドキュメントの「チェックリスト」を確認すること。

関連: `docs/next-steps-plan.md`（実行ランブック）、`scripts/setup_vastai.sh`（自動セットアップ）

---

## 新インスタンス選定・構築チェックリスト

1. **オファー選定**: 「US/UK表記・$1.33-1.55・252GB RAM・~1700Mbps・cuda 13.0」の同一スペック群（host 557452 / 572017 / 572018 / 579265 等）は**偽装ロケーションの中国経由ホスト**。回避すること。host 120872（Taiwan）は実績あり
2. **起動後すぐHF接続を確認**（`setup_vastai.sh` のステップ0が自動検査する）:
   ```bash
   curl -fsS --max-time 15 "https://huggingface.co/api/models?limit=1" -o /dev/null && echo HF_OK
   ```
3. SSH鍵認証が失敗し続ける場合はホスト不良を疑う（下記 #2）
4. セットアップは `bash /root/bootstrap.sh`（curl経由でsetup_vastai.shを実行）。冪等なので失敗したら修正後に再実行してよい
5. 前処理の実行は必ず **nohup分離**（SSH切断でプロセスが死ぬのを防ぐ）:
   ```bash
   nohup bash -c "bash /root/phase1.sh; echo PHASE1_EXIT=$?" > /root/phase1.log 2>&1 &
   ```

---

## 1. huggingface.co がDNS汚染+SNI遮断されるホスト（GFW経由の偽装ロケーション）

**症状**
- `hf auth login` / `hf download` が `Connection reset by peer` や `Connection refused` で失敗
- GitHub や PyPI へは正常に接続できるため、セットアップ途中まで気づけない

**診断**
```bash
getent hosts huggingface.co
# → 2a03:2880:...:face:b00c:... (MetaのIP) が返る = DNS汚染
cat /etc/resolv.conf
# → 中国系DNS (202.100.x.x 等) や tailscale search domain (taild5a16e.ts.net)
curl --resolve huggingface.co:443:<正しいCloudFront IP> https://huggingface.co
# → 正しいIPでもTLSリセット = SNIベース遮断（/etc/hostsでは回避不能）
```
判別ポイント: SNI `cdn-lfs-us-1.hf.co` は通るのに SNI `huggingface.co` だけ落ちる → GFWの選択的遮断。

**対策**
- **回避不能。インスタンスを破棄して別ホストに乗り換える**（checkpointバックアップがHF必須のため）
- 恒久対策として `setup_vastai.sh` にHF接続プリフライト（ステップ0）を追加済み（commit `163ea3f`）
- 同一オペレーターは複数ロケーションを名乗る（resolv.confの同一tailnetで判別可能）。同価格・同スペックの兄弟オファーも全て回避

## 2. SSH鍵認証が通らないホスト

**症状**: `Permission denied (publickey)`。`vastai attach ssh` は「already associated」を返し、rebootしても直らない。

**対策**: 10分粘って駄目なら破棄して別ホストへ（host 467312 / California で発生）。

## 3. segmentation-kit にLinux用Juliusバイナリが無い（空.labサイレント成功）

**症状**
- Julius alignmentが**異常に速く**（500+ files/s）「成功」する
- .labファイルは生成されるが**全て0バイト**
- precomputeが全件 `Empty .lab file` でスキップ → .pt が0件

**原因**
- segmentation-kit の `bin/` には Windows用 `julius-4.3.1.exe` しか無く、`segment_julius.pl` が呼ぶ `./bin/julius-4.3.1` がLinuxに存在しない（`sh: ./bin/julius-4.3.1: not found`）
- それでも perl スクリプトは空の .lab を作って exit 0 する
- 旧開発機では誰かが手動でシンボリックリンクを置いていた（undocumented）ため顕在化しなかった

**恒久対策（commit `11e5529`）**
- `run_segkit_batch` が システムの `julius`（apt版4.2.2）を `bin/julius-4.3.1` として自動シンボリックリンク
- 0バイトの .lab を成功ではなくエラーとして報告

## 4. 「ヴ」を含む発話424件がJulius整列不能

**症状**: `Error in loading model`（Julius stderr）で LOANWORD128 / VOICEACTRESS100 の一部が全話者分失敗。

**原因**
- yomi2voca.pl は「ゔ」（U+3094）を変換できない（EUC-JP時代の「う゛」規則しか無い）→ 不正な音素が文法生成を壊す
- 一方 pyopenjtalk-plus 自身は ヴ を `b` 音素で出力する（`ノヴェンバー` → `n o b e N b a a`）ため、モデル側の正解は最初からバ行

**恒久対策（commit `b777f17`, `ae30c55`）**
- `matcha.text.julius_to_pyopenjtalk.normalize_vu_kana` を追加: ヴァ/ヴィ/ヴャ/ヴュ/ヴョ/ヴェ/ヴォ/ヴ → バ行
- かな変換4箇所（prepare_jvs / run_optimized_pipeline / run_full_alignment_pipeline / prepare_julius_input）に適用
- 防御として duration整合に v↔b 対応を追加（`_corresponds` / `_to_julius`）

## 5. `--use-shm` がpt-output-dirと同一パスのとき生成物を自壊

**症状**: `Failed to copy to /dev/shm: [Errno 2] No such file or directory` と共に、**生成直後の.ptが全消失**。

**原因**: CLAUDE.mdの推奨コマンドは `--pt-output-dir /dev/shm/jvs_precomputed_aligned` + `--use-shm` の組み合わせだが、shmコピー処理が「既存の宛先をrmtree → src(=同一パス)をcopytree」するため、自分自身を削除してからコピーしようとする。

**恒久対策（commit `0d46e29`）**: src と dst の resolve() が一致する場合はコピーをスキップ。

## 6. setup_shm_cache.sh のパスハードコード

**症状**: `ERROR: Source not found: /data/Matcha-TTS/data/jvs/wavs`（旧開発機のパス）。

**恒久対策（commit `618562a`）**: スクリプト位置からリポジトリルートを自動導出（`MATCHA_ROOT` envで上書き可能）。

## 7. resampleワーカーのスレッド多重過剰（100倍遅くなる）

**症状**: `prepare_jvs.py` のResamplingが2 files/s（正常時は数百files/s）。load averageが800超（cgroup上限69コアに対して）。

**原因**: ProcessPoolExecutorの各ワーカーがBLAS/librosaの内部スレッドをコア数分起動し、16ワーカー×数十スレッドでコンテキストスイッチ崩壊。コア数の多いマシン（144コア等）で顕在化する。

**対策**: 並列プロセス実行時はワーカー内スレッドを1に固定する:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python scripts/prepare_jvs.py ... --num-workers 48
```

## 8. Julius並列数が高すぎるとモデルロードが失敗する

**症状**: 32並列で424件が `Error in loading model`（うち一部は再試行で成功）。

**対策**: `run_optimized_pipeline.py --num-workers 16`（開発機実績値）を使用。失敗分は再実行すれば冪等にリトライされる。

## 9. セットアップ関連の小物

| 問題 | 症状 | 対策（済み） |
|------|------|-------------|
| Cコンパイラ不在 | `command 'cc' failed`（Cython拡張ビルド） | aptに `build-essential` 追加（`b844b1d`） |
| espeak-ng不在 | `make test` の英語クリーナーテスト失敗 | aptに `espeak-ng` 追加 + `PHONEMIZER_ESPEAK_LIBRARY` 設定（`7fb7ea9`） |
| HFトークンのCR混入 | `hf auth login` が不正トークンで失敗 | Windows経由の転送後は `tr -d '\r\n'` で除去、`printf %s` で書き込む |
| リポジトリが古いまま | 再実行時に旧コードでsync | setup_vastai.shが `git fetch + merge --ff-only` で最新化（`aa704d8`） |
| extras漏れ | `uv sync --all-groups` はextrasを含まない | pyopenjtalk-plus等を本体依存へ昇格（`0cd2a31`） |

## 10. Juliusアライメントの既知の欠損（許容済み）

修正後も **24件（0.18%）** は整列不能で.pt生成から除外される:
- 全角英字（Ａ/Ｈ）がpyopenjtalkのkana出力に残る
- LOANWORD128の特殊モーラ（てゃ/てゅ/てょ/うょ/るぁ/ぐぉ）— yomi2voca未対応
- 全角マイナス「−」が句読点除去に残る
- 通常文でのJulius探索失敗（音声側、1件）

学習には 12,973 / 12,997 件を使用する。これ以上の追跡は費用対効果が悪いため許容とする。
