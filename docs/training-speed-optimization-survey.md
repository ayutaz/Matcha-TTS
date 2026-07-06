# 学習高速化 調査レポート（torch / Python / CUDA、2026-07-06）

4x RTX 5090 での `jvs_aligned` 学習（20Mパラメータ、DDP、可変長メル）を対象に、
最新OSS・論文・実測に基づく高速化候補を調査した。**採否の判断はこのドキュメントを見て行う**
（本レポート作成時点では調査のみで、「適用済み」セクション以外は未適用）。

調査ソース: PyTorch 2.8-2.10公式資料 / NVIDIA cuDNN・NCCLリリースノート / F5-TTS・CosyVoice2・
Fish-Speech・StableTTS等のOSS実装 / フローマッチング学習効率化論文（2024-2026）/ 本インスタンスでの実測。

---

## 1. 実測ベースライン（2026-07-06、インスタンス44019256）

### 環境検証（問題なし）

| 項目 | 実測値 | 判定 |
|------|--------|------|
| torch | 2.10.0+cu128（CUDA 12.8） | ✅ |
| `torch.cuda.get_arch_list()` | `sm_120` を含む | ✅ **Blackwellネイティブ**（PTX JITフォールバックなし。cu126だと起動すらしない） |
| cuDNN | 9.10.2 | ○（9.13+にBlackwell attention +5-10%の改善あり → cu130 wheel化で取り込める、優先度低） |
| ドライバ | 575.57.08 | ✅（cu128要件570+を満たす） |
| 電力制限 | 575W/枚（デフォルト無制限） | 要検討（→ 候補C-2） |

### FP32 vs bf16-mixed の実測比較（同一データ・同一seed系列）

| 指標 | FP32 (TF32有効) | bf16-mixed | 差 |
|------|------|------|-----|
| steps/sec（起動込み平均） | 2.51 | 2.79 | **+11%** |
| loss/train @ step 399 | 5.854 | 5.626 | 同等 |
| dur_loss @ step 399 | 3.218 | 2.999 | 同等 |
| diff_loss @ step 399 | 1.520 | 1.514 | 同等 |
| NaN / 発散 | なし | なし | — |

**考察**: bf16の利得が+11%にとどまるのは、20Mパラメータの小型モデルでは行列積が小さく
**GEMM律速ではない**ため（nvidia-smiのGPU利用率も20-60%で変動 = カーネル起動/CPU側の隙間が大きい）。
これは調査班3系統の分析（「LLM向けの派手なレバーは小型モデルに効かない」）と整合する。
lossカーブはFP32と統計的に同等でありbf16の品質リスクは現時点で観測されていない。

**現在のスループットでの見通し**: ~2.8 steps/s → 96 steps/epoch → **2500ep ≈ 24時間、~$45**。

### 今日適用済みの変更

- `trainer.precision=bf16-mixed` + `model.optimizer.fused=false`
  - **教訓**: fused AdamW は mixed precision + gradient clipping と併用不可（Lightningが
    `RuntimeError: does not allow for gradient clipping` で即クラッシュ。CLAUDE.mdのFP16時の記録と同根）
- TF32は元々有効（`train.py:40-41`）、DDP最適化（static_graph等）も適用済み

---

## 2. 候補一覧（採否判断用サマリー）

推奨度: ◎=強く推奨 / ○=推奨 / △=実測次第・投機的 / ×=非推奨

| # | 候補 | 期待効果 | リスク | 工数 | 推奨度 |
|---|------|---------|--------|------|:---:|
| A-1 | プロファイリングで律速特定（torch profiler / DCGM） | 投資先の確定 | なし | 小 | ◎（他の前提） |
| A-2 | Regional torch.compile（Conformer/decoderブロック単位） | 1.1〜1.5x | 中（再コンパイル・DDP相互作用） | 中 | ○ |
| A-3 | cuDNN SDPA backend 明示有効化 | attention区間で数十%（全体では小〜中） | 小（loss検証必須） | 小 | ○ |
| B-1 | frame-based dynamic batching（合計フレーム上限バッチ） | 10〜30% | 中（sampler改修・有効バッチ変動） | 中 | ○ |
| B-2 | cudnn.benchmark=True（bucket化shapeなら有効に働く可能性） | 数% | 小（bucket数多で逆効果） | 極小 | △実測 |
| B-3 | pin_memory + non_blocking H2D転送 | GPU飢餓時のみ | 小 | 小 | △実測 |
| C-1 | NCCL疎通確認 + `NCCL_P2P_DISABLE=1`（5090はP2P物理不可） | 安定性（速度は微小） | なし | 極小 | ○ |
| C-2 | 電力制限 `nvidia-smi -pl 450` + persistence mode | 熱安定→クロック変動減（速度は中立〜微増） | なし | 極小 | ○ |
| C-3 | cu130 wheel + 新cuDNN 9.1x | 数% | 小（ドライバ要件580+） | 小 | △ |
| X-1 | FP8学習（TransformerEngine / torchao） | 効果なし〜負（20Mでは overhead倒れ） | 大（品質） | 大 | × |
| X-2 | FlashAttention-3 / 4 | — | sm_120非対応（FA3はHopper専用） | — | × |
| X-3 | CUDA Graphs / reduce-overhead | 限定的 | 大（DDP+可変長で最も壊れやすい） | 大 | × |
| X-4 | Muon / schedule-free optimizer | 収束10-15%改善の報告 | 大（安定レシピと正面衝突） | 中 | ×（本番） |
| X-5 | 真のsequence packing（複数発話連結） | LLMでは2-3x | conv decoderに構造的不適合 | 特大 | × |
| X-6 | logit-normal等のtimestep sampling変更 | — | 品質劣化の自社実績あり | — | ×（uniform維持） |
| X-7 | EMA更新間引き | ~0%（20Mでは元コスト極小） | decay補正ミス | 小 | ×（不要） |
| X-8 | batch 64/GPU（有効バッチ256） | 通信/計算比改善 | 中（原論文レシピ逸脱） | 極小 | △（品質と相談） |

---

## 3. 各論

### A-1. プロファイリング（すべての前提）

GPU利用率が20-60%で変動しており、律速がGPU計算・カーネル起動・CPU collate・DDP通信の
どれかを確定しないと投資を外す。`nvidia-smi` のutilizationは「何かが動いた時間の割合」で
SM占有率とは別物なので信用しない。

- torch profiler: 1エポックだけ `profile(activities=[CPU, CUDA])` でカーネル間ギャップを確認
- DCGM: `DCGM_FI_PROF_SM_ACTIVE`(1002) / `PIPE_TENSOR_ACTIVE`(1004) / `DRAM_ACTIVE`(1005)
- 判定: utilization高 & tensor_active低 → カーネル起動律速 → A-2（compile）が効く

参考: https://arthurchiao.art/blog/understanding-gpu-performance/ /
https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html

### A-2. Regional torch.compile（フルモデルcompileはNGのまま）

- フルモデルcompile + DDP + 可変長は**2026年時点でも未解決**（DDPOptimizerがsymbolic shapeを
  具体値にrefine → 系列長ごとに再コンパイル → cache_size_limit到達で停止。issue #140229）。
  2026-04の「compile無効化」判断はフルcompileに関しては今も正しい
- **打開策**: 繰り返しブロック（Conformerブロック、decoderのTransformerブロック）単位で
  `torch.compile` を適用する Regional Compilation。DDPのラップ外・dynamic shapeガードの
  スコープ内に閉じるため両問題を回避。公式チュートリアルでは「フルcompileと実行速度差は僅少、
  コンパイル時間~11倍削減」
- 併用: `torch._dynamo.mark_dynamic(x, seq_dim)` で系列次元をsymbolic固定、
  `torch._dynamo.config.cache_size_limit` 引き上げ、`TORch_LOGS=recompiles` で再コンパイル監視
- 検証手順: 単GPUで数epochコンパイル安定性確認 → 4GPU DDPで loss一致 + steps/sec比較

参考: https://docs.pytorch.org/tutorials/recipes/regional_compilation.html /
https://github.com/pytorch/pytorch/issues/140229 /
https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/

### A-3. cuDNN SDPA backend（Blackwellでの正解。FA3/FA4は追わない）

- FlashAttention-3はHopper専用でsm_120非対応。FA4もGA未達。コミュニティwheelはtorch 2.10非対応
- PyTorch組込みの**cuDNN attention backend**がBlackwell最適化済み（cuDNN 9.x、dynamic shape対応）:
  ```python
  from torch.nn.attention import sdpa_kernel, SDPBackend
  with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION], set_priority=True):
      out = F.scaled_dot_product_attention(q, k, v)
  ```
- TTSの系列長は短め（音素~数百）なので全体寄与は中程度。cuDNN attentionは過去に正しさの
  エッジケースがあった経緯があるため、有効化後はloss曲線をFP32/現行と照合すること

参考: https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html /
https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html

### B-1. Frame-based dynamic batching（F5-TTS流）

- 「1バッチ=固定サンプル数」ではなく「1バッチ=合計フレーム数上限」でサンプル数を可変にする。
  F5-TTSは307,200フレーム/バッチで採用。CosyVoice・NeMoでも標準
- 既存の `BucketBatchSampler` / `DistributedBucketBatchSampler` の自然な拡張（長さの近い
  サンプルは既にまとまっているので、バケットごとにバッチサイズを変える形）
- 全長メル学習（out_size=null）でバッチサイズ32が最長発話に律速されている現状では、
  短い発話のバケットで2-3倍詰められる → 実効スループット10-30%向上の余地
- 注意: 有効バッチサイズが変動するためLRとの相互作用に理論上の揺れ。DDPでは各rankの
  フレーム数を揃える実装が必要

参考: https://arxiv.org/html/2410.06885v1 (F5-TTS) /
https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html

### C-1. NCCL / DDP通信（5090の特殊事情）

- **RTX 5090はGPU間P2Pがドライバレベルで無効**（GeForce系の仕様。NVIDIA公式回答あり）。
  NCCLはホストメモリ経由（SHM staging）で通信する
- 20Mパラメータの勾配はbf16で~40MB/step と小さく、既存のDDP設定
  （gradient_as_bucket_view / bucket_cap_mb=25 / static_graph）で概ね隠蔽可能
- 推奨: `nccl-tests` の `all_reduce_perf` で4枚のbusbwを一度実測。ハングやエラーが出る場合は
  `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` を明示（P2Pは元々不可なので副作用なし）

参考: https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776 /
https://github.com/NVIDIA/nccl/issues/1637

### C-2. 電力・熱の安定化

- RTX 5090はTDP 575W×4枚=2300W。ホストの電源・冷却次第でサーマルスロットリング→
  クロック変動→DDP同期待ちが発生しうる
- Blackwellはクロック上端の電力効率が悪く、`nvidia-smi -pl 450` 程度の制限で性能低下は数%、
  熱安定性は大幅に向上（4枚の同期学習では「最も遅い1枚」が全体を律速するため安定性が効く）

### 検証済み・現状維持でよいもの

- **validation頻度**: 既に`check_val_every_n_epoch=10` + 合成は10エポックごとに2サンプルのみ
  （`baselightningmodule.py`）= 既に軽量。追加削減の余地はほぼない
- **timestep sampling**: uniform維持が正解。Curriculum Sampling論文（2026）が
  「middle-biasedは漸近品質でuniformに劣る」を定量実証（logit-normal失敗の自社実績とも一致）
- **データローディング**: 全データRAM常駐 + num_workers=0 は「データがRAMに収まる」前提では
  正しい構成。GPU飢餓がprofilingで確認された場合のみ B-3 を検討
- **gradient checkpointing無効 + static_graph=true**（jvs_aligned）: メモリに余裕がある現状では
  再計算を省く正しい選択

### 非推奨の詳細理由

- **FP8（X-1）**: torchao実測で8Bモデル1.25x → モデルが小さいほど効果減。20Mでは量子化
  オーバーヘッドが勝ち、かつFP16ですらDP品質劣化した本プロジェクトでは品質リスクが過大
- **Muon等（X-4）**: LLMで収束10-15%改善の報告はあるが、本プロジェクトは
  optimizer/LR変更で品質退化を繰り返した記録があり（CLAUDE.md）、本番runでの変更は禁物。
  試すなら隔離した実験runのみ
- **sequence packing（X-5）**: Matchaのdecoderは1D U-Net convでサンプル境界を越えて
  リークするため構造的に不適合（block-diagonal attention maskで解決できるのはTransformer系のみ）
- **batch 64（X-8）**: VRAM的には可能で通信/計算比も改善するが、有効バッチ256は
  原論文実証レシピ（128, lr=1e-4固定）からの逸脱。品質ゲート付きで試すかは判断事項

---

## 4. 推奨実行順（適用する場合）

1. **A-1 プロファイリング**（30分）: 律速を確定。以降の投資判断の根拠
2. **C-1 + C-2**（10分）: NCCL疎通実測と電力制限。リスクゼロの安定化
3. **A-3 cuDNN SDPA**（半日）: 低リスク。loss照合付きで
4. **A-2 Regional compile**（1-2日）: 最大の攻め手。単GPU検証→DDP検証の段階導入
5. **B-1 frame-based batching**（1-2日）: sampler改修。効果はプロファイル結果次第
6. B-2 / B-3 / C-3 は上記の結果を見て

なお現行構成（bf16、~2.8 steps/s）でも **2500ep ≈ 24時間 / ~$45** であり、
「何もしない」のも合理的な選択肢である（実装・検証の人件費 vs $10-20の節約）。
