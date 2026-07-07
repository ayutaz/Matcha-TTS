# ボコーダ改善 調査レポート（2026-07-07）

日本語Matcha-TTS（`jvs_aligned`、2500ep完走モデル）の音質頭打ちはボコーダ由来と診断済み
（`docs/jvs-aligned-eval-report.md`: 学習モデルUTMOS 3.00 ≈ 汎用HiFi-GAN天井2.90）。
アコースティックモデルは既にボコーダ天井到達で、**残る音質向上の余地はボコーダ側にある**。

本レポートは4エージェント並列調査（コードベース統合分析 / HiFi-GAN JVS fine-tune / BigVGAN / Vocos・最新系）の統合。
**このドキュメントを確認してから実装に進む**（現時点では調査のみ・学習未着手）。

## デプロイ制約（2026-07-07 ユーザ確定）

以下の制約が候補選定を決定づける:
- **CPU推論が必須**
- **最終ターゲットはモバイル（モバイルCPU/GPU）**
- **配布を視野に入れる**（ライセンス重要）
- **最終的に必ずONNX化する。選定はONNX(特にCPU/モバイル)実行速度を一次基準とする**
  — PyTorch速度ではなくONNX化後の速度が判断軸。速度で不利なら音質優位でも本命から外す

### ★ONNX速度の結論（ユーザ仮説を一次情報で確認）
**ユーザ仮説「VocosはONNXにすると遅くなる」は成立**。iSTFT/STFTは **ONNXにiSTFT opが存在しない**（opset17でSTFTは追加されたが逆変換iSTFTは未実装、[ONNX #4777](https://github.com/onnx/onnx/issues/4777)）。VocosのiSTFTヘッドは(a) ONNX外で別実装（単一グラフ配布を放棄、[wetdog方式](https://huggingface.co/wetdog/vocos-mel-24khz-onnx)）か(b) convサブグラフ化（[mush42/istft-onnx](https://github.com/mush42/istft-onnx)、**torch比≥4倍遅**）が必要。
**WaveNeXtはiSTFTを線形層に置換 → Conv/GEMM/LayerNorm/Reshapeの標準opのみ** → ORT CPU・ONNX Runtime Mobile・TFLite・CoreML・ExecuTorchすべてでネイティブ最適化。
ConvNeXt backboneは両者共通で問題なし。差は**出力ヘッド(iSTFT vs linear)に局在**。

**決定的証拠**: BSC（バルセロナ）が同じMatcha-TTSのONNX多話者配布で **「Vocosと違い完全にONNX書き出しできるためWaveNeXtを選んだ」と明記**（[Interspeech 2024, Peiró-Lilja et al. §2.2](https://www.isca-archive.org/interspeech_2024/peirolilja24_interspeech.pdf)）。同論文のONNX-CPU実測（i7）: Matcha+WaveNeXt RTF **0.087** / Matcha+HiFi-GAN 0.089。**Vocosは表に載っていない（ONNX化できなかったため）**。

### 制約による判定の変化（ONNX速度確定後）
| 候補 | 判定 | 理由 |
|------|:---:|------|
| **WaveNeXt** (`BSC-LT/wavenext-mel`) | **デプロイ本命（確定）** | iSTFT無し=標準opのみ→ONNX/モバイル完全対応・HiFi-GAN同等以上に高速(RTF0.087)・軽量(13.68M)。日本語由来(NICT)・Apache-2.0・22kHz/80mel学習済み+ONNX export実績 |
| **HiFi-GAN**（現行/fine-tune） | **無難な第2候補** | transposed convのみ=ONNX完全ネイティブ・既に`export.py`で"wav"直接出力済み・sherpa対応。WaveNeXtにわずかに劣る程度。再学習回避ならこれ |
| **Vocos** (`BSC-LT/vocos-mel-22khz`) | **本命から除外** | PyTorchでは最速だがiSTFTがONNX/モバイルで遅い/非対応。同等品質ならiSTFT無しのWaveNeXtが合理的（BSC判断と一致） |
| **BigVGAN** | **リファレンスのみ（配布・デプロイ不可）** | 112M・GPU前提・CPU 0.21×RT。モバイル不可。品質上限の天井測定用途に限定 |
| **HiFi-GAN**（現行/fine-tune） | 現行baseline | transposed convでモバイル可だが音質が課題。fine-tuneは学習ループ未実装 |

**追加の決定軸（モバイル）**: iSTFT/STFT opのONNX/TFLite/CoreML/ExecuTorch書き出し可否、int8量子化、
sherpa-onnx等の既存モバイル推論経路。→ `## 6. モバイル/エッジ書き出し` に別途記載。

---

## エグゼクティブサマリー

1. **デプロイ制約（CPU必須・モバイル最終ターゲット・配布・ONNX化必須）を最優先すると、推奨はWaveNeXtで確定**。
   VocosはPyTorch最速だが**iSTFTがONNXに無く**、ONNX/モバイルで遅い/単一グラフ化不可 → 本命から除外。
   BSCが同じMatcha-TTSのONNX配布でVocosを捨てWaveNeXtを選んだ実例が決定的証拠（Interspeech 2024）。
2. **推奨は段階戦略**: ①WaveNeXt(`BSC-LT/wavenext-mel`)をゼロショットで現行HiFi-GANとA/B比較（コスト≈$0）→
   ②不足ならJVS fine-tune → ③既存 `matcha/onnx/export.py` の`MatchaWithVocoder`機構でWaveNeXt埋め込みONNX化。
3. **HiFi-GANは無難な第2候補**（ONNX完全ネイティブ・既に統合済み・再学習不要）。WaveNeXtにわずかに劣る程度。
4. **音質天井2.90の主因は「英語学習の汎用HiFi-GANがJVS未知」というドメイン差**であり、
   ドメイン適合ボコーダ（fine-tune）で3.3〜3.7域まで上がる余地がある。
5. **sherpa-onnxはupstream(shivammehta25)非対応**のため本プロジェクトのモデルをそのまま載せられない。
   自前ONNX（既存export機構）でモバイルORTに載せるのが第一経路。

---

## 1. 技術ゲート: 新ボコーダが満たすべきインターフェース（コードベース分析）

出典: `matcha/utils/audio.py:45-87`, `matcha/cli.py:26-29,90-96,123-128`, `matcha/models/matcha_tts.py:167`

### アコースティックモデルのmel仕様（厳密）
| パラメータ | 値 |
|------|------|
| n_fft / hop / win | 1024 / 256 / 1024 |
| n_mels / sr | 80 / 22050 |
| fmin / fmax | 0 / **8000** |
| center / pad | False / reflect `(1024-256)/2=384` 両端 |
| melフィルタ | librosa `librosa_mel_fn`（**Slaney正規化, htk=False**） |
| log変換 | `torch.log(torch.clamp(x, min=1e-5))`（自然対数、floor=1e-5） |

### ★最重要: ボコーダが受け取るのは「denormalize済みの自然対数log-mel」
- 学習/precompute時はz-score正規化 `(mel-mel_mean)/mel_std`（`mel_mean=-6.550095, mel_std=2.383771`）して保存
- しかし `synthesise()` は返り値で `denormalize()` = `decoder_out * mel_std + mel_mean` を計算（`matcha_tts.py:167`）
- `to_waveform()` はこの**denormalize済みmel**をボコーダに渡す（`cli.py:123-124`）
- **∴ ボコーダは値域≈`[-11.5, +2]`の生の自然対数log-melで動作する。z-score正規化melではない。mel_mean/stdはボコーダに渡らない**

### ボコーダIF contract
- 入力: `(B, 80, T)` denormalize済みlog-mel（上記mel paramと厳密一致必須）
- 出力: `[-1, 1]` の**22050Hz**波形、`waveform_len = T×256`
- `nn.Module.forward(mel)→wav`、CUDA/CPU対応、`torch.compile` 可、`(1,80,88)` zero-melでDenoiser初期化可
- **リポジトリはボコーダ推論専用**（`Generator`のみ使用）。学習部品（`MelDataset(fine_tuning=True)`・Discriminator・loss）は在るが**学習ループは未実装**

### 既存HiFi-GAN configとの整合性
`matcha/hifigan/config.py`（v1）はMatchaのmel仕様と**fmax=8000まで完全一致**（致命的不一致なし）。
唯一の注意点: 非fine-tuning MelDatasetはpeak正規化を掛けるがMatcha側は掛けない →
fine-tune時は `fine_tuning=True`（.npy mel直読み）パスを使い不一致を回避。

---

## 2. mel完全一致の公開ボコーダcheckpoint（★ゼロショット候補）

各checkpointの`config.json`を実確認。**mel式（Slaney/log(clamp)/reflect+center=False）まで一致**を確認済み。

| checkpoint | 方式 | sr | n_mels | fft/hop/win | fmax | License | 速度(HiFi-GAN比) |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **`BSC-LT/vocos-mel-22khz`** | Vocos | 22050 | 80 | 1024/256/1024 | **8000** | **Apache-2.0** | **~13倍速・CPU可** |
| **`BSC-LT/wavenext-mel`** | WaveNeXt | 22050 | 80 | 1024/256/1024 | **8000** | **Apache-2.0** | Vocos同等・CPU可 |
| **`nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`** | BigVGAN v2 | 22050 | 80 | 1024/256/1024 | **8000** | **MIT** | **~0.15倍（重い・GPU前提）** |
| `nvidia/bigvgan_22khz_80band`(v1) | BigVGAN v1 | 22050 | 80 | 1024/256/1024 | **8000** | MIT | 遅い |

**不一致で使えない代表例（参考）**: `charactr/vocos-mel-24khz`(24k/100mel)、`bigvgan_v2_22khz_80band_256x`(**fmax=null=11025**)、
kan-bayashi JVS/JSUT HiFi-GAN(24k/hop300/fmax7600) — いずれもmel不一致でそのままでは不可。

**日本語特化checkpointの状況**: 「日本語学習済み かつ Matchaのmelに合う」独立ボコーダは公開されていない。
ただしボコーダは概ね言語非依存（mel→波形変換）なので、英語/カタルーニャ語学習のcheckpointでも日本語で機能する見込み。

---

## 3. 各方式の詳細

### 3-A. Vocos（第一推奨）
- **アーキ**: ConvNeXt backbone + iSTFTヘッド（transposed conv無し・等時間解像度）→ CPUでも高速
- **品質**: periodicity error最小（HiFi-GANのartifactを構造的に低減）。UTMOS 3.734（BigVGAN 3.749とほぼ互角）、主観MOSはBigVGANと有意差なし
- **速度**: HiFi-GAN比~13倍、BigVGAN比~70倍。**Matchaの高速CFMの利点を殺さない**。ONNX版あり
- **`BSC-LT/vocos-mel-22khz`はMatcha-TTS統合を公式想定**（Matxa-TTSカタルーニャ語プロジェクトの実運用ボコーダ）
- Apache-2.0（商用・改変可）

### 3-B. WaveNeXt（音質最優先の横並び候補）
- VocosのiSTFTヘッドを学習可能な線形層に置換。**Vocosと同速で音質は上**と報告
- **日本語（JSUT）由来の研究**（NICT岡本・戸田ら, ASRU 2023）で日本語との相性実証あり
- `BSC-LT/wavenext-mel`（Apache-2.0）がmel完全一致 → Vocosの上位互換候補としてA/B比較の価値大

### 3-C. BigVGAN v2（最高忠実度・速度妥協時）
- **アーキ**: Snake activation（周期的帰納バイアス→調波構造に強い）+ anti-aliasing(low-pass) + MRD/CQT判別器
- **品質**: 同一パラメータでHiFi-GAN V1を全客観指標で上回る。日本語ピッチアクセントにsnakeが効きやすい
- **欠点**: 112M（HiFi-GANの8倍）、**CPU不利（~0.21×RT）でGPU前提**、v2の最速化は専用CUDAカーネル要（torch.compile非併用が無難）
- **ライセンスMIT**（「制限的ライセンス」は他のNVIDIA資産との混同と判明。v1/v2ともMITで商用可）

### 3-D. HiFi-GAN JVS fine-tune（優先度低）
- **レシピ**: `universal_v1`から継続。**generator `g_02500000` + discriminator `do_02500000` の両方が必須**
  （Matchaはgeneratorのみ同梱 → `do_`を別途入手。discriminatorスクラッチは金属artifactの原因）
- **★UTMOS逆転現象（重要な発見）**: 予測mel 3.00 > GT mel天井 2.90 という逆転は、JVS GT収録のノイズ/呼気をUTMOSが嫌い、
  over-smoothな予測melがクリーンに聞こえるため。**天井2.90は「英語学習universal HiFi-GANのドメイン差」由来**で、
  JVS domain fine-tuneでGT-mel天井は3.3〜3.7域に上がる余地。ただし純GT-mel fine-tuneは最終UTMOSを上げない/下げるリスク
- **generated(teacher-forced) mel fine-tuneが最終TTS品質に効く**（exposure bias解消）。Matcha(CFM)では
  「Julius GT durationで固定→ODE decodeしGT波形とフレーム整合する予測mel」が等価物。**jvs_alignedのJulius durationで追加コストほぼゼロで実現可能**
- **推奨二段構成**: ①GT-melでdomain適応(100-200k step) → ②teacher-forced aligned melで exposure bias解消(50-100k step)。単一5090で1日以内
- **欠点**: 学習ループ実装が必要（upstream jik876移植 or 自前実装）。天井2.9を大きく超えたいならBigVGAN学習の方が上限が高い

---

## 4. 推奨戦略（ONNX速度最優先・段階的）

**本命 = WaveNeXt**（ONNX/モバイル速度・配布で確定）。**第2候補 = 現行HiFi-GAN継続**（再学習不要のフォールバック）。
Vocos/BigVGANは「PyTorch上の品質天井の参照」としてのみA/Bに含める（デプロイ候補ではない）。

### Phase A: ゼロショット差し替えA/B比較（最優先、コスト≈$0、~半日）
1. **melフィルタ数値照合**（唯一の技術リスク）: 同一wavで「Matcha `mel_spectrogram()`」と各ボコーダのmel特徴量を照合し、
   Slaney正規化・power=1・reflect+center=Falseが一致することを確認（ここが合えばゼロショット動作）
2. **GT mel再合成での天井測定**: 各ボコーダにGT melを通しUTMOS測定（現行HiFi-GAN天井2.90との比較）
3. **予測mel合成でのA/B**: `jvs_aligned` の予測mel → 各ボコーダ → UTMOS + 試聴
4. 比較対象: **`BSC-LT/wavenext-mel`（本命）** / 現行`hifigan_univ_v1`（baseline） / 参照として`BSC-LT/vocos-mel-22khz`・`nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`（品質天井の目安）
5. `cli.py` の `VOCODER_URLS`/`load_vocoder` にWaveNeXt選択肢を追加（推論専用なので低リスク）

**判定基準**: WaveNeXtゼロショットが現行HiFi-GANを品質で上回れば → Phase Cへ（ONNX化して確定）。
不足なら → Phase B（JVS fine-tune）。VocosがWaveNeXtより明確に良くても**ONNX速度で不利なため採らない**。

### Phase B: WaveNeXt JVS fine-tune（Phase Aで不足なら）
- `BSC-LT/wavenext-mel`（Catalan/多言語学習）から**JVS mel（22050/80/hop256/fmax8000）で継続fine-tune**し話者性を詰める
- [wetdog/wavenext_pytorch](https://github.com/wetdog/wavenext_pytorch)（22kHz/80mel設定可・ONNX export同梱）を学習基盤に。単一5090で数時間〜1日
- exposure bias解消のためgenerated(teacher-forced) melも使える（jvs_alignedのJulius durationでODE decode → GT波形整合mel。追加コストほぼゼロ）
- 4GPUは「GT-mel / generated-mel / LR違い」の並列スイープに使うのが費用対効果高い（GAN vocoderのDDPは収穫逓減）

### Phase C: ONNX化・配布（本命WaveNeXt確定後）
- 既存 `matcha/onnx/export.py` の **`MatchaWithVocoder`**（vocoder埋め込み・出力名`"wav"`で単一グラフ化）を **WaveNeXt対応に拡張**
- `matcha/cli.py` の `load_vocoder`（現状HiFi-GANのみ）にWaveNeXtローダを追加
- WaveNeXtはiSTFT無し=標準opのみなので、既存のend-to-end ONNX書き出し・`matcha/onnx/infer.py`（`"wav"`判定）がそのまま流用可能
- ONNX-CPU RTF計測でHiFi-GAN(0.089)と比較。int8はサイズ削減用途（速度/品質はA/B、conv int8はORTで遅化例あり要注意）
- sherpa-onnx配布を狙うなら別途icefall形式ONNXへの準拠が必要（upstream非対応のため）→ 後追い

### 判断ポイント（確定済み）
- ~~CPU推論を残すか~~ → **必須確定**。iSTFT無しのWaveNeXt/HiFi-GANのみが候補（Vocos/BigVGAN除外）
- **ライセンス**: WaveNeXt=Apache-2.0、HiFi-GAN=MIT系。配布問題なし
- ゼロショットで十分か → Phase Aの結果で決定

---

## 4-bis. Phase A 実測結果（2026-07-07、ローカルCPU）

`scripts/eval_vocoder_ab.py` で jvs_aligned 予測mel（5話者×10文=50サンプル, n_timesteps=32）を
WaveNeXtゼロショット vs 現行HiFi-GAN で paired UTMOS 比較:

| ボコーダ | UTMOS | 破綻(<1.5) |
|------|:---:|:---:|
| WaveNeXt（BSC-LT/wavenext-mel, ゼロショット） | **2.925 ± 0.349** | 0 |
| HiFi-GAN univ（現行baseline） | 2.983 ± 0.413 | 0 |
| paired delta (WaveNeXt − HiFi-GAN) | **−0.058（SE 0.047, t=−1.24, 有意差なし, win率44%）** | — |

**結論**:
1. **mel互換性は実証** — 破綻サンプル0。config一致（Slaney/Slaney/log(clamp)）が実音声で裏付けられた（非互換ならUTMOS~1の破綻音になる）
2. **ゼロショットは品質同等**（統計的に区別不能）。両者ともJVS未適合のため予想通り。音質の上積みは無し
3. HiFi-GAN 2.983 は出荷モデル評価の3.00（1000サンプル）と一致 → harness検証OK
4. **判定**: ゼロショットではHiFi-GANを上回らない（同等）→ 明確な品質向上には **Phase B（WaveNeXt JVS fine-tune）** が必要。
   ただしWaveNeXtは品質同等かつONNX/モバイル対応（iSTFT無し）なので、デプロイ目標には既に十分（parity + 展開性）

## 4-ter. BigVGAN診断プローブ（2026-07-07、濁りの原因切り分け）

ユーザ試聴で「WaveNeXt/HiFi-GAN両方が同等に濁る」→ 濁りの原因（ボコーダ品質 vs fmax構造天井）を
切り分けるため、`scripts/eval_bigvgan_probe.py` で同一予測mel（2話者×10文=20）を最強のBigVGAN v2
（fmax8k・mel互換・anti-aliasing・112M）でも合成して比較:

| ボコーダ | UTMOS | vs HiFi-GAN |
|------|:---:|:---:|
| BigVGAN v2 fmax8k（最強・参照のみ） | 3.019 ± 0.525 | +0.023（win 50%） |
| WaveNeXt（ゼロショット） | 3.070 ± 0.423 | +0.074（win 70%） |
| HiFi-GAN univ（現行） | 2.996 ± 0.488 | — |

**結論（濁りはfmax構造天井、ボコーダでは直らない）**:
- **3つとも~3.0で統計的に横並び**。高域強化に特化した最強のBigVGANですら現行HiFi-GANと同等（+0.023）
- → 濁りの原因は**ボコーダの品質/アーキではなくfmax=8000の帯域制限**（melに8kHz以上の情報が無く、どのボコーダも同じ帯域制限音しか出せない）。前回「予測mel(3.00)≈GT mel天井(2.90)」＝アコースティックモデルは天井到達、とも整合
- **Phase B（WaveNeXt fine-tune）は濁り改善には投資価値が低い**（最強BigVGANでも改善しない以上、fine-tuneでも減らない公算大）。この診断でGPU出費を回避
- **濁りの本質的改善は fmax=8000→11025 引き上げ = アコースティックモデル再学習**（mel統計再計算・全前処理やり直し・2500ep再学習）が必要な破壊的変更のみ
- 副産物: WaveNeXtは品質同等以上＋ONNX/モバイル最速 → **デプロイ用途にはWaveNeXtが最適**（Phase C）

**注**: GT mel天井の直接測定（本物のJVS録音mel→各ボコーダ）はローカルにJVS wavが無く未実施。
fmax引き上げの前に最終確認したい場合はインスタンス（JVS .pt保有）で実施可能。ただしBigVGANプローブ＋前回データで
「fmax構造天井」の結論は十分堅い。

## 5. 期待値と限界

- **期待**: ドメイン適合WaveNeXtで天井2.90→3.3〜3.7域、最終TTS UTMOSは予測mel品質に律速されつつ +0.1〜0.4程度。ONNX-CPU RTF ~0.087（HiFi-GAN同等）
- **限界1（fmax=8000の構造的天井）**: 条件付けmelが8kHzでband-limitのため、8-11kHzは推定生成しかできず摩擦音/歯擦音の鮮明さに天井。
  根本解消はfmax=11025化だがアコースティックモデル再学習（mel統計再計算）を伴う破壊的変更 → 今回非推奨
- **限界2**: ボコーダはmel→波形のみ改善。ただし**jvs_alignedはMAS退化を既に解決済み（退化率0%）**なので、
  mel品質は良好でボコーダ差し替えがクリーンに効く条件は揃っている（BigVGAN調査の「退化melだと伸び限定」懸念は本モデルには非該当）

---

## 6. モバイル/エッジ・ONNX書き出し（デプロイ制約の核心）

### iSTFTがONNXの鬼門（Vocos除外の根拠）
- `torch.stft`/`torch.istft` は長年ONNXエクスポート非対応（[pytorch/audio #982](https://github.com/pytorch/audio/issues/982), [pytorch #65666](https://github.com/pytorch/pytorch/issues/65666)）
- ONNXはopset17で**STFT opを追加したが逆変換iSTFT opは今も無い**（[ONNX #4777](https://github.com/onnx/onnx/issues/4777)）
- Vocos公式のONNX要望は未解決（[gemelo-ai/vocos #38](https://github.com/gemelo-ai/vocos/issues/38)）。公開ONNX実装[wetdog/vocos-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx)は**iSTFTをONNX外で実行**（単一グラフ配布を放棄）
- iSTFTをconvサブグラフ化する[mush42/istft-onnx](https://github.com/mush42/istft-onnx)は**torch比≥4倍遅**
- ONNX Runtime Mobileの削減ビルドはSTFT/DFT contrib opを含む保証がない。TFLite/CoreML/ExecuTorchもiSTFTネイティブ無し
- **ConvNeXt backbone自体は標準op（Conv1d/LayerNorm/Linear）で問題なし**。低速化はiSTFTヘッドに局在 → WaveNeXtは線形ヘッドでこれを回避

### ONNX-CPU実測（BSC Interspeech 2024, i7 12th Gen、同一Matcha音響モデル）
| 構成 | サイズ | RTF (GPU) | RTF (CPU) |
|------|:---:|:---:|:---:|
| Matcha + HiFi-GAN | 123 MB | 0.013 | 0.089 |
| Matcha + WaveNeXt | 122 MB | 0.010 | **0.087** |
| Matcha + Vocos | — | — | **表に無し（ONNX化できず）** |

WaveNeXtはHiFi-GANより僅かに高速・軽量。**BSCがVocosを表に載せていない事実がVocos除外の直接証拠**。

### sherpa-onnx（配布ランタイム基盤）の位置づけ
- Android/iOS/RPi/WASM/HarmonyOSのonnxruntimeビルドを持つ優秀な基盤。Matcha-TTS対応済み（英語ljspeech・中国語baker、ボコーダはvocos-22khz-univ.onnx 51MB or hifigan）
- **重大な制約**: sherpa-onnxは **shivammehta25 upstream由来のMatchaを非対応**と明記（icefallレシピのI/Oシグネチャ準拠モデルのみ）。本プロジェクトはupstream由来 → そのままでは載らない
- 日本語Matchaはsherpa公式に無し（[Issue #3028](https://github.com/k2-fsa/sherpa-onnx/issues/3028)未解決）
- → **第一経路は自前ONNX**（既存`matcha/onnx/export.py`でモバイルORTに載せる）。sherpa配布はicefall形式準拠が必要で後追い

### int8量子化
- ONNX dynamic int8はCPUで最大~3倍・サイズ~1/4だが、**convが主のボコーダはORTでint8がむしろ遅化する例あり**（[onnxruntime #12854](https://github.com/microsoft/onnxruntime/issues/12854)）
- 波形生成は量子化ノイズに敏感 → **int8はサイズ削減手段と位置づけ**、速度/品質はA/B確認。WaveNeXtの線形ヘッドはiSTFTより量子化が素直

---

## 主要ソース
- Vocos: [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos), [arXiv 2306.00814](https://arxiv.org/html/2306.00814v3), [BSC-LT/vocos-mel-22khz](https://huggingface.co/BSC-LT/vocos-mel-22khz)
- WaveNeXt: [NICT ASRU2023 demo](https://ast-astrec.nict.go.jp/demo_samples/asru_2023_okamoto/), [BSC-LT/wavenext-mel](https://huggingface.co/BSC-LT/wavenext-mel)
- BigVGAN: [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN), [arXiv 2206.04658](https://arxiv.org/pdf/2206.04658), [nvidia/bigvgan_v2_22khz_80band_fmax8k_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_fmax8k_256x)
- HiFi-GAN fine-tune: [jik876/hifi-gan](https://github.com/jik876/hifi-gan), [arXiv 2010.05646](https://arxiv.org/pdf/2010.05646), [ESPnet2-TTS arXiv 2110.07840](https://arxiv.org/pdf/2110.07840), [kan-bayashi/ParallelWaveGAN JSUT hifigan.v1](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/jsut/voc1/conf/hifigan.v1.yaml)
- ONNX/モバイル: [BSC Interspeech 2024（WaveNeXt採用理由・RTF表）](https://www.isca-archive.org/interspeech_2024/peirolilja24_interspeech.pdf), [WaveNeXt ASRU2023](https://ieeexplore.ieee.org/document/10389765/), [wetdog/wavenext_pytorch(ONNX export)](https://github.com/wetdog/wavenext_pytorch), [ONNX iSTFT未実装 #4777](https://github.com/onnx/onnx/issues/4777), [vocos ONNX要望 #38](https://github.com/gemelo-ai/vocos/issues/38), [mush42/istft-onnx(4倍遅)](https://github.com/mush42/istft-onnx), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- コードベース: `matcha/utils/audio.py`, `matcha/cli.py`, `matcha/models/matcha_tts.py`, `matcha/hifigan/`, `matcha/onnx/export.py`（`MatchaWithVocoder`）, `matcha/onnx/infer.py`
