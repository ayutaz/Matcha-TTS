# ボコーダ改善 調査レポート（2026-07-07）

日本語Matcha-TTS（`jvs_aligned`、2500ep完走モデル）の音質頭打ちはボコーダ由来と診断済み
（`docs/jvs-aligned-eval-report.md`: 学習モデルUTMOS 3.00 ≈ 汎用HiFi-GAN天井2.90）。
アコースティックモデルは既にボコーダ天井到達で、**残る音質向上の余地はボコーダ側にある**。

本レポートは4エージェント並列調査（コードベース統合分析 / HiFi-GAN JVS fine-tune / BigVGAN / Vocos・最新系）の統合。
**このドキュメントを確認してから実装に進む**（現時点では調査のみ・学習未着手）。

---

## エグゼクティブサマリー

1. **Matchaのmel仕様にビット単位で一致する高品質ボコーダの公開checkpointが4つ存在**し、
   いずれも**再学習ゼロ（ゼロショット）で現行HiFi-GANと差し替え比較できる**。これが最大の発見。
2. **推奨は段階戦略**: まず①ゼロショットでVocos/WaveNeXt/BigVGANを差し替えA/B比較（コスト≈$0、数時間）→
   ②不足ならJVS fine-tune。いきなりfine-tuneは非効率。
3. HiFi-GAN JVS fine-tuneは実装コスト（学習ループ未実装）と「UTMOS逆転現象」リスクがあり、
   **ゼロショット差し替えより優先度は低い**。
4. **音質天井2.90の主因は「英語学習の汎用HiFi-GANがJVS未知」というドメイン差**であり、
   ドメイン適合ボコーダ（fine-tune or 別checkpoint）で3.3〜3.7域まで上がる余地がある。

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

## 4. 推奨戦略（段階的・低リスク順）

### Phase A: ゼロショット差し替えA/B比較（最優先、コスト≈$0、~半日）
mel完全一致の3系統を現行HiFi-GANと差し替えて比較。**学習不要**。
1. **melフィルタ数値照合**（唯一の技術リスク）: 同一wavで「Matcha `mel_spectrogram()`」と各ボコーダのmel特徴量を照合し、
   Slaney正規化・power=1・reflect+center=Falseが一致することを確認（ここが合えばゼロショット動作）
2. **GT mel再合成での天井測定**: 各ボコーダにGT melを通しUTMOS測定（現行HiFi-GAN天井2.90との比較）
3. **予測mel合成でのA/B**: `jvs_aligned` の予測mel → 各ボコーダ → UTMOS + 試聴
4. 候補: `BSC-LT/vocos-mel-22khz`（本命）/ `BSC-LT/wavenext-mel`（音質） / `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`（最高忠実度）
5. `cli.py` の `VOCODER_URLS`/`load_vocoder` にボコーダ選択肢を追加（推論専用なので低リスク）

### Phase B: JVS fine-tune（Phase Aで不足なら）
- ゼロショットで最良だったボコーダをJVSでfine-tune。VocosはBSC checkpointから数時間の継続学習で話者性を詰められる
- HiFi-GAN二段fine-tuneを選ぶ場合は学習ループ実装が前提。4GPUは「GT-mel/generated-mel/LR違い/BigVGAN比較」の並列スイープに使うのが費用対効果高い

### 判断ポイント（実装前に決めたいこと）
- **CPU推論を残すか**: 残すならVocos/WaveNeXt（BigVGANはGPU前提）
- **ライセンス**: 配布予定ならApache-2.0(Vocos/WaveNeXt) / MIT(BigVGAN)いずれも問題なし
- **ゼロショットで十分か、fine-tuneまで行くか**: Phase Aの結果を見て決定

---

## 5. 期待値と限界

- **期待**: ドメイン適合ボコーダで天井2.90→3.3〜3.7域、最終TTS UTMOSは予測mel品質に律速されつつ +0.1〜0.4程度
- **限界1（fmax=8000の構造的天井）**: 条件付けmelが8kHzでband-limitのため、8-11kHzは推定生成しかできず摩擦音/歯擦音の鮮明さに天井。
  根本解消はfmax=11025化だがアコースティックモデル再学習（mel統計再計算）を伴う破壊的変更 → 今回非推奨
- **限界2**: ボコーダはmel→波形のみ改善。ただし**jvs_alignedはMAS退化を既に解決済み（退化率0%）**なので、
  mel品質は良好でボコーダ差し替えがクリーンに効く条件は揃っている（BigVGAN調査の「退化melだと伸び限定」懸念は本モデルには非該当）

---

## 主要ソース
- Vocos: [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos), [arXiv 2306.00814](https://arxiv.org/html/2306.00814v3), [BSC-LT/vocos-mel-22khz](https://huggingface.co/BSC-LT/vocos-mel-22khz)
- WaveNeXt: [NICT ASRU2023 demo](https://ast-astrec.nict.go.jp/demo_samples/asru_2023_okamoto/), [BSC-LT/wavenext-mel](https://huggingface.co/BSC-LT/wavenext-mel)
- BigVGAN: [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN), [arXiv 2206.04658](https://arxiv.org/pdf/2206.04658), [nvidia/bigvgan_v2_22khz_80band_fmax8k_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_fmax8k_256x)
- HiFi-GAN fine-tune: [jik876/hifi-gan](https://github.com/jik876/hifi-gan), [arXiv 2010.05646](https://arxiv.org/pdf/2010.05646), [ESPnet2-TTS arXiv 2110.07840](https://arxiv.org/pdf/2110.07840), [kan-bayashi/ParallelWaveGAN JSUT hifigan.v1](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/jsut/voc1/conf/hifigan.v1.yaml)
- コードベース: `matcha/utils/audio.py`, `matcha/cli.py`, `matcha/models/matcha_tts.py`, `matcha/hifigan/`
