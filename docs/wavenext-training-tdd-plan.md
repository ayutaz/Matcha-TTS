All load-bearing facts are confirmed against the live repo. Key confirmations: `mel_spectrogram` expects 2D `(B,T)` (internally `unsqueeze(1)`→pad 384→`squeeze(1)`), `spectral_normalize_torch`=`log(clamp(x,1e-5))`, cache key includes `fmax`; `WaveNextHead.forward` returns **2D `(B, T*hop)`** (so no `.squeeze(1)` needed in training); `load_wavenext` drops `feature_extractor.*` and asserts **both** `not missing` and `not unexpected`; `tests/__init__.py` exists (0-byte) and no `conftest.py`; ruff `exclude` currently lists `matcha/hifigan`/`matcha/wavenext` and `UP006`/`UP045` are **not** ignored; `soundfile` is **not** a declared dependency; `einops` is L46; `packages.find` excludes only tests/examples. No `wavenext_train/` exists yet (clean slate).

Below is the integrated TDD implementation plan.

---

# WaveNeXt 学習コード (fmax=11025) TDD 実装計画

本計画は6実装面の TDD spec と敵対的検証(verifier)結果を統合し、**確定した単一の実装指示**にしたものである。verifier が指摘した BLOCKER / 要修正は全て本文に反映済み。実装は本計画のみを参照すればよい(元 spec の未修正箇所は上書きされる)。

方針: **TDD厳守(失敗テスト先行)・wetdog逐語移植・本体依存のみ・matcha側は完全非破壊**。

---

## 0. クロスサーフェス確定判断(最重要 — 面間の矛盾解決)

元 spec 間に4つの不整合があり、実装前に確定させる。

### 判断1: MelSpecReconstructionLoss は matcha mel (fmax=11025, 80-bin) を使う ★決定

`loss`面と`train-config`面で recon loss の実装が矛盾していた。

| 面 | recon loss 実装 |
|---|---|
| `loss` / `experiment` | `MatchaMelFeatures`(matcha mel, fmax=11025, 80-bin, `safe_log`撤去) |
| `train-config` | wetdog逐語 `torchaudio.transforms.MelSpectrogram`(128-bin, center=True, htk無しslaney) |

**確定: matcha mel 版を採用**(`torchaudio.transforms.MelSpectrogram` は不採用)。根拠:
- プロジェクト authoritative 決定「torchaudio MelSpectrogram混入は却下」に従う。
- fmax=11025 再学習の目的自体が「学習mel領域 == 推論mel領域(`synthesise`→`to_waveform` の非正規化log-mel)」の一致であり、recon loss も同一領域にすべき。
- **副次的利点**: `MatchaMelFeatures` は無パラメータ → recon loss が state_dict にキーを一切出さない。`torchaudio.transforms.MelSpectrogram` は `mel_fb`/`window` バッファを登録し `melspec_loss.mel_spec.*` キーを生む。matcha mel 採用で抽出(extract)が backbone./head. のみで自然に閉じる。
- トレードオフ(承知の上): wetdog の 128-bin 知覚L1 より mel 分解能は粗い(80-bin)。領域一致を優先して受容する。

→ `wavenext_train/loss.py::MelSpecReconstructionLoss` は逐語対象外。`Disc/Gen/FeatureMatching` の3クラスのみ wetdog 逐語。

### 判断2: feature モジュール名は `wavenext_train/features.py` に統一

元 spec は `features.py` / `feature_extractor.py` / `feature_extractors.py` の3表記が混在。**`wavenext_train/features.py`** に統一(SOUND判定の features面が正)。全モジュールは `from wavenext_train.features import MatchaMelFeatures` で参照し、**再定義禁止**(単一真実源)。

### 判断3: 実験クラスは `WaveNeXtExp`、判別器・損失は内部構築(wetdog VocosExp 準拠)

`experiment`面(`WaveNextGANExp`, 全注入) と `train-config`面(`WaveNeXtExp`, 内部構築)が矛盾。**`WaveNeXtExp` で MPD/MRD/各loss を `__init__` 内部で構築**する wetdog VocosExp 準拠設計に統一。生成器(`feature_extractor`/`backbone`/`head`/`melspec_loss`)のみ注入。テストは in-file fake を使わず、実 MPD/MRD をtiny入力(B=2, num_samples=4096)で回す(CPU数秒で通ることを discriminators面が実測済み)。

### 判断4: max_steps は「バッチ単位」、scheduler の `//2` は廃止

manual-opt で scheduler を **1バッチ1回** 手動 `.step()` するため、wetdog の `trainer.max_steps//2` は不要。`num_training_steps = hparams.max_steps`(バッチ数)を直接使い、停止も `n_batches >= max_steps` で行う(`trainer.max_steps=-1`)。→ experiment面 TEST2 の `max_steps//2=500` 前提は破棄し、`num_training_steps` 直接指定に書き換える。

---

## 1. TDD 実装順序(依存グラフ)

各ステップは **失敗テスト作成(RED) → 実装(GREEN) → `uv run pytest <file>` 緑確認** を厳守。RED原因は常に `ModuleNotFoundError`/`ImportError`(モジュール未作成)。

```
Step 0  pyproject.toml 前提整備(下記 §11)+ wavenext_train/__init__.py + uv sync
          └─ import 解決の土台。§11 の ruff exclude / soundfile 追加はここで実施
Step 1  features        (依存: matcha.utils.audio のみ)          ← 最初。他面の土台
Step 2a dataset         (依存: soundfile, torchaudio.functional, lightning)  ┐
Step 2b discriminators  (依存: einops, torchaudio.transforms, weight_norm)   ├ 並列可(相互独立)
Step 2c loss            (依存: features[Step1], torch)                        ┘
Step 3  experiment      (依存: features, loss, discriminators, matcha.wavenext.models)
Step 4  train/config/scripts (依存: 全上流 + matcha.wavenext.models, matcha.cli.load_wavenext)
```

理由: `loss.MelSpecReconstructionLoss` が `features.MatchaMelFeatures` に依存するため features を最優先。dataset/discriminators は features 非依存で並列可。experiment は全モジュールを結線。train/config/scripts は最終統合(extract round-trip 検証含む)。

---

## 2. 確定テストスイート(verifier修正反映済み・先行作成)

全テスト CPU 完結・ネットワーク不要・random init。合成wavは `soundfile`(`sf.write`)で生成。

### 2.1 `tests/test_matcha_mel_features.py`(features, 6件, 全CPU)

| test | 確定 assert |
|---|---|
| `test_matches_matcha_mel_spectrogram_exactly` | `torch.manual_seed(0); a=torch.rand(2,16384)*2-1`。`torch.equal(MatchaMelFeatures(fmax=11025)(a), mel_spectrogram(a,1024,80,22050,256,1024,0.0,11025,center=False))`(**atol=0** コア領域一致) |
| `test_output_shape_B_T_to_B_80_Tframes` | `MatchaMelFeatures(fmax=11025)(torch.zeros(3,256*40)).shape == (3,80,40)` |
| `test_stft_is_fp32_under_autocast_and_bf16_input` | `with torch.autocast('cuda',enabled=False): out=feat(a)` → `out.dtype==torch.float32`。`feat(a.to(torch.bfloat16)).dtype==float32` かつ `isfinite().all()` |
| `test_frames_times_hop_equals_input_length` | `T=256*50; out=feat(torch.zeros(1,T)); out.shape[-1]*256==T`(center=False 保証) |
| `test_fmax_is_honored` | `not torch.allclose(feat_11025(a), MatchaMelFeatures(fmax=8000)(a))` |
| `test_no_zscore_normalization` | `not hasattr(feat,'mel_mean') and not hasattr(feat,'mel_std')`; `feat(torch.rand(1,16384)*2-1).min().item() < -1.0` |

### 2.2 `tests/test_wavenext_dataset.py`(dataset, 6件, 全CPU)★verifier修正3点反映

| test | 確定 assert |
|---|---|
| `test_getitem_returns_num_samples_1d_waveform` | 22050Hz長30000のmono wav。`ds[0].shape==(16384,)`, `ndim==1`, `dtype==float32`, `isfinite().all()` |
| `test_short_audio_is_repeat_padded_to_num_samples` | 22050Hz長5000のwav。`ds[0].shape==(16384,)`; **出力間ブロックのみ検証** `allclose(out[5000:10000], out[0:5000])` かつ `allclose(out[10000:15000], out[0:5000])`。**生wavとの比較は禁止**(train gainが乱数、seed非固定) |
| `test_stereo_is_downmixed_to_mono` | (30000,2)ステレオwav。`train=False` の `ds[0].shape==(16384,)`, `ndim==1`, `isfinite().all()` のみ。**「peak値がmeanを証明」の主張は削除**(peak正規化が絶対振幅を消すため証明不能) |
| `test_output_value_range_within_unit_interval` | 22050Hz(resample無)ピーク1.0含むwav。train/val とも `out.abs().max() <= 1.0` |
| `test_peak_normalization_matches_sox_norm_semantics` | val(gain=-3固定,resample無)グローバルピーク=1.0を先頭16384内に配置。`ds[0].abs().max() ≈ 10**(-3/20)=0.70795`(atol=1e-3)。**+回帰ガード: `'sox_effects' not in inspect.getsource(wavenext_train.dataset)`** |
| `test_dataloader_produces_batched_waveforms` | filelist 5行、`VocosDataModule.train_dataloader()` batch_size=3,num_workers=0。`batch.shape==(3,16384)`, `dtype==float32` |

### 2.3 `tests/test_discriminators.py`(discriminators, 8件, 全CPU)

| test | 確定 assert |
|---|---|
| `test_mpd_forward_returns_four_lists_per_period` | `mpd(y(2,16384),y_hat(2,16384))` 4値。各 list len==5、`y_d_r` は2D `shape[0]==2`、全 `isfinite` |
| `test_mpd_fmap_lengths_and_shape` | `fmap_rs` 各inner len==5、各要素4D `shape[0]==2` |
| `test_mrd_forward_returns_four_lists_per_resolution` | 各 list len==3、logits は **4D**(DiscriminatorR は flatten しない)、`isfinite` |
| `test_mrd_fmap_lengths` | `fmap_rs` 各inner len==21(5band×4 + conv_post) |
| `test_default_periods_and_fft_sizes` | `[d.period for d in mpd.discriminators]==[2,3,5,7,11]`; `[d.window_length ...]==[2048,1024,512]` |
| `test_discriminatorp_reflect_pad_on_indivisible_length` | `y=torch.randn(2,12345)`; `mpd(y,y)` 例外なし・全 finite(reflect-pad分岐) |
| `test_gan_backward_flows_to_discriminator_params` | MPD+MRD 全 logits+fmap 平均和 → `.backward()`。`y_hat.grad` finite かつ MPD param `.grad is not None` |
| `test_unconditional_num_embeddings_none_path` | 既定 `num_embeddings=None` で `bandwidth_id` 無し forward 成功、`hasattr(d,'emb')==False` |

### 2.4 `tests/test_wavenext_train_loss.py`(loss, 6件, 全CPU)★verifier修正[A][B]反映

| test | 確定 assert |
|---|---|
| `test_discriminator_loss_hinge_scalar_and_perdisc_lists` | 3-tuple。`loss.numel()==1`,finite,`len(r)==len(g)==2`。手計算 `Σ[mean(relu(1-dr))+mean(relu(1+dg))]` と `allclose(atol=1e-6)`。**+決定性: `big=full((1,1,4),10.); nb=full((1,1,4),-10.); DiscriminatorLoss()([big],[nb])[0].squeeze()≈0`** |
| `test_generator_loss_is_hinge_relu_not_neg_mean` | `loss==Σmean(relu(1-dg))`(atol=1e-6)。**回帰ガード: `not allclose(loss, Σ(-dg.mean()))`** |
| `test_feature_matching_loss_l1_over_ragged_maps` | ragged `List[List[Tensor]]`。手計算一致。恒等入力で `loss.item()==0.0` |
| `test_mel_recon_uses_matcha_features_fmax11025_no_double_log` | `isinstance(mloss.mel_features, MatchaMelFeatures)`。恒等で `loss==0`。**atol=0**: `allclose(mloss(yh,y), F.l1_loss(mel_spectrogram(y,...,11025,center=False), mel_spectrogram(yh,...)))`。**異なる2ペア両方 `isfinite().all()`(二重log退行検知)**。`(B,1,T)` 入力受理 |
| `test_all_losses_finite_and_backward_cpu` | **Gen/Mel に加え Disc/FM も forward+backward**: `DiscriminatorLoss()(dr,dg)` の `dloss.backward()` → `dr[0].grad`/`dg[0].grad` finite; `FeatureMatchingLoss()(a,b).backward()` → `a[0][0].grad` finite |
| `test_generator_and_mel_backward_cpu` | (元 all_losses から分離) `GeneratorLoss([leaf]).backward()`; `MelSpecReconstructionLoss(fmax=11025)(yh,y).backward()` → grad finite |

### 2.5 `tests/test_wavenext_experiment.py`(experiment, 7件, 全CPU)★verifier修正 T1/T3/EMA反映

tiny構成: `VocosBackbone(input_channels=80,dim=16,intermediate_dim=32,num_layers=2)` + `WaveNextHead(dim=16,n_fft=1024,hop_length=256)`、`num_samples=T*256`。MPD/MRD は実物(内部構築)。

| test | 確定 assert |
|---|---|
| `test_configure_optimizers_returns_two_opts_two_scheds` | `len(opts)==2, len(scheds)==2`。opts は AdamW、**`opt.defaults['lr']==initial_learning_rate`**(★`param_groups['lr']` は warmup中 0.0 なので不可)、`param_groups[i]['betas']==(0.8,0.9)`。opt0 param-set==mpd∪mrd、opt1==backbone∪head(feature_extractor 無パラメータ)。scheds は `LambdaLR` |
| `test_cosine_warmup_lambda_matches_formula` | `get_cosine_schedule_with_warmup(opt, num_warmup_steps=50, num_training_steps=500)`(★`//2`無し)。`fn(0)==0.0, fn(25)≈0.5, fn(50)≈1.0, fn(500)≈0.0, fn(275)≈0.5`。`transformers` を import しない |
| `test_n_batches_buffer_increments_one_per_batch` | `n_batches` は long buffer 初期0。**loader は ≥3バッチを供給**(`TensorDataset(randn(6,T*256))`, batch_size=2, `limit_train_batches=3`)。fit後 `int(n_batches.item())==3`、`trainer.global_step`(≈6)と乖離 |
| `test_pretrain_gate_toggles_train_discriminator` | `pretrain_mel_steps=2`。`n_batches.fill_(1); on_train_batch_start()` → `train_discriminator is False`。`fill_(2)` → `True`(境界 inclusive `>=`) |
| `test_detach_blocks_generator_gradient_in_disc_step` | `d_in=audio_hat.detach()`; `d_in.requires_grad is False`; D-step backward後、`backbone`/`head` 全 param `.grad is None`、MPD param `.grad is not None` |
| `test_training_step_runs_full_gan_and_steps_both_optimizers` | `pretrain_mel_steps=0`, `Trainer(max_steps=2,...)`.fit 完走。spy で戻り dict の `loss_g`/`loss_d` finite、両 optimizer LR がscheduler後変化 |
| `test_pretrain_phase_skips_disc_and_adversarial_terms` | `pretrain_mel_steps=100`(未到達)。spy dict の `loss_d is None`、`loss_g == mel_loss_coeff*mel_loss`。fit前後で mpd/mrd param byte-identical、backbone/head は変化 |

補足: 戻り dict のspyはサブクラス/monkeypatchで `training_step` の `out` をlistに追記(manual-opt では Lightning が戻り値を無視するため)。`self.ema_decay` 未定義バグは **§3 experiment で `__init__` に `self.ema_decay = ema_decay` を追加**して回避済み。

### 2.6 `tests/test_wavenext_train.py`(config/scripts/統合, 7件, 全CPU)

module面と重複する cosine/n_batches の単体検証は 2.5 に集約し、本ファイルは config+scripts+E2E に限定。

| test | 確定 assert |
|---|---|
| `test_init_from_bsc_keeps_only_backbone_head` | 混在dictを save。`build_init_state_dict()` 返りは全キー `('backbone.','head.')` 始まり、`feature_extractor.*` 0件、キー数一致 |
| `test_extract_generator_roundtrips_load_wavenext` | 実 `WaveNeXtVocoder().state_dict()` の backbone./head. + ダミー `multiperioddisc.*`/`multiresddisc.*`/`n_batches` を混ぜた `{'state_dict':...}` を .ckpt保存。`extract(ckpt)` → save → `matcha.cli.load_wavenext(out,'cpu')` 例外なし。`set(out.keys())==set(WaveNeXtVocoder().state_dict().keys())` |
| `test_extract_prefers_ema_weights` | `ema_state_dict`(別値)付与時 `extract(prefer_ema=True)` が EMA由来値(`torch.equal`)。EMA非在時 state_dict フォールバック |
| `test_make_filelist_enumerates_all_wavs` | サブディレクトリ含む N個の.wav + 1個の.txt。`make_filelist(dir)` は .wav のみ N行・sorted・実在パス・.txt除外 |
| `test_config_loads_and_builds_model_cpu` | `load_config('configs_wavenext/wavenext_11025.yaml')` → dict。`build_model(cfg['model'])` が `WaveNeXtExp`、`feature_extractor` は `MatchaMelFeatures` かつ `.fmax==11025`、`automatic_optimization is False`。`no_grad` で `generator_forward(audio(1,16384))→(1,16384)`、`melspec_loss(audio_hat,audio)` finite スカラ |
| `test_fit_two_batches_manual_opt_cpu` | ダウンサイズ(dim=32,intermediate_dim=64,num_layers=1,num_samples=4096,batch_size=2,pretrain_mel_steps=0,max_steps=2)。`L.Trainer(accelerator='cpu',precision='32-true',max_epochs=1,limit_val_batches=0,logger=False,enable_checkpointing=False).fit(model,dm)` 完走、`n_batches>=1` |
| `test_existing_wavenext_suite_untouched` | `tests/test_wavenext.py` 緑維持の目印: `WaveNeXtVocoder()` param数 13.6–13.8M、state_dict は backbone./head. のみ、`head.linear_2.bias` 不在 |

---

## 3. 各モジュール確定実装 spec

全ファイル冒頭に、逐語移植分は `MIT License, Copyright (c) 2023 Charactr Inc.`(gemelo-ai/vocos fork)を明記。

### `wavenext_train/__init__.py`
空(1行 docstring 可)。パッケージマーカー。

### `wavenext_train/features.py` ★単一真実源

確定シグネチャ(§0 判断2)。`mel_spectrogram` は 2D `(B,T)` を要求するため入力次元を正規化してから呼ぶ(確認済み: 内部で `unsqueeze(1)`→pad384→`squeeze(1)`)。

```python
import torch
from torch import nn
from matcha.utils.audio import mel_spectrogram


class MatchaMelFeatures(nn.Module):
    """wetdog MelSpectrogramFeatures の forward契約を matcha mel で再実装(fmax=11025・非正規化)。
    forward(audio, **kwargs) -> (B, n_mels, T')。D11: STFT を FP32固定。"""
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024,
                 n_mels=80, fmin=0.0, fmax=11025.0, center=False):
        super().__init__()
        self.sample_rate = sample_rate; self.n_fft = n_fft; self.hop_length = hop_length
        self.win_length = win_length; self.n_mels = n_mels; self.fmin = fmin
        self.fmax = fmax; self.center = center

    def forward(self, audio, **kwargs):
        if audio.dim() == 3:      # (B,1,T) -> (B,T)  ※generator wrapper 経由の保険
            audio = audio.squeeze(1)
        elif audio.dim() == 1:    # (T,) -> (1,T)
            audio = audio.unsqueeze(0)
        device_type = "cuda" if audio.is_cuda else "cpu"   # verifier推奨: hardcode 'cuda' を動的化
        with torch.autocast(device_type=device_type, enabled=False):   # D11#6 stft FP32固定
            mel = mel_spectrogram(audio.float(), self.n_fft, self.n_mels, self.sample_rate,
                                  self.hop_length, self.win_length, self.fmin, self.fmax,
                                  center=self.center)
        return mel  # (B, 80, T'), 自然対数 log-mel、z-score正規化なし
```

無パラメータ(register_buffer/Parameter 無し)→ state_dict に一切出ない。

### `wavenext_train/dataset.py` ★verifier BLOCKER修正: sox除去 + soundfileロード

**修正点**: (a) module docstring に `sox_effects` 文字列を出さない(回帰ガードテストが FAIL する)。(b) `torchaudio.load` は torch2.10+ で torchcodec 依存クラッシュ → `soundfile.sf.read` に置換(`backend='soundfile'` 指定も 2.10 では無効、実測)。(c) `pytorch_lightning`→`lightning`。

```python
"""WaveNeXt GAN 学習用 波形データセット。
Ported from wetdog/wavenext_pytorch vocos/dataset.py. MIT License (c) 2023 Charactr Inc.
D11補正:
  - torchaudio の sox バックエンド(norm 効果)は torchaudio>=2.9 で削除 → pure-torch peak-norm へ置換。
  - torchaudio.load は torch>=2.10 で torchcodec 依存クラッシュ → soundfile で読み込み(repo規約)。
  - pytorch_lightning → lightning(本体依存)。"""
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import torch
import torchaudio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str; sampling_rate: int; num_samples: int; batch_size: int; num_workers: int


def load_audio(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    return torch.from_numpy(data.T).contiguous(), int(sr)       # (C, T)


def peak_normalize(y, gain_db):
    target = 10.0 ** (gain_db / 20.0)
    peak = y.abs().max()
    return y * (target / peak) if peak > 0 else y


class VocosDataset(Dataset):
    def __init__(self, cfg, train):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate; self.num_samples = cfg.num_samples; self.train = train
    def __len__(self): return len(self.filelist)
    def __getitem__(self, index):
        y, sr = load_audio(self.filelist[index])              # D11: torchaudio.load → soundfile
        if y.size(0) > 1: y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y = peak_normalize(y, float(gain))                    # D11: sox norm → pure-torch peak-norm
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start:start + self.num_samples]
        else:
            y = y[:, :self.num_samples]
        return y[0]


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params, val_params):
        super().__init__(); self.train_config = train_params; self.val_config = val_params
    def _get_dataloder(self, cfg, train):   # 綴りは upstream 逐語保持
        return DataLoader(VocosDataset(cfg, train=train), batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers, shuffle=train, pin_memory=True)
    def train_dataloader(self): return self._get_dataloder(self.train_config, train=True)
    def val_dataloader(self): return self._get_dataloder(self.val_config, train=False)
```

### `wavenext_train/discriminators.py` ★完全逐語(byte-for-byte ベンダー)

**実装時に raw を取得してそのまま置く**(§5 検証ゲート参照)。MITヘッダのみ追記、コード無改変。確定事実:
- import は `typing`(List/Optional/Tuple), `torch`, **`from einops import rearrange`**, `from torch import nn`, `from torch.nn import Conv2d`, `from torch.nn.utils import weight_norm`, **`from torchaudio.transforms import Spectrogram`**(byte-for-byte移植には einops + torchaudio.transforms が必須。**削らない**。両者とも本体依存)。
- `MultiPeriodDiscriminator(periods=(2,3,5,7,11), num_embeddings=None)`
- `DiscriminatorP(period, in_channels=1, kernel_size=5, stride=3, lrelu_slope=0.1, num_embeddings=None)` — forward は `x=x.unsqueeze(1); b,c,t=x.shape; ...; x.view(b,c,t//period,period)`。sub-disc 引数は `cond_embedding_id`。logits は `flatten(1,-1)` で2D。fmap 5本。
- `MultiResolutionDiscriminator(fft_sizes=(2048,1024,512), num_embeddings=None)`
- `DiscriminatorR(window_length, num_embeddings=None, channels=32, hop_factor=0.25, bands=((0.0,0.1),(0.1,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.0)))` — logits は flatten せず4D。fmap 21本。
- 両 `Multi*.forward(y, y_hat, bandwidth_id=None) -> (y_d_rs, y_d_gs, fmap_rs, fmap_gs)`

**ruff**: `Tuple`/`Optional` 注釈が UP006/UP045 を発火し `make format` が逐語を破壊する。→ §11 で `wavenext_train` を ruff exclude に追加(byte-identical 保全)。

### `wavenext_train/loss.py` ★Disc/Gen/FM は逐語、Mel のみ matcha mel(§0判断1)

```python
"""GAN(hinge)/FeatureMatching/mel-L1 losses.
Disc/Gen/FeatureMatching は wetdog vocos/loss.py 逐語。MIT License (c) 2023 Charactr Inc.
MelSpecReconstructionLoss のみ MatchaMelFeatures(fmax=11025) を再利用(torchaudio 不使用):
matcha mel_spectrogram は既に log(clamp(x,1e-5)) を適用済み=log-mel のため、upstream の safe_log を撤去(二重log回避)。"""
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from wavenext_train.features import MatchaMelFeatures


class MelSpecReconstructionLoss(nn.Module):
    def __init__(self, mel_features: MatchaMelFeatures = None, fmax: float = 11025):
        super().__init__()
        self.mel_features = mel_features or MatchaMelFeatures(fmax=fmax)
    def forward(self, y_hat, y):                 # arg順は wetdog 準拠
        return F.l1_loss(self.mel_features(y), self.mel_features(y_hat))


class GeneratorLoss(nn.Module):                  # 逐語: hinge relu(1-dg)
    def forward(self, disc_outputs):
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0)); gen_losses.append(l); loss += l
        return loss, gen_losses


class DiscriminatorLoss(nn.Module):              # 逐語: relu(1-real)+relu(1+fake)
    def forward(self, disc_real_outputs, disc_generated_outputs):
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses, g_losses = [], []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0)); g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss; r_losses.append(r_loss); g_losses.append(g_loss)
        return loss, r_losses, g_losses          # tensor を append(.item()化しない=逐語)


class FeatureMatchingLoss(nn.Module):            # 逐語: Σ mean(|rl-gl|)
    def forward(self, fmap_r, fmap_g):
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg): loss += torch.mean(torch.abs(rl - gl))
        return loss
```

### `wavenext_train/experiment.py` ★§0判断3/4 + verifier修正(EMA/lr/max_steps)反映

生成器は `matcha.wavenext.models` の `VocosBackbone`+`WaveNextHead` を **再利用**(2D `(B, T*hop)` head出力を判別器へ直接 — WaveNeXtVocoder wrapper の unsqueeze は使わない)。

```python
import math
import lightning as L
import torch
from matcha.wavenext.models import VocosBackbone, WaveNextHead
from wavenext_train.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from wavenext_train.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """transformers.get_cosine_schedule_with_warmup と数値等価な自前 LambdaLR(D11#4)。"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class WaveNeXtExp(L.LightningModule):
    def __init__(self, feature_extractor, backbone, head, melspec_loss,
                 sample_rate=22050, initial_learning_rate=1e-4, num_warmup_steps=500,
                 max_steps=1_000_000, mel_loss_coeff=45, mrd_loss_coeff=0.1,
                 pretrain_mel_steps=0, decay_mel_coeff=False, gradient_clip_val=None,
                 mpd_periods=(2, 3, 5, 7, 11), mrd_fft_sizes=(2048, 1024, 512),
                 use_ema=False, ema_decay=0.9995):
        super().__init__()
        self.automatic_optimization = False          # D11#1
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head", "melspec_loss"])
        self.feature_extractor = feature_extractor; self.backbone = backbone; self.head = head
        self.melspec_loss = melspec_loss
        self.multiperioddisc = MultiPeriodDiscriminator(periods=tuple(mpd_periods))
        self.multiresddisc = MultiResolutionDiscriminator(fft_sizes=tuple(mrd_fft_sizes))
        self.disc_loss = DiscriminatorLoss(); self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff
        self.use_ema = use_ema
        self.ema_decay = ema_decay                    # ★verifier修正: 未定義バグ回避
        # D11#2: バッチ単位カウンタ(global_step は 2/batch のため不使用)。persistent=True で resume 整合
        self.register_buffer("n_batches", torch.zeros(1, dtype=torch.long))
        if use_ema:
            self._ema_shadow = {n: p.detach().clone() for n, p in self._gen_named_params()}

    def _gen_named_params(self):
        for n, p in self.backbone.named_parameters(): yield f"backbone.{n}", p
        for n, p in self.head.named_parameters(): yield f"head.{n}", p

    def generator_forward(self, audio):               # D11#5: 2D(B,L)出力、FP32 mel は features 内で担保
        features = self.feature_extractor(audio)      # (B,80,T)
        return self.head(self.backbone(features))     # (B, T*hop) 2D

    def _should_train_disc(self):
        return int(self.n_batches.item()) >= self.hparams.pretrain_mel_steps

    def on_train_batch_start(self, *args):
        self.train_discriminator = self._should_train_disc()

    def configure_optimizers(self):                   # D11#4: index0=disc, index1=gen(順序固定)
        opt_d = torch.optim.AdamW([{"params": self.multiperioddisc.parameters()},
                                   {"params": self.multiresddisc.parameters()}],
                                  lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
        opt_g = torch.optim.AdamW([{"params": self.backbone.parameters()},
                                   {"params": self.head.parameters()}],   # feature_extractor 無パラメータ
                                  lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
        n = self.hparams.max_steps                    # ★バッチ単位、//2 廃止(§0判断4)
        sch_d = get_cosine_schedule_with_warmup(opt_d, self.hparams.num_warmup_steps, n)
        sch_g = get_cosine_schedule_with_warmup(opt_g, self.hparams.num_warmup_steps, n)
        return [opt_d, opt_g], [sch_d, sch_g]

    def _clip(self, opt):
        if self.hparams.gradient_clip_val:
            self.clip_gradients(opt, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")

    def training_step(self, batch, batch_idx):
        audio = batch; opt_d, opt_g = self.optimizers(); sch_d, sch_g = self.lr_schedulers()
        train_disc = self._should_train_disc()
        audio_hat = self.generator_forward(audio)     # D11#3: 生成器 forward は1回のみ
        out = {"loss_d": None, "loss_g": None, "mel_loss": None}
        if train_disc:
            opt_d.zero_grad(set_to_none=True)
            y_hat_d = audio_hat.detach()              # D11#3: 生成器へ勾配を流さない
            real_mp, gen_mp, _, _ = self.multiperioddisc(y=audio, y_hat=y_hat_d)
            real_mrd, gen_mrd, _, _ = self.multiresddisc(y=audio, y_hat=y_hat_d)
            loss_mp, loss_mp_real, _ = self.disc_loss(real_mp, gen_mp)
            loss_mrd, loss_mrd_real, _ = self.disc_loss(real_mrd, gen_mrd)
            loss_mp = loss_mp / len(loss_mp_real); loss_mrd = loss_mrd / len(loss_mrd_real)
            loss_d = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd
            self.manual_backward(loss_d); self._clip(opt_d); opt_d.step()
            out["loss_d"] = loss_d.detach()
        opt_g.zero_grad(set_to_none=True)
        if train_disc:
            _, gen_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(y=audio, y_hat=audio_hat)
            _, gen_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(y=audio, y_hat=audio_hat)
            loss_gen_mp, list_mp = self.gen_loss(gen_mp); loss_gen_mrd, list_mrd = self.gen_loss(gen_mrd)
            loss_gen_mp = loss_gen_mp / len(list_mp); loss_gen_mrd = loss_gen_mrd / len(list_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_rs_mp, fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_rs_mrd, fmap_gs_mrd) / len(fmap_rs_mrd)
        else:
            loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0.0
        mel_loss = self.melspec_loss(audio_hat, audio)   # FP32 mel は MatchaMelFeatures 内で担保
        loss_g = (loss_gen_mp + self.hparams.mrd_loss_coeff * loss_gen_mrd
                  + loss_fm_mp + self.hparams.mrd_loss_coeff * loss_fm_mrd
                  + self.mel_loss_coeff * mel_loss)
        self.manual_backward(loss_g); self._clip(opt_g); opt_g.step()
        out["loss_g"] = loss_g.detach(); out["mel_loss"] = mel_loss.detach()
        sch_d.step(); sch_g.step()                    # D11#2: 1バッチ1回
        if self.use_ema: self._update_ema()
        self.n_batches += 1
        if int(self.n_batches.item()) >= self.hparams.max_steps: self.trainer.should_stop = True
        return out

    def on_train_batch_end(self, *args):
        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * self._mel_coeff_decay(int(self.n_batches.item()))

    def _mel_coeff_decay(self, step, num_cycles=0.5):
        if step < self.hparams.num_warmup_steps: return 1.0
        progress = float(step - self.hparams.num_warmup_steps) / float(max(1, self.hparams.max_steps - self.hparams.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    @torch.no_grad()
    def _update_ema(self):
        cur = dict(self._gen_named_params())
        for k, shadow in self._ema_shadow.items(): shadow.mul_(self.ema_decay).add_(cur[k].detach(), alpha=1.0 - self.ema_decay)

    def validation_step(self, batch, batch_idx):
        mel_loss = self.melspec_loss(self.generator_forward(batch), batch)
        self.log("val/mel_loss", mel_loss, prog_bar=True, sync_dist=True); return mel_loss
```

### `wavenext_train/train.py` / `configs_wavenext/wavenext_11025.yaml` / 3スクリプト

- **train.py**: argparse + config。`load_config` は PyYAML 依存を避け **`OmegaConf.load`** を使う(omegaconf は hydra 経由で確実に存在。PyYAML は transitive のみ)。`build_model`/`build_datamodule`/`main`。`--config`/`--init-state`/`--ckpt-path`/`--train-filelist`/`--val-filelist`。`L.Trainer(precision=cfg trainer.precision既定"bf16-mixed", max_steps=-1, ...)`。**manual-opt のため `Trainer(gradient_clip_val=...)` は渡さない**(モジュール側 `self.clip_gradients`)。fused AdamW 不使用(bf16+fused+clip 非互換, CLAUDE.md)。`feature_extractor` の import は `from wavenext_train.features import MatchaMelFeatures`(§0判断2)。
- **config yaml**: `feature_extractor{sample_rate:22050,n_fft:1024,hop_length:256,n_mels:80,fmin:0,fmax:11025,center:false}`, `backbone{input_channels:80,dim:512,intermediate_dim:1536,num_layers:8}`(WAVENEXT_CONFIG一致=BSC重み流用可), `head{dim:512,n_fft:1024,hop_length:256,padding:same}`, `melspec_loss{fmax:11025}`(matcha mel — `MelSpecReconstructionLoss(fmax=11025)`。**旧draftの128-bin/norm/mel_scale/clip_val は削除**), `num_samples:16384(train)/48384(val)`, `initial_learning_rate:1.0e-4`, `mel_loss_coeff:45`, `mrd_loss_coeff:0.1`, `num_warmup_steps:500`, `max_steps:1000000`, `pretrain_mel_steps:2000`(stage1)/0(stage2), `gradient_clip_val:10.0`, `disc{mpd_periods:[2,3,5,7,11],mrd_fft_sizes:[2048,1024,512]}`, `trainer{precision:bf16-mixed,max_steps:-1,...}`。
- **scripts/init_wavenext_from_bsc.py**: `build_init_state_dict(bsc)` = `state_dict` 展開後 `startswith(("backbone.","head."))` フィルタ。
- **scripts/extract_wavenext_generator.py**: `extract(ckpt, prefer_ema=True)` = `_find_ema`(ema_state_dict/callback best-effort)優先→無ければ `state_dict`→`backbone./head.` フィルタ。main で `matcha.cli.load_wavenext(out,'cpu')` round-trip 検証。**matcha mel recon loss 採用により `melspec_loss.*` キーは元々出ないため drop対象は `multiperioddisc./multiresddisc./n_batches` のみ**。
- **scripts/make_wavenext_filelist.py**: `make_filelist(dir, recursive)` = `sorted(rglob('*.wav'))` を `str(resolve())`。

---

## 4. D11 チェックリスト(実装確認方法)

| # | 項目 | 実装 | 確認テスト/方法 |
|---|---|---|---|
| D11-1 | Lightning 2.x manual optimization | `automatic_optimization=False`、`manual_backward`+手動 `zero_grad/step`、scheduler 手動 `.step()` | `test_config_loads_and_builds_model_cpu`(`is False`)、`test_training_step_runs_full_gan...`(fit完走) |
| D11-2 | `n_batches` 会計(global_step非依存) | `register_buffer("n_batches")` を末尾で+1、gate/scheduler/停止すべて n_batches 基準 | `test_n_batches_buffer_increments_one_per_batch`(fit後==3、global_step≈6と乖離)、`test_pretrain_gate_toggles...` |
| D11-3 | detach で D-step の勾配遮断 | 生成器 forward 1回、D-step は `audio_hat.detach()` | `test_detach_blocks_generator_gradient_in_disc_step`(backbone/head `.grad is None`、MPD `.grad is not None`) |
| D11-4 | 自前 LambdaLR cosine+warmup(transformers不使用) | `get_cosine_schedule_with_warmup`(local) | `test_cosine_warmup_lambda_matches_formula`(fn(0/25/50/275/500))、grep で `transformers` import 無し |
| D11-5 | 生成器は matcha 再利用・2D出力 | `VocosBackbone`+`WaveNextHead` import、head 2D `(B,L)` を判別器へ直接 | `test_config_loads_and_builds`(generator_forward→`(1,16384)`)、models.py 確認済み(`view(b,-1)`) |
| D11-6 | STFT/mel を FP32固定 | `MatchaMelFeatures.forward` が `audio.float()`+`autocast(enabled=False)`。recon loss も同経由 | `test_stft_is_fp32_under_autocast_and_bf16_input`(dtype==float32) |
| D11-EMA | 任意EMA(既定OFF、byte-faithful) | `use_ema`、`self.ema_decay=ema_decay`(★修正)、shadow は非buffer | 既定OFFで全テスト緑。EMA有効時のcheckpoint永続化は §8 の既知制約 |
| D11-抽出 | 学習.ckpt→backbone./head.のみ→load_wavenext round-trip | extract フィルタ + `load_wavenext` の `not missing`/`not unexpected` assert | `test_extract_generator_roundtrips_load_wavenext`、`test_extract_prefers_ema_weights` |

---

## 5. wetdog 逐語検証ゲート(実装手順)

逐語対象は **`discriminators.py`(byte-for-byte)** と **`loss.py` の Disc/Gen/FM 3クラス(式レベル)**、**`dataset.py`(D11差分3点以外)**、**`experiment.py` の loss合成/configure_optimizers 構造**。

手順(discriminators を例に):
1. **raw取得**: `WebFetch https://raw.githubusercontent.com/wetdog/wavenext_pytorch/main/vocos/discriminators.py`。
2. **byte配置**: 取得内容をそのまま `wavenext_train/discriminators.py` に置き、先頭にMITヘッダのみ追記(コード無改変)。**推測で書かない**。
3. **署名 assert**(`test_default_periods_and_fft_sizes` 等): `periods==[2,3,5,7,11]`, `fft_sizes==[2048,1024,512]`, DiscriminatorP 5conv channel(1→32→128→512→1024→1024), DiscriminatorR band conv(3,9)/(3,3)。
4. **shape assert**(forward 4タプル、DiscriminatorP=2D/fmap5, DiscriminatorR=4D/fmap21)。
5. **数値ゲート**(loss): 手計算 hinge `relu(1-dr)+relu(1+dg)` と `allclose(atol=1e-6)`、gen は `relu(1-dg)` かつ `not allclose(-dg.mean())`。
6. **領域一致ゲート**(features/recon): `torch.equal(MatchaMelFeatures(11025)(a), mel_spectrogram(a,...,11025,center=False))`(**atol=0**)。

loss.py/dataset.py/experiment.py も同様に raw を突合し、**逐語部は式・属性名・返却タプル形状を1:1**、**D11差分部のみ** 差し替える。逐語からの意図的差分(下記)はコメントで明示し「transcription drift でない」と分かるようにする:
- dataset: `sox_effects.apply_effects_tensor`→`peak_normalize`、`torchaudio.load`→`soundfile`、`pytorch_lightning`→`lightning`、gain の `f"{gain:.2f}"` 丸め落ち(val は同値・train は乱数augのみで無害)。
- experiment: `global_step` property override→`n_batches` buffer、2回目 forward→`detach`、`transformers`→自前 LambdaLR、`optimizer_idx`(PL1.8)→Lightning2.x manual、`gen_params` から param-free `feature_extractor` 除外、scheduler `//2` 廃止(§0判断4)。
- loss: `MelSpecReconstructionLoss` を matcha mel 化+`safe_log` 撤去(§0判断1)。

---

## 6. 非破壊の保証

新規追加のみ。matcha側は **git diff 空**であること。

- **追加のみ**: `wavenext_train/`(7 py), `configs_wavenext/`(1 yaml), `scripts/`(3 py), `tests/`(6 test file)。
- **無改変**: `matcha/wavenext/{models,modules,vocoder}.py`, `matcha/cli.py`(`load_wavenext` 含む), `matcha/onnx/export.py`, `tests/test_wavenext.py`。生成器と `load_wavenext` は **import 再利用**(再移植しない)。
- **既存256テスト緑維持**: `test_existing_wavenext_suite_untouched` を回帰の目印にし、`make test` で全体緑を確認。
- **依存追加は soundfile のみ**(§11)。`transformers`/`encodec`/`pytorch_lightning` は導入しない。`einops`/`torchaudio`/`scipy`/`numpy`/`lightning` は既存本体依存。
- **抽出契約**: 学習.ckpt は `multiperioddisc./multiresddisc./n_batches` を含むため `load_wavenext` へ直渡し不可 → 必ず `extract_wavenext_generator.py` 経由(matcha mel recon loss 採用で `melspec_loss.*`/`feature_extractor.*` は元々キー無し)。`extract` は `set(out.keys())==set(WaveNeXtVocoder().state_dict().keys())` を満たし round-trip 成立。
- **VOCODER_URLS 不変更**: 学習binは `--vocoder-checkpoint-path` 直渡しで消費。DL試行を発生させない。
- **import 解決**: `tests/__init__.py`(0byte, 実在)により pytest prepend で repo-root が `sys.path[0]` に入る(conftest 無し・pythonpath 設定無しでも解決)。加えて **Step0 の `uv sync`** で setuptools `packages.find` が `wavenext_train` を editable install に登録 → **pytest 外(train.py/extract 等のスクリプト実行)でも CWD非依存で import 可能**になる(§11)。

---

## 7. CPU完結テスト vs GPU学習

- **テストは全て CPU 完結**(全6ファイル・計40件目安)。GAN forward/backward・manual-opt fit・領域一致 atol=0・scheduler 数値・extract round-trip すべて CPU で通る。合成wavは `soundfile` でローカル生成、ネットワーク不要。
- **`test_fit_two_batches` / experiment の fit系は `precision='32-true'`**(bf16 は CPU 不可)。bf16-mixed 経路自体は CPU で検証不能 → 本番GPUで確認。
- **fit系テストの所要**: 実 MPD+MRD を tiny入力で回すため数秒/件。閾値超過時は `@pytest.mark.slow` 付与を検討(既定 `make test` は slow スキップ)。ただし現状は unmarked で `make test` に含める方針。
- **GPU が必要なのは本番学習のみ**: RTX 5090 等で `bf16-mixed`(CLAUDE.md既定)、`model.optimizer.fused=false`。2段学習(stage1 MoeSpeech: BSC gen init+scratch disc+`pretrain_mel_steps=N` / stage2 つくよみ fine-tune: `pretrain_mel_steps=0`)。

---

## 8. 最大リスクと対策

| # | リスク | 深刻度 | 対策(確定) |
|---|---|---|---|
| R1 | **recon loss 領域の設計判断**(matcha 80-bin vs wetdog torchaudio 128-bin)。誤って torchaudio 版に戻すと領域不一致 | 高 | §0判断1で matcha mel 確定。loss.py に「torchaudio MelSpectrogram を再導入しない」コメント。`test_mel_recon...`(atol=0)+`isinstance(MatchaMelFeatures)` でガード |
| R2 | **dataset の torchaudio.load クラッシュ**(torch2.10 torchcodec、`backend='soundfile'` も無効) | 高(CPUテスト即死) | `soundfile.sf.read` へ置換(§3)。テストwav生成も soundfile |
| R3 | **ruff `make format` が逐語 discriminators.py を UP006/UP045 で自動改変** | 高(byte-identical破壊) | §11 で `[tool.ruff] exclude` に `wavenext_train` 追加(matcha/hifigan・matcha/wavenext と同じ前例) |
| R4 | **dataset docstring の `sox_effects` 文字列**が回帰ガードテストを FAIL | 高(GREEN不成立) | docstring から `sox_effects` 除去(§3 の文言に確定) |
| R5 | **experiment TEST1 の lr assert**(warmup中 `param_groups['lr']==0.0`) | 中(テスト赤) | `opt.defaults['lr']` で assert(§2.5) |
| R6 | **experiment TEST3 の loader が2バッチしか供給せず n_batches==3 不成立** | 中(テスト赤) | loader を ≥3バッチ供給に(`randn(6,...)` bs=2 + `limit_train_batches=3`) |
| R7 | **EMA有効時の `self.ema_decay` 未定義 / shadow が非checkpoint** | 中 | `__init__` に `self.ema_decay=ema_decay`。既定OFF。本番EMAは callback化 or shadow を buffer昇格(後続作業) |
| R8 | **import が tests/__init__.py 依存(pytest外で解決不能)** | 中 | Step0 で `uv sync` し packages.find に登録 → CWD非依存化。scripts は repo-root から実行 |
| R9 | **推論時 音響モデルの mel が fmax=8000 のままだと領域不一致**(エラー無・音質劣化) | 中(本コード範囲外) | 親計画で音響モデルを fmax=11025 で再学習/整合させること(併走する moespeech/tsukuyomi config が fmax11025) |
| R10 | **configure_optimizers の返却順(0=disc,1=gen)固定** | 低 | `# ORDER FIXED` コメント + `test_configure_optimizers` の param-set 同定でガード |
| R11 | **max_epochs のみ指定で trainer.max_steps=-1 → cosine 退化** | 低 | max_steps は model側(バッチ単位)で制御、停止は `n_batches>=max_steps`。config で明示 |
| R12 | **discriminators の byte-vendor で einops/torchaudio.transforms を誤削除** | 低 | §3/§5 に「削らない」明記。両者本体依存で動作 |

---

## 9. pyproject.toml 変更(必須・Step0)

matcha側コードは無改変だが、以下の**設定変更のみ**必要(非破壊・逐語保全のため)。

1. **`[tool.ruff] exclude` に `"wavenext_train"` 追加**(必須, R3/R4)。現状 `"matcha/hifigan", "matcha/wavenext"` の並びに追記。→ `uv run ruff check .` 緑維持 + `make format` の逐語自動改変を防止。
2. **`[project].dependencies` に `"soundfile"` 追加**(必須, R2)。現状 librosa 経由の transitive のみ。dataset/テストが直接使用するため明示宣言。
3. **`load_config` は `OmegaConf.load` を使用**(推奨, R? PyYAML transitive回避)。または `[project].dependencies` に `"PyYAML"` 追加。omegaconf は hydra-core で確実に存在するため前者推奨。
4. **`[tool.setuptools.packages.find]` は変更しない**(R8): `wavenext_train` を除外せず、`uv sync` で editable install に登録させ import を CWD非依存化。学習専用パッケージが wheel に載るのは無害(新規runtime依存なし・推論は非import)。

変更後 `uv sync && uv run ruff check . && make test` で全緑を確認してから Step1 に進む。