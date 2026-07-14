# M3: Duration PredictorのFiLM話者条件付け + Blank zero-init

## マイルストーン概要

Duration Predictor（DP）は現在、話者情報への直接的なアクセスを持たない。encoder出力の`detach`コピー（`x_dp = torch.detach(x)`）を入力として受け取るのみであり、話者embeddingはencoderのConformer層を通過した間接的な混合表現としてしか供給されない。Alphacephei分析（2025/01）はこの構造的欠陥を「大きな問題」として明示的に指摘している。

100話者のJVS設定では、同一音素でも話者ごとに発話速度・リズムが大きく異なる。DPが話者を直接条件付けできないため、全話者の平均的なdurationに回帰し、個別話者の韻律パターンを再現できない。

本マイルストーンでは以下の2点を実施する:

1. **FiLM（Feature-wise Linear Modulation）** による話者条件付けをDPに追加し、話者embeddingから直接gamma/betaを生成してConv層出力を変調する
2. **Blank embedding zero-init** により、embedding空間でblank（index 0）と実音素の初期分離を促進し、MASおよびDPの精度向上を支援する

### 一から作り直すとしたらの思考

このマイルストーン全体を一から設計するとした場合:

- **FiLM vs Concatenation**: 現状のDPはencoder出力にspeaker embeddingがconcatされた256ch入力を受け取る。これは間接的な話者情報であり、`detach`によりencoderの勾配も遮断されている。FiLMを選択した理由は、(1) concatenation方式はすでに間接的に存在するが不十分であることが実証済み、(2) FiLMはnorm後の特徴量を直接スケール・シフトするため条件付けの表現力が高い、(3) identity init（gamma=1, beta=0）により既存重みとの互換性を完全に保証できる、の3点。Cross-attention方式も検討したが、DPの受容野が5トークンと狭く、グローバルな話者情報にはFiLMのような全チャネル一括変調が適切
- **Blank zero-init**: 代替として`nn.Embedding`の`padding_idx=0`を使う方法もあるが、これは勾配を完全にゼロにしてしまいblankの学習が不可能になる。zero-initは初期値のみゼロにして学習は許容するため、blankが必要な最小限のembeddingを獲得しつつ実音素との分離を維持できる
- **実装順序**: FiLM追加（T-M3-01）とBlank zero-init（T-M3-02）は独立した変更であり並行実施可能。ただしテスト時は組み合わせ効果の確認が必要

### 依存関係

```
M1: Julius Alignment ──→ M2: DataModule対応 ──→ M4: 学習実行 ──→ M5: 評価
                                                    ^
M3: FiLM DP + Blank init ─────────────────────────┘
```

- **前提**: なし（M1/M2と並行して実施可能）
- **ブロック**: M4（学習実行にはM3の完了が必要）
- **並行可能**: M1、M2

### 完了条件

1. DurationPredictorが`spks`引数を受け取り、FiLMで話者条件付けを行うこと
2. `n_spks=1`のときFiLM層が生成されず、既存の単一話者モデルと完全に同一の挙動を維持すること
3. FiLMのidentity init（gamma=1, beta=0）により、初期化直後の出力が変更前と一致すること
4. Blank embedding（index 0）がゼロベクトルで初期化されること
5. 全既存テスト（`make test`）がパスすること
6. 新規テストで上記1-4を検証すること
7. 既存チェックポイントのロード時にFiLM重みがない場合のフォールバック処理（または明示的なドキュメント）

### 想定期間

- T-M3-01（FiLM追加）: 0.5日（実装） + 0.5日（テスト・レビュー）
- T-M3-02（Blank zero-init）: 0.25日（実装・テスト）
- 合計: 約1日

---

## T-M3-01: DurationPredictorへのFiLM話者条件付け追加 {#t-m3-01}

### 1. タスク目的とゴール

DurationPredictorの各Conv1d層の出力に対して、話者embeddingから生成されたgamma（スケール）とbeta（シフト）を適用するFiLM（Feature-wise Linear Modulation）層を追加する。これにより、DPが話者ごとの発話速度・リズムパターンを直接学習可能になる。

**ゴール**:
- `n_spks > 1`のとき、DPが話者embeddingを直接受け取りFiLMで条件付けする
- `n_spks = 1`のとき、FiLM層は生成されず既存動作と完全同一
- identity init（gamma=1, beta=0）により、初期化直後は条件付けなしと同一出力を保証
- パラメータ増加: +66,560（現在の395,009の16.9%増）

### 2. 実装する内容の詳細

#### 2.1 DurationPredictor.__init__の変更

**ファイル**: `matcha/models/components/text_encoder.py` L85-97

**変更前**:
```python
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)
```

**変更後**:
```python
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_spks=1, spk_emb_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout
        self.n_spks = n_spks

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

        if n_spks > 1:
            # FiLM: speaker-conditioned affine transform after each LayerNorm
            # Each Linear produces gamma (scale) and beta (shift) for filter_channels
            self.film_1 = nn.Linear(spk_emb_dim, filter_channels * 2)
            self.film_2 = nn.Linear(spk_emb_dim, filter_channels * 2)
            # Identity init: gamma=1, beta=0 preserves pre-trained behavior
            self._init_film_identity(self.film_1, filter_channels)
            self._init_film_identity(self.film_2, filter_channels)

    @staticmethod
    def _init_film_identity(film_layer, filter_channels):
        """Initialize FiLM layer to identity transform (gamma=1, beta=0)."""
        nn.init.zeros_(film_layer.weight)
        nn.init.zeros_(film_layer.bias)
        # gamma portion of bias = 1.0 (first half)
        film_layer.bias.data[:filter_channels] = 1.0
```

**補足**:
- `_init_film_identity`で`weight`を全ゼロ、`bias`のgamma部分を1.0、beta部分を0.0に設定
- これにより入力`spks`の値に関わらず初期出力は`gamma=1, beta=0`、すなわち恒等変換
- `weight`がゼロのため、学習初期のFiLMは入力に無関係な固定値を出力し、既存重みの挙動を保存

#### 2.2 DurationPredictor.forwardの変更

**ファイル**: `matcha/models/components/text_encoder.py` L99-109

**変更前**:
```python
    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
```

**変更後**:
```python
    def forward(self, x, x_mask, spks=None):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        if self.n_spks > 1 and spks is not None:
            gamma_1, beta_1 = self.film_1(spks).chunk(2, dim=-1)
            # spks: (B, spk_emb_dim) -> film output: (B, filter_channels*2)
            # gamma_1, beta_1: (B, filter_channels) -> unsqueeze for (B, C, T) broadcast
            x = gamma_1.unsqueeze(-1) * x + beta_1.unsqueeze(-1)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        if self.n_spks > 1 and spks is not None:
            gamma_2, beta_2 = self.film_2(spks).chunk(2, dim=-1)
            x = gamma_2.unsqueeze(-1) * x + beta_2.unsqueeze(-1)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
```

**補足**:
- FiLM適用位置は`norm`の後、`drop`の前。LayerNormで正規化された特徴量に対してスケール・シフトを適用する
- `spks`は`(B, spk_emb_dim)`形状のテンソル（既にembedding lookupされた64次元ベクトル）
- `unsqueeze(-1)`で`(B, C, 1)`に変形し、時間軸方向にbroadcast
- `n_spks == 1`または`spks is None`のとき、FiLMは完全にスキップされオーバーヘッドゼロ

#### 2.3 TextEncoder.__init__でのDurationPredictor生成の変更

**ファイル**: `matcha/models/components/text_encoder.py` L397-402

**変更前**:
```python
        self.proj_w = DurationPredictor(
            self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )
```

**変更後**:
```python
        self.proj_w = DurationPredictor(
            self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
```

#### 2.4 TextEncoder.forwardでのspks引数パススルー

**ファイル**: `matcha/models/components/text_encoder.py` L433-434

**変更前**:
```python
        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)
```

**変更後**:
```python
        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask, spks=spks)
```

**補足**:
- `spks`は`TextEncoder.forward`の引数として既に受け取っている（L404: `def forward(self, x, x_lengths, spks=None)`）
- `n_spks=1`のとき`spks=None`が渡され、DP内部で条件分岐によりFiLMはスキップされる
- `n_spks > 1`のとき`spks`は`(B, 64)`のembeddingテンソル（`MatchaTTS.forward`のL184で`self.spk_emb(spks)`済み）

#### 2.5 チェックポイント互換性対応

既存チェックポイント（FiLM layer未定義）からのロード時に`film_1`, `film_2`キーが欠損する。
以下の2つの対応方針を実装する:

**方針A: 新規学習（推奨）**
- 外部アライナーdurationを使用する新しい学習は一から開始するため、チェックポイント互換性は不要
- jvs_aligned.yamlでは`ckpt_path`を指定しない

**方針B: 既存チェックポイントからのfine-tune（将来対応）**
- DurationPredictorに`load_state_dict`オーバーライドを追加:
```python
def load_state_dict(self, state_dict, strict=True):
    # FiLM keys missing from old checkpoints → use identity init
    if self.n_spks > 1:
        for key in ['film_1.weight', 'film_1.bias', 'film_2.weight', 'film_2.bias']:
            if key not in state_dict:
                state_dict[key] = getattr(self, key.rsplit('.', 1)[0]).state_dict()[key.rsplit('.', 1)[1]]
    super().load_state_dict(state_dict, strict=strict)
```

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| 実装エージェント | 1 | DurationPredictor/TextEncoderの変更、FiLM層の追加 |
| テストエージェント | 1 | 単体テスト・回帰テストの作成と実行 |

合計2エージェント。変更が`text_encoder.py`1ファイルに集中するため少人数で実施可能。

### 4. 提供範囲とテスト項目

#### 提供範囲

- `matcha/models/components/text_encoder.py`の変更（DurationPredictor + TextEncoder）
- `tests/test_text_encoder.py`への新規テスト追加

#### テスト項目

以下のテストを`tests/test_text_encoder.py`に追加する。

**テストクラス: `TestDurationPredictorFiLM`**

| テスト名 | アサーション |
|---------|------------|
| `test_dp_film_instantiation_multispeaker` | `n_spks=2`でDPを生成し、`hasattr(dp, 'film_1')` と `hasattr(dp, 'film_2')` が`True` |
| `test_dp_no_film_single_speaker` | `n_spks=1`でDPを生成し、`hasattr(dp, 'film_1')` が`False` |
| `test_dp_film_identity_init` | 初期化直後の`film_1`で任意の`spks`入力に対し`gamma`が全て`1.0`、`beta`が全て`0.0` |
| `test_dp_film_output_shape` | `n_spks=2`のDP forward出力が`(B, 1, T)`形状 |
| `test_dp_film_identity_preserves_output` | identity init状態でFiLMありとなしの出力が`torch.allclose`で一致（rtol=1e-5） |
| `test_dp_film_different_speakers_different_output` | 学習後（ランダム重み）に異なる`spks`で異なる`logw`を生成 |
| `test_dp_film_parameter_count` | FiLM追加後のパラメータ数が`filter_channels=256, spk_emb_dim=64`で+66,560 |
| `test_dp_forward_without_spks_multispeaker` | `n_spks=2`のDPに`spks=None`を渡してもエラーにならない（FiLMスキップ） |
| `test_dp_load_old_checkpoint_without_film` | FiLM keyなしのstate_dictをロードしてもエラーにならないことを検証（`load_state_dict`オーバーライドにより欠損キーが自動補完される） |
| `test_dp_load_old_checkpoint_preserves_identity` | 欠損FiLM keyがidentity init値で補完されることを検証（補完後の`film_1`出力が`gamma=1, beta=0`） |

**テストクラス: `TestTextEncoderFiLMIntegration`**

| テスト名 | アサーション |
|---------|------------|
| `test_encoder_passes_spks_to_dp` | `n_spks=2`のTextEncoder forwardで`spks`を渡し、出力形状が正しい |
| `test_encoder_multispeaker_output_shape` | 多話者TextEncoderの`mu, logw, x_mask`形状が`(B, 80, T), (B, 1, T), (B, 1, T)` |
| `test_encoder_single_speaker_unchanged` | `n_spks=1`のTextEncoderの出力が変更前と同一（回帰テスト） |

**既存テストの回帰確認**:
- `make test`で全256テストがパスすることを確認
- 特に`tests/test_text_encoder.py`の既存全テストが変更なしでパス

### 5. 懸念事項とレビュー項目

| 懸念事項 | 対策 | レビュー時の確認 |
|---------|------|----------------|
| **既存チェックポイントとの互換性** | FiLM重みは`n_spks > 1`時のみ生成。`n_spks=1`の英語モデル（LJSpeech）は完全互換。多話者の既存JVSチェックポイントはFiLM重みが欠損するため、方針A（新規学習、`ckpt_path`指定なし）または方針B（`load_state_dict`オーバーライドによるidentity init自動補完）で対応する（詳細は2.5節を参照） | 方針Aの場合: 新規学習で`ckpt_path`が未指定であることを確認。方針Bの場合: `load_state_dict`オーバーライドによりFiLM重みがidentity initされることをテストで確認 |
| **FP32精度** | FiLMの`gamma * x + beta`演算はFP32では問題なし。FP16ではgammaが大きくなるとoverflow可能性あり | JVS学習設定が`precision="32-true"`であることを確認 |
| **勾配フロー** | `x_dp = torch.detach(x)`によりencoderへの勾配は遮断済み。FiLMの勾配は`spks`を経由して`spk_emb`に流れるため、話者embeddingがDP損失からも学習される | `spk_emb`の勾配が`dur_loss`と`diff_loss`の両方から供給されることを確認 |
| **FiLM適用位置** | norm後（gamma/betaがLayerNormのgamma/betaと機能的に重複する可能性）。ただしLayerNormのgamma/betaはチャネル共通、FiLMのgamma/betaは話者条件付きなので役割は異なる | 適用位置がnorm後、drop前であることをコードレビューで確認 |
| **`spks=None`のハンドリング** | `n_spks > 1`でも推論時のテスト等で`spks=None`が渡される可能性。条件分岐`self.n_spks > 1 and spks is not None`で安全にスキップ | 条件分岐のロジックをレビュー |
| **DDP学習** | FiLM層は通常のParameterでありDDPの同期対象。特別な対応不要 | DDP設定との競合がないことを確認 |

### 6. 一から作り直すとしたら

- **AdaIN（Adaptive Instance Normalization）方式**: LayerNormをまるごとFiLM対応のAdaINに置き換える方式も検討可能。しかしLayerNormの`gamma/beta`と機能重複し、既存重みの互換性が破壊される。FiLMをnorm後に追加する方式なら既存のnorm重みを保持したまま話者条件付けを上乗せできるため、現方式が最適
- **DPアーキテクチャ自体の拡張**: 受容野5トークン（2層Conv, k=3）は日本語のプロソディ文脈には狭い。dilation追加や層数増加も検討に値するが、M3スコープ外（CLAUDE.mdの「~20行」の変更方針に準拠）。将来的にはM5の評価結果を見て判断
- **FiLM層数**: 現設計ではconv_1/conv_2の各normの後に1つずつ（計2層）。proj（最終1x1 conv）の後には追加しない。projは1チャネル出力のため、FiLMのスケール/シフトは単純なバイアス加算と等価になり効果が薄い

### 7. 後続タスクへの連絡事項

- **M4（学習実行）**: FiLMのidentity initにより、学習初期は条件付けなしと同等の挙動。FiLM重みの学習はoptimizer設定を変更する必要はない（同一のlr=1e-4、AdamWで学習可能）
- **M4（チェックポイント）**: 新規学習の場合は`ckpt_path`指定不要（方針A）。既存チェックポイントからfine-tuneする場合は`load_state_dict`オーバーライドによりFiLM重みがidentity initで自動補完される（方針B）。詳細は2.5節を参照
- **M5（評価）**: 評価時にはDP出力の話者依存性を検証する。同一テキストで異なる話者を指定した際にduration分布が異なることを確認する
- **T-M3-02**: Blank zero-initはembedding初期化のみでFiLMとは完全に独立。並行実施可能

---

## T-M3-02: Blank embedding zero-init {#t-m3-02}

### 1. タスク目的とゴール

TextEncoderの音素embedding（`self.emb`）において、blank（index 0）のembeddingベクトルをゼロで初期化する。intersperse処理（音素列の各音素間にblank=0を挿入）により、入力シーケンスの約50%がblankで占められる。初期化時点でblankと実音素のembeddingが類似していると、encoderのmu_x出力でblank/phonemeの区別が困難になり、MASがblankにフレームを大量割当する退化を助長する。

**ゴール**:
- blank（index 0）のembeddingを`torch.zeros`で初期化
- 実音素のembeddingは従来通り`Normal(0, n_channels^{-0.5})`で初期化（変更なし）
- blankは学習可能のまま維持（`padding_idx`は使用しない）
- embedding空間でblankと実音素の初期距離を最大化

### 2. 実装する内容の詳細

#### 2.1 TextEncoder.__init__でのblank embedding zero-init

**ファイル**: `matcha/models/components/text_encoder.py` L372-373

**変更前**:
```python
        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)
```

**変更後**:
```python
        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)
        # Zero-init blank embedding (index 0) to promote blank/phoneme separation
        # in encoder mu_x output. Blank remains trainable (no padding_idx).
        self.emb.weight.data[0].zero_()
```

**補足**:
- 1行の追加のみ
- `self.emb.weight.data[0].zero_()`はin-place操作で、`Normal`初期化後にindex 0だけを上書き
- `padding_idx=0`を使用しない理由: `padding_idx`はforward時に当該indexのembeddingの勾配を恒久的にゼロにする。blankは学習対象として維持し、学習を通じてblankに最適なembeddingを獲得させる必要がある
- 初期状態でblankがゼロベクトル、実音素が`N(0, σ)`で初期化されるため、embedding空間のL2距離がblank-phoneme間で`O(σ * sqrt(n_channels))`、phoneme-phoneme間で`O(σ * sqrt(2 * n_channels))`となる。blankは原点に位置し、実音素はn_channels次元超球面上に分布する

#### 2.2 数値的影響の確認

`self.n_channels = 192`の場合:
- `σ = 192^{-0.5} = 0.0722`
- 実音素embeddingのL2ノルム期待値: `σ * sqrt(n_channels) = 0.0722 * sqrt(192) = 1.0`
- blank embeddingのL2ノルム: `0.0`
- blank-phoneme L2距離期待値: `~1.0`
- phoneme-phoneme L2距離期待値: `σ * sqrt(2 * n_channels) = ~1.41`

この初期分離により、encoderの初期段階からblankと実音素のmu_x出力に差異が生まれ、MASのlog_prior計算でblank位置が実音素と区別されやすくなる。

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| 実装・テストエージェント | 1 | 1行変更 + テスト作成・実行 |

合計1エージェント。変更が最小限のため1人で完結。

### 4. 提供範囲とテスト項目

#### 提供範囲

- `matcha/models/components/text_encoder.py`の1行追加
- `tests/test_text_encoder.py`への新規テスト追加

#### テスト項目

以下のテストを`tests/test_text_encoder.py`に追加する。

**テストクラス: `TestBlankEmbeddingZeroInit`**

| テスト名 | アサーション |
|---------|------------|
| `test_blank_embedding_is_zero_after_init` | `TextEncoder`初期化直後に`encoder.emb.weight.data[0]`が全ゼロ: `torch.all(encoder.emb.weight.data[0] == 0.0)` |
| `test_phoneme_embeddings_are_nonzero` | index 1以降のembeddingがゼロでない: `torch.any(encoder.emb.weight.data[1:] != 0.0)` |
| `test_blank_embedding_is_trainable` | `encoder.emb.weight.requires_grad`が`True`（`padding_idx`を使用していないことの確認） |
| `test_blank_embedding_receives_gradient` | ダミーforward + backward後に`encoder.emb.weight.grad[0]`がゼロでない（勾配が流れることの確認） |
| `test_blank_phoneme_l2_distance` | 初期化直後にblank（index 0）と任意の実音素（index 1-54）のL2距離が0より大きい |

**既存テストの回帰確認**:
- `make test`で全256テストがパスすることを確認
- 特に`tests/test_text_encoder.py`の`TestTextEncoderInstantiation::test_embedding_shape`が変更なしでパス

### 5. 懸念事項とレビュー項目

| 懸念事項 | 対策 | レビュー時の確認 |
|---------|------|----------------|
| **勾配消失リスク** | ゼロベクトルはReLU等で勾配が0になる可能性がある。ただしembeddingは直接activationに通されず、`* math.sqrt(n_channels)`でスケーリング後にtranspose→prenet→encoder→MASの経路をたどる。ゼロスケーリングは`0 * sqrt(192) = 0`だがprenetのConv層（k=5）が隣接音素から情報を注入するため勾配は流れる | backward後のblank embedding勾配がゼロでないことをテストで確認 |
| **既存チェックポイントとの互換性** | 初期化時のみの変更であり、`state_dict`のロード時にはチェックポイントの値で上書きされる。既存チェックポイントのblank embeddingはゼロではないため、ロード後の動作に影響なし | ロードテスト不要（初期化のみの変更） |
| **`math.sqrt(n_channels)`スケーリングとの相互作用** | `x = self.emb(x) * math.sqrt(self.n_channels)`（L423）により、embeddingはsqrt(192)=13.86倍される。blank=0のときスケーリング後も0。これは意図された動作（blankが「空の入力」として機能） | L423のスケーリングがblank=0に適用されることを理解した上でレビュー |
| **日本語55語彙 vs 英語178語彙** | zero-initはblank index=0のみに適用されるため、語彙サイズに無関係。英語モデルにも安全に適用可能 | `n_vocab`に依存しないことを確認 |
| **学習中のblank embeddingの挙動** | 学習が進むとblankのembeddingはゼロから離れて最適値に収束する。初期のゼロ状態はあくまで学習の出発点を制御するもの | 長期学習でblank embeddingが意味のある値を獲得することをM5で確認 |

### 6. 一から作り直すとしたら

- **`padding_idx=0`方式**: `nn.Embedding(n_vocab, n_channels, padding_idx=0)`を使えば常にゼロが維持される。しかしblankは学習対象であるべき。blankは単なるパディングではなく、intersperse後のシーケンスで音素間の遷移を表現する重要な要素。勾配を完全にゼロにすると、blankの最適なembeddingが獲得できず、duration予測とメル生成の両方に悪影響
- **小さな定数での初期化**: `self.emb.weight.data[0].fill_(1e-6)`のように極小値で初期化する方式も考えられる。しかし`0.0`と比較してメリットがなく、コードの意図が不明瞭になる
- **全blankの特殊初期化**: intersperse後のシーケンスでは奇数位置が全てblank（index 0）。共通のembeddingを参照するため、index 0のzero-initで全blankに適用される。位置ごとに異なるblank embeddingを持つ設計も考えられるが、アーキテクチャ変更が大きすぎるためM3スコープ外

### 7. 後続タスクへの連絡事項

- **M4（学習実行）**: blank zero-initは新規学習からのみ有効。既存チェックポイントからの再開時はチェックポイントのembeddingが優先される。blank zero-initの効果を検証するには新規学習（`ckpt_path`なし）が必要
- **M5（評価）**: 学習中のblank embeddingのL2ノルム推移をログすることを推奨。ゼロから出発して適切な値に収束するかを確認する。`TensorBoardLogger`のcustom scalarで`||emb[0]||_2`を記録
- **T-M3-01**: Blank zero-initはFiLMとは完全に独立した変更。同一ファイル（`text_encoder.py`）の異なる箇所を修正するため、gitのマージ競合は発生しない
- **外部アライナー（M1/M2）との相互作用**: blank zero-initは`use_precomputed_durations=true`の学習でも有効。encoderのmu_x出力品質が向上し、prior_lossの収束が改善される可能性がある
