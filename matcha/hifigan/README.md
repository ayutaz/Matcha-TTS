# HiFi-GAN: 効率的かつ高忠実度な音声合成のための敵対的生成ネットワーク

### Jungil Kong, Jaehyeon Kim, Jaekyoung Bae

私たちの[論文](https://arxiv.org/abs/2010.05646)では、
高忠実度な音声を効率的に生成できるGANベースのモデルであるHiFi-GANを提案しました。<br/>
本リポジトリでは、実装および学習済みモデルをオープンソースとして公開しています。

**概要：**
音声合成に関する最近のいくつかの研究では、生の波形を生成するために敵対的生成ネットワーク（GAN）が採用されています。
これらの手法はサンプリング効率やメモリ使用量を改善するものの、
サンプル品質は自己回帰モデルやフローベースの生成モデルにはまだ及んでいません。
本研究では、効率的かつ高忠実度な音声合成を両立するHiFi-GANを提案します。
音声信号はさまざまな周期を持つ正弦波信号で構成されているため、
音声の周期的パターンのモデリングがサンプル品質の向上に不可欠であることを示します。
単一話者データセットに対する主観的人間評価（平均オピニオンスコア、MOS）では、提案手法が
単一のV100 GPU上でリアルタイムの167.9倍の速度で22.05 kHzの高忠実度音声を生成しながら、
人間の品質に匹敵する類似性を示すことが確認されました。さらに、未知の話者のメルスペクトログラム反転や
エンドツーエンド音声合成におけるHiFi-GANの汎用性を示します。最後に、HiFi-GANの小規模版は
CPU上でリアルタイムの13.4倍の速度でサンプルを生成し、自己回帰モデルに匹敵する品質を実現します。

音声サンプルは[デモウェブサイト](https://jik876.github.io/hifi-gan-demo/)をご覧ください。

## 前提条件

1. Python >= 3.6
2. 本リポジトリをクローンしてください。
3. Pythonの依存パッケージをインストールしてください。[requirements.txt](requirements.txt)を参照してください。
4. [LJ Speechデータセット](https://keithito.com/LJ-Speech-Dataset/)をダウンロードして展開してください。
   すべてのwavファイルを`LJSpeech-1.1/wavs`に移動してください。

## 学習

```
python train.py --config config_v1.json
```

V2またはV3のGeneratorを学習する場合は、`config_v1.json`を`config_v2.json`または`config_v3.json`に置き換えてください。<br>
チェックポイントと設定ファイルのコピーは、デフォルトで`cp_hifigan`ディレクトリに保存されます。<br>
`--checkpoint_path`オプションを追加することでパスを変更できます。

V1 Generatorでの学習中の検証損失。<br>
![検証損失](./validation_loss.png)

## 学習済みモデル

私たちが提供する学習済みモデルもご利用いただけます。<br/>
[学習済みモデルのダウンロード](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)<br/>
各フォルダの詳細は以下の通りです：

| フォルダ名    | Generator | データセット | ファインチューニング                                    |
| ------------ | --------- | ----------- | ------------------------------------------------------ |
| LJ_V1        | V1        | LJSpeech    | なし                                                   |
| LJ_V2        | V2        | LJSpeech    | なし                                                   |
| LJ_V3        | V3        | LJSpeech    | なし                                                   |
| LJ_FT_T2_V1  | V1        | LJSpeech    | あり ([Tacotron2](https://github.com/NVIDIA/tacotron2)) |
| LJ_FT_T2_V2  | V2        | LJSpeech    | あり ([Tacotron2](https://github.com/NVIDIA/tacotron2)) |
| LJ_FT_T2_V3  | V3        | LJSpeech    | あり ([Tacotron2](https://github.com/NVIDIA/tacotron2)) |
| VCTK_V1      | V1        | VCTK        | なし                                                   |
| VCTK_V2      | V2        | VCTK        | なし                                                   |
| VCTK_V3      | V3        | VCTK        | なし                                                   |
| UNIVERSAL_V1 | V1        | Universal   | なし                                                   |

他のデータセットへの転移学習のベースとして使用できるDiscriminator重み付きのユニバーサルモデルを提供しています。

## ファインチューニング

1. [Tacotron2](https://github.com/NVIDIA/tacotron2)を使用して、Teacher-Forcingによりnumpy形式のメルスペクトログラムを生成します。<br/>
   生成されたメルスペクトログラムのファイル名は音声ファイルと一致し、拡張子は`.npy`である必要があります。<br/>
   例：
   `   Audio File : LJ001-0001.wav
Mel-Spectrogram File : LJ001-0001.npy`
2. `ft_dataset`フォルダを作成し、生成されたメルスペクトログラムファイルをコピーします。<br/>
3. 以下のコマンドを実行します。
   ```
   python train.py --fine_tuning True --config config_v1.json
   ```
   その他のコマンドラインオプションについては、学習セクションを参照してください。

## wavファイルからの推論

1. `test_files`ディレクトリを作成し、wavファイルをそのディレクトリにコピーします。
2. 以下のコマンドを実行します。
   `   python inference.py --checkpoint_file [generatorチェックポイントファイルのパス]`
   生成されたwavファイルはデフォルトで`generated_files`に保存されます。<br>
   `--output_dir`オプションを追加することでパスを変更できます。

## エンドツーエンド音声合成の推論

1. `test_mel_files`ディレクトリを作成し、生成されたメルスペクトログラムファイルをそのディレクトリにコピーします。<br>
   メルスペクトログラムは[Tacotron2](https://github.com/NVIDIA/tacotron2)、
   [Glow-TTS](https://github.com/jaywalnut310/glow-tts)などを使用して生成できます。
2. 以下のコマンドを実行します。
   `   python inference_e2e.py --checkpoint_file [generatorチェックポイントファイルのパス]`
   生成されたwavファイルはデフォルトで`generated_files_from_mel`に保存されます。<br>
   `--output_dir`オプションを追加することでパスを変更できます。

## 謝辞

本実装にあたり、[WaveGlow](https://github.com/NVIDIA/waveglow)、[MelGAN](https://github.com/descriptinc/melgan-neurips)、
および[Tacotron2](https://github.com/NVIDIA/tacotron2)を参考にしました。
