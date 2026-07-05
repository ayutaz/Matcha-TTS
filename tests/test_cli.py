"""Tests for the Matcha-TTS command line interface (matcha/cli.py).

Covers:
  - process_text: Japanese token-level blank-skip display (regression for
    commit a965cbd where char-level slicing corrupted multi-char phonemes
    like 'ch') and the English char-level path
  - cli(): language auto-detection from the model's n_vocab attribute
  - batched_synthesis: length-sorting with original-index mapping for
    output file naming
  - batched_collate_fn: right zero-padding and order preservation
  - validate_args: custom-checkpoint branch
  - get_texts: UTF-8 file reading, newline stripping, empty-line dropping
  - assert_required_models_available: pretrained download skipped for custom
    checkpoints
"""

import argparse
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Module-level sys.modules mocking (intentional; same pattern as
# tests/test_text_ja.py).
#
# matcha.cli transitively imports matcha.text.cleaners, which references the
# ``phonemizer`` package.  Installing a minimal stub before any matcha import
# keeps this module importable in environments without espeak/phonemizer.
# ---------------------------------------------------------------------------
_fake_phonemizer = types.ModuleType("phonemizer")
_fake_backend = types.ModuleType("phonemizer.backend")


class _FakeEspeakBackend:
    """Minimal stand-in so cleaners.py can be imported."""

    def __init__(self, **kwargs):
        pass

    def phonemize(self, text_list, strip=True, njobs=1):
        return text_list


_fake_backend.EspeakBackend = _FakeEspeakBackend
_fake_phonemizer.backend = _fake_backend

_fake_espeak = types.ModuleType("phonemizer.backend.espeak")
_fake_espeak_espeak = types.ModuleType("phonemizer.backend.espeak.espeak")
_fake_backend.espeak = _fake_espeak
_fake_espeak.espeak = _fake_espeak_espeak

sys.modules["phonemizer"] = _fake_phonemizer
sys.modules["phonemizer.backend"] = _fake_backend
sys.modules["phonemizer.backend.espeak"] = _fake_espeak
sys.modules["phonemizer.backend.espeak.espeak"] = _fake_espeak_espeak

# Now it is safe to import matcha modules ------------------------------------
import matcha.cli as cli_module  # noqa: E402
from matcha.cli import (  # noqa: E402
    batched_collate_fn,
    batched_synthesis,
    get_texts,
    process_text,
    validate_args,
)
from matcha.text import cleaned_text_to_sequence, text_to_sequence  # noqa: E402

# ---------------------------------------------------------------------------
# 1. process_text
# ---------------------------------------------------------------------------


class TestProcessText:
    """Test text preprocessing including the phonetised-text display."""

    def test_japanese_token_level_display(self, capsys):
        """Japanese display must skip blanks token-wise, keeping 'ch' intact."""
        text = "k o n n i ch i w a"
        out = process_text(0, text, torch.device("cpu"), cleaners=["basic_cleaners"], language="ja")

        seq = cleaned_text_to_sequence(text, language="ja")
        # Odd positions hold the phoneme ids, even positions the blank (0)
        assert out["x"][0, 1::2].tolist() == seq
        assert (out["x"][0, 0::2] == 0).all()
        assert out["x_lengths"].item() == 2 * len(seq) + 1
        assert out["x_orig"] == text

        captured = capsys.readouterr().out
        phonetised_lines = [line for line in captured.splitlines() if "Phonetised text" in line]
        assert len(phonetised_lines) == 1
        # Regression for a965cbd: char-level slicing would corrupt 'ch' into
        # fragments; token-level [1::2] on split() restores the input string.
        assert phonetised_lines[0] == f"[0] - Phonetised text: {text}"
        assert "ch" in phonetised_lines[0].split()

    def test_english_char_level_display(self, capsys):
        """English keeps the original char-level x_phones[1::2] display."""
        out = process_text(7, "hello", torch.device("cpu"), cleaners=["basic_cleaners"], language="en")

        seq = cleaned_text_to_sequence("hello", language="en")
        assert out["x"][0, 1::2].tolist() == seq
        assert (out["x"][0, 0::2] == 0).all()
        assert out["x_lengths"].item() == 2 * len(seq) + 1
        # x_phones interleaves the pad symbol '_' between characters
        assert out["x_phones"] == "_h_e_l_l_o_"

        captured = capsys.readouterr().out
        assert "[7] - Phonetised text: hello" in captured

    def test_default_cleaners_japanese(self, monkeypatch):
        """cleaners=None with language='ja' must default to japanese_cleaners."""
        recorded = {}

        def fake_text_to_sequence(text, cleaner_names, *, language="en"):
            recorded["cleaners"] = cleaner_names
            recorded["language"] = language
            return [1, 2, 3], "cleaned"

        monkeypatch.setattr(cli_module, "text_to_sequence", fake_text_to_sequence)
        process_text(0, "こんにちは", torch.device("cpu"), cleaners=None, language="ja")

        assert recorded["cleaners"] == ["japanese_cleaners"]
        assert recorded["language"] == "ja"

    def test_default_cleaners_english(self, monkeypatch):
        """cleaners=None with language='en' must default to english_cleaners2."""
        recorded = {}

        def fake_text_to_sequence(text, cleaner_names, *, language="en"):
            recorded["cleaners"] = cleaner_names
            recorded["language"] = language
            return [1, 2, 3], "cleaned"

        monkeypatch.setattr(cli_module, "text_to_sequence", fake_text_to_sequence)
        process_text(0, "hello", torch.device("cpu"), cleaners=None, language="en")

        assert recorded["cleaners"] == ["english_cleaners2"]
        assert recorded["language"] == "en"


# ---------------------------------------------------------------------------
# 2. cli() language auto-detection from n_vocab
# ---------------------------------------------------------------------------


class TestCliLanguageAutoDetect:
    """Test cli()'s auto-detection of language from the loaded model."""

    def _invoke_cli(self, monkeypatch, tmp_path, model_stub, extra_argv=()):
        """Run cli() with all heavy dependencies stubbed; return recorded call."""
        ckpt = tmp_path / "custom.ckpt"
        ckpt.write_bytes(b"\x00")
        argv = [
            "matcha-tts",
            "--text",
            "hello",
            "--checkpoint_path",
            str(ckpt),
            "--vocoder",
            "hifigan_univ_v1",
            *extra_argv,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(cli_module, "get_device", lambda args: torch.device("cpu"))
        monkeypatch.setattr(
            cli_module,
            "assert_required_models_available",
            lambda args: {"matcha": args.checkpoint_path, "vocoder": None},
        )
        monkeypatch.setattr(cli_module, "load_matcha", lambda name, path, device: model_stub)
        monkeypatch.setattr(cli_module, "load_vocoder", lambda name, path, device: (MagicMock(), MagicMock()))

        recorded = {}

        def record_synthesis(args, device, model, vocoder, denoiser, texts, spk):
            recorded["args"] = args
            recorded["texts"] = texts
            recorded["spk"] = spk

        monkeypatch.setattr(cli_module, "unbatched_synthesis", record_synthesis)
        monkeypatch.setattr(cli_module, "batched_synthesis", record_synthesis)

        cli_module.cli()
        return recorded

    def test_n_vocab_55_detects_japanese(self, monkeypatch, tmp_path, capsys):
        model = types.SimpleNamespace(n_vocab=55, n_spks=1)
        recorded = self._invoke_cli(monkeypatch, tmp_path, model)

        assert recorded["args"].language == "ja"
        assert recorded["args"].cleaners == ["japanese_cleaners"]
        assert "Auto-detected language: Japanese (n_vocab=55)" in capsys.readouterr().out

    def test_n_vocab_178_detects_english(self, monkeypatch, tmp_path):
        model = types.SimpleNamespace(n_vocab=178, n_spks=1)
        recorded = self._invoke_cli(monkeypatch, tmp_path, model)

        assert recorded["args"].language == "en"
        assert recorded["args"].cleaners is None

    def test_explicit_language_overrides_auto_detection(self, monkeypatch, tmp_path):
        model = types.SimpleNamespace(n_vocab=55, n_spks=1)
        recorded = self._invoke_cli(monkeypatch, tmp_path, model, extra_argv=("--language", "en"))

        assert recorded["args"].language == "en"
        assert recorded["args"].cleaners is None

    def test_model_without_n_vocab_defaults_to_english(self, monkeypatch, tmp_path):
        model = types.SimpleNamespace(n_spks=1)  # no n_vocab attribute
        recorded = self._invoke_cli(monkeypatch, tmp_path, model)

        assert recorded["args"].language == "en"
        assert recorded["args"].cleaners is None


# ---------------------------------------------------------------------------
# 3. batched_synthesis — length sort + original-index mapping
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Fake model whose mel output encodes the first non-blank token id."""

    def __init__(self):
        self.batch_lengths = []
        self.batch_spks = []

    def synthesise(self, x, x_lengths, n_timesteps, temperature, spks, length_scale):
        self.batch_lengths.append(x_lengths.tolist())
        self.batch_spks.append(spks)
        b = x.shape[0]
        # x[:, 1] is the first real (non-blank) token after intersperse
        mel = x[:, 1].to(torch.float32).view(b, 1, 1).expand(b, 2, 4).contiguous()
        return {
            "mel": mel,
            "mel_lengths": torch.full((b,), 4, dtype=torch.long),
            "rtf": 0.0,
        }


class TestBatchedSynthesis:
    """Test that outputs map back to the original text order despite sorting."""

    # Distinct first letters and clearly different lengths so that the
    # length-sorted order [1, 3, 4, 2, 0] differs from the original order.
    TEXTS = ["ccccc", "a", "dddd", "bb", "eee"]

    def _run(self, monkeypatch, tmp_path, texts, spk_id=None, batch_size=2):
        saved = []

        def fake_save_to_folder(filename, output, folder):
            saved.append((filename, output))
            return Path(folder) / f"{filename}.wav"

        monkeypatch.setattr(cli_module, "save_to_folder", fake_save_to_folder)
        monkeypatch.setattr(
            cli_module, "to_waveform", lambda mel, vocoder, denoiser=None, denoiser_strength=0.00025: mel
        )

        # The hardcoded num_workers=8 would spawn worker processes (very slow
        # on Windows); force num_workers=0 via a wrapper.
        real_dataloader = torch.utils.data.DataLoader

        def dataloader_without_workers(*dl_args, **dl_kwargs):
            dl_kwargs["num_workers"] = 0
            return real_dataloader(*dl_args, **dl_kwargs)

        monkeypatch.setattr(torch.utils.data, "DataLoader", dataloader_without_workers)

        args = argparse.Namespace(
            cleaners=["basic_cleaners"],
            language="en",
            batch_size=batch_size,
            steps=2,
            temperature=0.0,
            speaking_rate=1.0,
            denoiser_strength=0.00025,
            output_folder=str(tmp_path),
            spk=spk_id,
        )
        model = _RecordingModel()
        spk = torch.tensor([spk_id], dtype=torch.long) if spk_id is not None else None
        batched_synthesis(args, torch.device("cpu"), model, MagicMock(), MagicMock(), texts, spk)
        return saved, model

    def test_original_index_mapping_across_batches(self, monkeypatch, tmp_path):
        saved, model = self._run(monkeypatch, tmp_path, self.TEXTS)

        assert len(saved) == len(self.TEXTS)
        # Every original index gets exactly one output file
        assert sorted(name for name, _ in saved) == [f"utterance_{i:03d}" for i in range(len(self.TEXTS))]

        # The mel payload (first non-blank token id) identifies the source
        # text, so each utterance_{i} must correspond to TEXTS[i].
        for name, output in saved:
            orig_idx = int(name.split("_")[1])
            expected_id = cleaned_text_to_sequence(self.TEXTS[orig_idx], language="en")[0]
            assert output["mel"].flatten()[0].item() == pytest.approx(float(expected_id))

    def test_batches_are_length_sorted(self, monkeypatch, tmp_path):
        _, model = self._run(monkeypatch, tmp_path, self.TEXTS)

        # x lengths are 2*len(text)+1: original [11, 3, 9, 5, 7]
        flattened = [length for batch in model.batch_lengths for length in batch]
        assert flattened == sorted(flattened)
        assert flattened == [3, 5, 7, 9, 11]
        # batch_size=2 over 5 texts -> 3 batches
        assert [len(batch) for batch in model.batch_lengths] == [2, 2, 1]

    def test_speaker_suffixed_naming(self, monkeypatch, tmp_path):
        texts = ["ccc", "a", "bb"]
        saved, model = self._run(monkeypatch, tmp_path, texts, spk_id=3)

        assert sorted(name for name, _ in saved) == [f"utterance_{i:03d}_speaker_003" for i in range(len(texts))]
        # spk tensor is expanded to the batch size for each batch
        assert [spks.shape[0] for spks in model.batch_spks] == [2, 1]

        for name, output in saved:
            orig_idx = int(name.split("_")[1])
            expected_id = cleaned_text_to_sequence(texts[orig_idx], language="en")[0]
            assert output["mel"].flatten()[0].item() == pytest.approx(float(expected_id))


# ---------------------------------------------------------------------------
# 4. batched_collate_fn
# ---------------------------------------------------------------------------


class TestBatchedCollateFn:
    """Test padding and order preservation in the batch collate function."""

    def test_pads_right_and_preserves_order(self):
        lengths = [3, 7, 5]
        batch = []
        for length in lengths:
            x = torch.arange(1, length + 1, dtype=torch.long)[None]  # (1, L), non-zero ids
            batch.append({"x": x, "x_lengths": torch.tensor([length], dtype=torch.long)})

        out = batched_collate_fn(batch)

        assert out["x"].shape == (3, 7)
        assert out["x"].dtype == torch.long
        assert torch.equal(out["x_lengths"], torch.tensor([3, 7, 5]))
        # Content is preserved and padding is zeros on the right
        assert torch.equal(out["x"][0, :3], torch.arange(1, 4))
        assert (out["x"][0, 3:] == 0).all()
        assert torch.equal(out["x"][1], torch.arange(1, 8))
        assert torch.equal(out["x"][2, :5], torch.arange(1, 6))
        assert (out["x"][2, 5:] == 0).all()


# ---------------------------------------------------------------------------
# 5. validate_args — custom checkpoint branch
# ---------------------------------------------------------------------------


def _default_args(**overrides):
    """Namespace mirroring the argparse defaults in cli()."""
    defaults = {
        "model": "matcha_ljspeech",
        "checkpoint_path": None,
        "vocoder": None,
        "text": None,
        "file": None,
        "spk": None,
        "temperature": 0.667,
        "speaking_rate": None,
        "steps": 5,
        "cpu": False,
        "denoiser_strength": 0.00025,
        "output_folder": ".",
        "batched": False,
        "batch_size": 32,
        "language": None,
        "cleaners": None,
        "compile": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestValidateArgs:
    """Test validate_args, focusing on the custom-checkpoint branch."""

    def test_custom_checkpoint_defaults_speaking_rate(self):
        args = _default_args(text="hello", checkpoint_path="model.ckpt", vocoder="hifigan_univ_v1")
        args = validate_args(args)
        assert args.speaking_rate == 1.0

    def test_custom_checkpoint_warns_on_non_universal_vocoder(self):
        args = _default_args(text="hello", checkpoint_path="model.ckpt", vocoder="hifigan_T2_v1")
        with pytest.warns(UserWarning, match="custom model checkpoint"):
            validate_args(args)

    def test_batched_requires_positive_batch_size(self):
        args = _default_args(
            text="hello", checkpoint_path="model.ckpt", vocoder="hifigan_univ_v1", batched=True, batch_size=0
        )
        with pytest.raises(AssertionError, match="Batch size"):
            validate_args(args)

    def test_negative_temperature_rejected(self):
        args = _default_args(text="hello", checkpoint_path="model.ckpt", vocoder="hifigan_univ_v1", temperature=-1)
        with pytest.raises(AssertionError, match="temperature"):
            validate_args(args)

    def test_requires_text_or_file(self):
        args = _default_args(checkpoint_path="model.ckpt", vocoder="hifigan_univ_v1")
        with pytest.raises(AssertionError):
            validate_args(args)


# ---------------------------------------------------------------------------
# 6. get_texts — UTF-8 file reading
# ---------------------------------------------------------------------------


class TestGetTexts:
    """Test text-source resolution (inline text vs UTF-8 file)."""

    def test_reads_utf8_file_undamaged(self, tmp_path):
        lines = ["こんにちは、世界。", "ありがとうございました。"]
        text_file = tmp_path / "texts.txt"
        text_file.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

        args = argparse.Namespace(text=None, file=str(text_file))
        assert get_texts(args) == lines

    def test_strips_newlines_and_drops_empty_lines(self, tmp_path):
        """Regression: raw readlines() output reached batched_synthesis, so a
        blank line became a length-1 all-blank sequence synthesised as noise."""
        text_file = tmp_path / "texts.txt"
        text_file.write_text("hello world\n\n  \nsecond line\n\n", encoding="utf-8")

        args = argparse.Namespace(text=None, file=str(text_file))
        assert get_texts(args) == ["hello world", "second line"]

    def test_inline_text_takes_priority(self):
        args = argparse.Namespace(text="hello", file=None)
        assert get_texts(args) == ["hello"]


# ---------------------------------------------------------------------------
# 7. assert_required_models_available — custom checkpoint skips download
# ---------------------------------------------------------------------------


class TestAssertRequiredModelsAvailable:
    """Test that a custom checkpoint skips the pretrained-model download."""

    def _run(self, monkeypatch, tmp_path, args):
        downloads = []
        monkeypatch.setattr(cli_module, "get_user_data_dir", lambda: tmp_path)
        monkeypatch.setattr(
            cli_module, "assert_model_downloaded", lambda path, url: downloads.append((Path(path), url))
        )
        paths = cli_module.assert_required_models_available(args)
        return paths, downloads

    def test_custom_checkpoint_skips_pretrained_download(self, monkeypatch, tmp_path):
        """Regression: the inverted hasattr condition asserted the pretrained
        matcha download even when --checkpoint_path was provided."""
        args = _default_args(checkpoint_path="model.ckpt", vocoder="hifigan_univ_v1")
        paths, downloads = self._run(monkeypatch, tmp_path, args)

        # Only the vocoder is checked/downloaded, never the pretrained matcha
        assert [p.name for p, _ in downloads] == ["hifigan_univ_v1"]
        assert downloads[0][1] == cli_module.VOCODER_URLS["hifigan_univ_v1"]
        assert paths["matcha"] == "model.ckpt"
        assert paths["vocoder"] == tmp_path / "hifigan_univ_v1"

    def test_no_checkpoint_downloads_pretrained_model(self, monkeypatch, tmp_path):
        args = _default_args(vocoder="hifigan_T2_v1")
        paths, downloads = self._run(monkeypatch, tmp_path, args)

        assert [p.name for p, _ in downloads] == ["matcha_ljspeech.ckpt", "hifigan_T2_v1"]
        assert downloads[0][1] == cli_module.MATCHA_URLS["matcha_ljspeech"]
        assert paths["matcha"] == tmp_path / "matcha_ljspeech.ckpt"

    def test_missing_checkpoint_attribute_treated_as_none(self, monkeypatch, tmp_path):
        """Regression: args without a checkpoint_path attribute used to raise
        AttributeError in the condition's second operand."""
        args = argparse.Namespace(model="matcha_ljspeech", vocoder="hifigan_univ_v1")
        paths, downloads = self._run(monkeypatch, tmp_path, args)

        assert [p.name for p, _ in downloads] == ["matcha_ljspeech.ckpt", "hifigan_univ_v1"]
        assert paths["matcha"] == tmp_path / "matcha_ljspeech.ckpt"
