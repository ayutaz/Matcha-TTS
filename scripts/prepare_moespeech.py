"""Prepare the MoeSpeech-plus corpus for MoeSpeech->Tsukuyomi pretraining.

MoeSpeech (ayousanz/moe-speech-plus, gated) ships one zip per speaker, each with
``data/<spk>/wav/<spk>_NNN.wav`` (44.1 kHz mono) plus a JSON per utterance
(``anime_whisper_transcription`` / ``parakeet_jp_transcription`` / ``speechMOS`` /
emotion...). We use ``anime_whisper_transcription`` (chosen after comparing both ASR
fields: parakeet had 5% garbled chars that break g2p; anime_whisper had 0% and natural
punctuation for prosody). BGM/SFX are absent (voice-only). speechMOS is not used to
filter (it under-scores game/anime voice — the UTMOS reversal).

Four subcommands (all local, $0):
  prepare    download zips -> sanitize transcript -> trim silence -> resample 22050
             -> per-speaker wav + manifest.jsonl (+ speakers.json)
  select     pick a subset (keeps GLOBAL speaker ids 0..472 so subset->full resume works)
  stats      two-pass raw log-mel mean/std at fmax=11025 (unnormalized)
  precompute mel(fmax=11025) + normalize + .pt named ``{spk}_{stem}.pt`` (MAS, no durations)

The precompute .pt schema matches scripts/precompute_dataset.py exactly so
PrecomputedTextMelDataModule reads it unchanged.
"""

import argparse
import contextlib
import io
import json
import os
import random
import re
import unicodedata
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import normalize
from matcha.utils.utils import intersperse

# Mel params — identical to precompute_dataset.py except F_MAX=11025 (Nyquist at 22050 Hz).
N_FFT = 1024
N_MELS = 80
SAMPLE_RATE = 22050
HOP_LENGTH = 256
WIN_LENGTH = 1024
F_MIN = 0.0
F_MAX = 11025

REPO_ID = "ayousanz/moe-speech-plus"
TRANSCRIPT_FIELD = "anime_whisper_transcription"

# --------------------------------------------------------------------------- #
# Transcript sanitization (D7) — triple defense so g2p never crashes the run.
# --------------------------------------------------------------------------- #
_HAS_JP = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")  # hiragana / katakana / kanji
_ANNOT = re.compile(r"[（(【\[「『][^）)】\]」』]*[）)】\]」』]|[♪♩♫♬※→←↑↓…‥〜~＿_]+")


def clean_transcript(t):
    """NFKC-normalize, strip annotations/decorations, drop whitespace."""
    if not t:
        return None
    t = unicodedata.normalize("NFKC", t)
    t = _ANNOT.sub("", t)
    t = re.sub(r"\s+", "", t)
    return t.strip() or None


def accept_transcript(t, dur, min_c=2, max_c=140, min_d=0.4, max_d=14.0):
    """Reject empty / non-Japanese / too-short-long text or out-of-range duration."""
    if not t:
        return False
    if not _HAS_JP.search(t):
        return False
    if not (min_c <= len(t) <= max_c):
        return False
    return dur is None or (min_d <= float(dur) <= max_d)


def g2p_ok(t):
    """Final defense: does japanese_cleaners g2p produce a non-empty sequence?"""
    try:
        seq, _ = text_to_sequence(t, ["japanese_cleaners"], language="ja")
    except Exception:  # noqa: BLE001 - any g2p failure => drop this utterance, don't crash
        return False
    return len(seq) > 0


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def _trim_silence(wav_1d, sr, top_db=30, margin_ms=50):
    """Energy-based leading/trailing silence trim (mirrors prepare_jvs.trim_silence)."""
    if wav_1d.numel() == 0:
        return wav_1d
    frame = int(sr * 0.01)
    n = wav_1d.numel() // frame
    if n == 0:
        return wav_1d
    frames = wav_1d[: n * frame].reshape(n, frame)
    rms = (frames**2).mean(dim=1).sqrt()
    peak = rms.max()
    if peak <= 0:
        return wav_1d
    thr = peak * (10 ** (-top_db / 20))
    above = (rms > thr).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return wav_1d
    margin = int(sr * margin_ms / 1000)
    s = max(0, above[0].item() * frame - margin)
    e = min(wav_1d.numel(), (above[-1].item() + 1) * frame + margin)
    return wav_1d[s:e]


def _process_zip(zip_path, spk_id, wav_root, trim, keep_zip):
    """Extract one speaker zip -> sanitized 22050 Hz wavs + manifest rows.

    Returns (spk_id, rows, n_seen, n_dropped). rows: list of {wav, spk, text}.
    """
    rows = []
    n_seen = n_dropped = 0
    spk_dir = wav_root / f"spk_{spk_id:03d}"
    spk_dir.mkdir(parents=True, exist_ok=True)
    resamplers = {}
    with zipfile.ZipFile(zip_path) as z:
        wavs = [n for n in z.namelist() if n.lower().endswith(".wav")]
        for wname in sorted(wavs):
            n_seen += 1
            stem = Path(wname).stem  # e.g. 18563891_000 (already speaker-prefixed -> globally unique)
            jname = wname[:-4] + ".json"
            if jname not in z.namelist():
                n_dropped += 1
                continue
            meta = json.loads(z.read(jname))
            text = clean_transcript(meta.get(TRANSCRIPT_FIELD))
            if not accept_transcript(text, meta.get("duration")) or not g2p_ok(text):
                n_dropped += 1
                continue
            # decode wav from the zip bytes
            data, sr = sf.read(io.BytesIO(z.read(wname)), dtype="float32")
            wav = torch.from_numpy(data)
            if wav.ndim > 1:
                wav = wav.mean(dim=1)  # downmix to mono
            if sr != SAMPLE_RATE:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = resamplers[sr](wav.unsqueeze(0)).squeeze(0)
            if trim:
                wav = _trim_silence(wav, SAMPLE_RATE)
            if wav.numel() < int(SAMPLE_RATE * 0.2):  # <0.2s after trim -> drop
                n_dropped += 1
                continue
            out_wav = spk_dir / f"{stem}.wav"
            sf.write(str(out_wav), wav.numpy(), SAMPLE_RATE, subtype="PCM_16")
            rows.append({"wav": str(out_wav), "spk": spk_id, "text": text})
    if not keep_zip:
        with contextlib.suppress(OSError):
            os.remove(os.path.realpath(zip_path))
    return spk_id, rows, n_seen, n_dropped


def cmd_prepare(args):
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(REPO_ID, repo_type="dataset")
    zips = sorted(f for f in files if f.endswith(".zip"))
    spk_to_id = {Path(z).stem: i for i, z in enumerate(zips)}  # deterministic global ids 0..472
    if args.max_speakers:
        zips = zips[: args.max_speakers]
    print(f"[prepare] {len(zips)} zip(s) (of {len(spk_to_id)} total speakers) -> {args.work_dir}")

    work = Path(args.work_dir)
    wav_root = work / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)
    (work / "speakers.json").write_text(json.dumps(spk_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = work / "manifest.jsonl"
    seen = dropped = kept = 0
    with open(manifest, "w", encoding="utf-8") as mf, ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {}
        for z in zips:
            spk_id = spk_to_id[Path(z).stem]
            zp = hf_hub_download(REPO_ID, z, repo_type="dataset")
            futures[ex.submit(_process_zip, zp, spk_id, wav_root, not args.no_trim, args.keep_zips)] = z
        for fut in tqdm(as_completed(futures), total=len(futures), desc="prepare", unit="spk"):
            _spk, rows, ns, nd = fut.result()
            seen += ns
            dropped += nd
            for r in rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    drop_pct = 100 * dropped / seen if seen else 0
    print(f"[prepare] kept {kept} / seen {seen} ({dropped} dropped, {drop_pct:.1f}%). manifest -> {manifest}")
    if drop_pct > 5:
        print(f"[prepare] WARNING drop rate {drop_pct:.1f}% > 5% — inspect transcripts / strengthen _ANNOT")


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #
def cmd_select(args):
    rows = _read_jsonl(args.manifest)
    rng = random.Random(args.seed)
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["spk"]].append(r)
    spks = sorted(by_spk)
    if args.max_speakers:
        spks = spks[: args.max_speakers]  # id-ascending, deterministic (GLOBAL ids preserved)
    sel = []
    for s in spks:
        u = by_spk[s]
        rng.shuffle(u)
        if args.max_utts_per_spk:
            u = u[: args.max_utts_per_spk]
        sel += u
    rng.shuffle(sel)
    if args.max_total:
        sel = sel[: args.max_total]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[select] {len(sel)} utts from {len(spks)} speakers (global ids preserved) -> {args.out}")


# --------------------------------------------------------------------------- #
# stats (two-pass, raw log-mel, fmax=11025)
# --------------------------------------------------------------------------- #
def _raw_mel(wav_path):
    d, sr = sf.read(wav_path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    assert sr == SAMPLE_RATE, f"{wav_path}: {sr}!={SAMPLE_RATE}"
    return mel_spectrogram(
        torch.from_numpy(d).unsqueeze(0), N_FFT, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX, center=False
    ).squeeze()  # unnormalized raw log-mel


def _sum_stats(wav_path):
    m = _raw_mel(wav_path)
    return float(m.sum()), float((m**2).sum()), int(m.numel())


def cmd_stats(args):
    rows = _read_jsonl(args.manifest)
    if args.frac < 1.0:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: max(1, int(len(rows) * args.frac))]
    wavs = [r["wav"] for r in rows]
    s = sq = n = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for fut in tqdm(
            as_completed([ex.submit(_sum_stats, w) for w in wavs]), total=len(wavs), desc="stats", unit="utt"
        ):
            ds, dsq, dn = fut.result()
            s += ds
            sq += dsq
            n += dn
    mean = s / n
    std = (sq / n - mean**2) ** 0.5
    out = {"mel_mean": mean, "mel_std": std, "f_max": F_MAX, "n_utts": len(wavs), "frames_x_mels": n}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[stats] mel_mean={mean:.6f} mel_std={std:.6f} (fmax={F_MAX}, {len(wavs)} utts) -> {args.out}")


# --------------------------------------------------------------------------- #
# precompute (mel + normalize + .pt, spk-id-prefixed name, MAS = no durations)
# --------------------------------------------------------------------------- #
def _precompute_one(row, out_dir, mel_mean, mel_std):
    wav = row["wav"]
    spk = int(row["spk"])
    stem = Path(wav).stem
    out = out_dir / f"{spk}_{stem}.pt"  # D3: spk-id prefix avoids the parent.name collision
    d, sr = sf.read(wav, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    assert sr == SAMPLE_RATE, f"{wav}: {sr}!={SAMPLE_RATE}"
    mel = mel_spectrogram(
        torch.from_numpy(d).unsqueeze(0), N_FFT, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX, center=False
    ).squeeze()
    mel = normalize(mel, mel_mean, mel_std)
    try:
        text_norm, cleaned = text_to_sequence(row["text"], ["japanese_cleaners"], language="ja")
    except Exception as e:  # noqa: BLE001 - skip g2p failures (already filtered in prepare, belt-and-suspenders)
        return str(out), f"g2p error: {e}"
    text_norm = torch.IntTensor(intersperse(text_norm, 0))  # add_blank=True
    torch.save({"mel": mel, "text": text_norm, "spk": spk, "cleaned_text": cleaned}, out)
    return str(out), None


def cmd_precompute(args):
    rows = _read_jsonl(args.manifest)
    if args.val_frac > 0:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac))
        splits = {"val": rows[:n_val], "train": rows[n_val:]}
    else:
        splits = {"train": rows}
    for split, split_rows in splits.items():
        out_dir = Path(args.out_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        errors = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_precompute_one, r, out_dir, args.mel_mean, args.mel_std) for r in split_rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"precompute:{split}", unit="utt"):
                _p, err = fut.result()
                if err:
                    errors += 1
                    tqdm.write(f"SKIP {err}")
        print(f"[precompute:{split}] wrote {len(split_rows) - errors} .pt to {out_dir} ({errors} skipped)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare MoeSpeech-plus for pretraining (fmax=11025)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="download zips -> sanitized 22050 Hz wavs + manifest")
    pp.add_argument("--work-dir", default="data/moespeech")
    pp.add_argument("--max-speakers", type=int, default=None, help="process only first N speaker zips (subset)")
    pp.add_argument("--no-trim", action="store_true", help="skip silence trimming")
    pp.add_argument("--keep-zips", action="store_true", help="keep downloaded zips (default: delete to save disk)")
    pp.add_argument("--num-workers", type=int, default=8)
    pp.set_defaults(func=cmd_prepare)

    ps = sub.add_parser("select", help="pick a subset (global speaker ids preserved)")
    ps.add_argument("--manifest", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--max-speakers", type=int, default=None)
    ps.add_argument("--max-utts-per-spk", type=int, default=None)
    ps.add_argument("--max-total", type=int, default=None)
    ps.add_argument("--seed", type=int, default=42)
    ps.set_defaults(func=cmd_select)

    pst = sub.add_parser("stats", help="two-pass raw log-mel mean/std at fmax=11025")
    pst.add_argument("--manifest", required=True)
    pst.add_argument("--out", required=True)
    pst.add_argument("--frac", type=float, default=1.0, help="subsample fraction (full set: 0.1 is plenty)")
    pst.add_argument("--seed", type=int, default=42)
    pst.add_argument("--num-workers", type=int, default=8)
    pst.set_defaults(func=cmd_stats)

    pc = sub.add_parser("precompute", help="mel(fmax=11025) + normalize + .pt (MAS, no durations)")
    pc.add_argument("--manifest", required=True)
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--mel-mean", type=float, required=True)
    pc.add_argument("--mel-std", type=float, required=True)
    pc.add_argument("--val-frac", type=float, default=0.01)
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--num-workers", type=int, default=8)
    pc.set_defaults(func=cmd_precompute)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
