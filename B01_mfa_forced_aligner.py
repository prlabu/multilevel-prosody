
import re
import random
import subprocess
from pathlib import Path
import shutil 
import string


# --- paths ---
wav_dir = Path("/Users/ly546/Documents/data/mngu0_s1_wav_16kHz_1.1.0")
utt_dir = Path("/Users/ly546/Documents/data/mngu0_s1_lab_1.1.1")
out_dir = Path("/Users/ly546/Documents/data/mngu0_s1_TextGrid") #  Path("/Users/ly546/Documents/MFA/output")
acoustic_model = "/Users/ly546/Documents/MFA/pretrained_models/acoustic/english_mfa.zip"
dictionary = "/Users/ly546/Documents/MFA/pretrained_models/dictionary/english_uk_mfa.dict"

# --- parameters ---
n_samples = 1000  # number of random test files to align

# --- extract sentences and write .txt files ---
utt_files = list(utt_dir.glob("*.utt"))
print("Found", len(utt_files), ".utt files in", utt_dir)

# --- pick random subset for testing ---
# test_utts = random.sample(utt_files, k=min(n_samples, len(utt_files)))
# ----- or run on all files:  
test_utts = utt_files 


for utt_path in utt_dir.glob("*.utt"):
    text = utt_path.read_text(errors="ignore")

    # Extract the sentence inside the iform field
    match = re.search(r'iform\s+"?\\?"?([^"]+)"?', text)
    if not match:
        print(f"⚠️ Could not parse {utt_path.name}")
        continue

    sentence = match.group(1)
    # Remove escaped quotes and punctuation, normalize spaces
    sentence = sentence.replace('\\"', '').strip()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = re.sub(r'\s+', ' ', sentence)

    # Split into words, uppercase each
    words = [w.upper() for w in sentence.split() if w.strip()]
    mfa_formatted = "\n".join(words) + "\n"

    txt_path = utt_path.with_suffix(".txt")
    txt_path.write_text(mfa_formatted)
    print(f"Wrote {txt_path.name}: {len(words)} words")





# --- run MFA align_one ---
for utt_path in test_utts:
    base = utt_path.stem
    wav_path = wav_dir / f"{base}.wav"
    txt_path = utt_path.with_suffix(".txt")

    if not wav_path.exists() or not txt_path.exists():
        print(f"Skipping {base}: missing wav or txt")
        continue

    if (out_dir / f"{base}.TextGrid").exists():
        print(f"Skipping {base}: already aligned")
        continue

    print(f"\nAligning {base} ...")
    cmd = [
        "mfa", "align_one",
        str(wav_path),
        str(txt_path),
        dictionary,
        acoustic_model,
        str(out_dir)
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)

    # shutil.copyfile(wav_path, out_dir / f"{base}.wav")

print("\n✅ Done! Check .TextGrid files in", out_dir)
print("Open them in Praat to visually inspect the alignments.")
