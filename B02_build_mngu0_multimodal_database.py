import pandas as pd
from pathlib import Path
from tqdm import tqdm
from praatio import textgrid

# --- paths ---
textgrid_dir = Path("/Users/ly546/Documents/data/mngu0_s1_TextGrid")
out_csv = Path("/Users/ly546/Documents/data/mngu0_segment_index.csv")

# --- configuration ---
# which tiers to extract from each TextGrid
tiers_to_extract = ["words", "phones"]   # or whatever your tier names are in the TextGrids

rows = []

for tg_path in tqdm(sorted(textgrid_dir.glob("*.TextGrid"))):
    utterance_id = tg_path.stem  # e.g. mngu0_s1_0401
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
    # print(tg)

    utt_flag = 1
    for tier_name in tiers_to_extract:
        if tier_name not in tg.tierNames:
            print(f"⚠️  {tier_name} tier not found in {utterance_id}")
            continue

        tier = tg.getTier(tier_name)

        if tier_name=="words" and utt_flag==1: 
            idx = 1
            segment_id = f"{utterance_id}_utterance-{idx:03d}"  # e.g. _word-002 or _phone-012
            start = min([start for (start, end, label) in tier.entries])
            end = max([end for (start, end, label) in tier.entries])
            label = ' '.join([label for (start, end, label) in tier.entries])
        
            rows.append(
                dict(
                    utterance_id=utterance_id,
                    segment_tier='UTTERANCE',  
                    segment_id=segment_id,
                    label=label.strip(),
                    start_time=start,
                    end_time=end,
                    duration=end - start,
                    textgrid_path=str(tg_path),
                )
            )
            utt_flag=0 

        for idx, (start, end, label) in enumerate(tier.entries, start=1):
            if not label.strip():  # skip blank intervals
                continue

            segment_id = f"{utterance_id}_{tier_name[:-1]}-{idx:03d}"  # e.g. _word-002 or _phone-012
            rows.append(
                dict(
                    utterance_id=utterance_id,
                    segment_tier=tier_name.upper().rstrip("S"),  # PHONEME or WORD
                    segment_id=segment_id,
                    label=label.strip(),
                    start_time=start,
                    end_time=end,
                    duration=end - start,
                    textgrid_path=str(tg_path),
                )
            )

# --- assemble and save ---
df = pd.DataFrame(rows)
df = df.sort_values(["utterance_id", "start_time"]).reset_index(drop=True)
df.to_csv(out_csv, index=False)
print(f"✅  Wrote {len(df)} segments to {out_csv}")
