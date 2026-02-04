# Removes lines with special characters from the combined_human_dataset.csv human emails. When used with --dry-run just prints stats, doesn't remove

import argparse
import csv
import sys
import tempfile
import unicodedata
from pathlib import Path
import shutil
import csv, sys
csv.field_size_limit(sys.maxsize)

def has_special(text):
    if text is None:
        return False
    for ch in text:
        cat = unicodedata.category(ch)
        # treat control/unassigned/private-use/etc (categories starting with 'C')
        # and miscellaneous symbols (emoji etc, category 'So') as "special"
        if cat[0] == "C" or cat == "So":
            return True
    return False

def process(path: Path, dry_run: bool):
    removed = 0
    kept = 0
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "body" not in reader.fieldnames:
            print('Error: "body" column not found in CSV', file=sys.stderr)
            sys.exit(2)
        header = reader.fieldnames
        for r in reader:
            body = r.get("body", "")
            if has_special(body):
                removed += 1
            else:
                kept += 1
                rows.append(r)

    if dry_run:
        print(f"would remove: {removed}")
        print(f"would keep:   {kept}")
        return

    # write filtered CSV back to file (atomic replace)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False) as tf:
        writer = csv.DictWriter(tf, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
        tmpname = tf.name
    shutil.move(tmpname, str(path))
    print(f"removed: {removed}")
    print(f"kept:    {kept}")

def parse_args():
    p = argparse.ArgumentParser(description="Filter rows whose body contains special characters.")
    p.add_argument("--dry-run", action="store_true", help="only print counts, do not modify the file")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(Path("data/combined_human_dataset.csv"), args.dry_run)