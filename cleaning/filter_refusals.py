import pandas as pd
import argparse
import re
import sys

def detect_refusal(text):
    """
    Determines if a text string is likely a refusal based on regex patterns.
    """
    if not isinstance(text, str):
        return False

    # Convert to lowercase for matching
    text_lower = text.lower()

    # Robust regex patterns for common refusal phrases
    patterns = [
        r"^i (can not|cannot|can't) (do|fulfill|provide|answer|assist)",
        r"^(i'm|i am) (sorry|unable to|not able to) ",
        r"^(as an ai|as a language model),? i",
        r"i (decline|refuse) ",
        r"i am not (allowed|permitted) to",
        r"that (is not|isn't) (appropriate|something i can)"
    ]

    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True

    return False

def main():
    parser = argparse.ArgumentParser(description="Filter LLM refusals from data/outputs.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving files.")
    args = parser.parse_args()

    # Hardcoded file path
    filename = "data/outputs.csv"

    try:
        # Load CSV
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Validate columns
    required_cols = {"model", "prompt", "output"}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)

    # Find refusals
    df['is_refusal'] = df['output'].apply(detect_refusal)

    # Create a numeric ID for prompts to handle long text in tables
    df['prompt_id'] = df['prompt'].astype('category').cat.codes + 1

    # --- Print Prompt Key ---
    print("\n" + "="*60)
    print("PROMPT KEY (Numbered Reference)")
    print("="*60)

    # Get unique prompts sorted by ID
    unique_prompts = df[['prompt_id', 'prompt']].drop_duplicates().sort_values('prompt_id')

    for _, row in unique_prompts.iterrows():
        # Truncate long prompts for the display key
        prompt_text = row['prompt']
        if len(prompt_text) > 60:
            display_text = prompt_text[:57] + "..."
        else:
            display_text = prompt_text
        print(f"{row['prompt_id']}: {display_text}")
    print("="*60)

    # --- Stats ---

    # 1. Overall
    total_count = len(df)
    refusal_count = df['is_refusal'].sum()
    pass_count = total_count - refusal_count

    # 2. Stats by model
    stats_model = df.groupby('model')['is_refusal'].agg(
        Total='count',
        Refusals='sum',
        Passes=lambda x: x.count() - x.sum()
    )

    # 3. Stats by prompt (using the ID becaues prompts are long)
    stats_prompt = df.groupby('prompt_id')['is_refusal'].agg(
        Total='count',
        Refusals='sum',
        Passes=lambda x: x.count() - x.sum()
    )

    # 4. Stats by model/prompt (using the ID)
    stats_combined = df.groupby(['model', 'prompt_id'])['is_refusal'].agg(
        Total='count',
        Refusals='sum',
        Passes=lambda x: x.count() - x.sum()
    ).unstack(fill_value=0)

    stats_combined_clean = df.groupby(['model', 'prompt_id'])['is_refusal'].agg(
        Total='count',
        Refusals='sum',
        Passes=lambda x: x.count() - x.sum()
    )

    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total Responses: {total_count}")
    print(f"Total Refusals:  {refusal_count}")
    print(f"Total Passes:    {pass_count}")
    print(f"Refusal Rate:    {(refusal_count/total_count)*100:.2f}%")

    print("\n" + "="*60)
    print("BREAKDOWN BY MODEL")
    print("="*60)
    print(stats_model.to_string())

    print("\n" + "="*60)
    print("BREAKDOWN BY PROMPT ID")
    print("="*60)
    print(stats_prompt.to_string())

    print("\n" + "="*60)
    print("BREAKDOWN BY MODEL & PROMPT ID")
    print("="*60)
    print(stats_combined_clean.to_string())

    # --- File Handling ---
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE")
        print("="*60)
        print(f"Would have taken out (removed): {refusal_count} rows")
        print(f"Would have left in (kept):      {pass_count} rows")
    else:
        print("\n" + "="*60)
        print("SAVING FILES")
        print("="*60)

        # Filter data
        passes_df = df[~df['is_refusal']].drop(columns=['is_refusal', 'prompt_id'])
        refusals_df = df[df['is_refusal']].drop(columns=['is_refusal', 'prompt_id'])

        # Save
        passes_file = "passes_clean.csv"
        refusals_file = "refusals_extracted.csv"

        passes_df.to_csv(passes_file, index=False)
        refusals_df.to_csv(refusals_file, index=False)

        print(f"Saved {len(passes_df)} passing responses to: {passes_file}")
        print(f"Saved {len(refusals_df)} refusal responses to: {refusals_file}")

if __name__ == "__main__":
    main()
