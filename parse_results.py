import re
import pandas as pd
from pathlib import Path

log_path = Path('evaluate_results/evaluate.log')
lines = log_path.read_text().split('\n')

results = []
current_name = None

for line in lines:
    if 'Evaluating: ' in line:
        current_name = line.split('Evaluating: ')[1].strip()
    elif '--> WER:' in line and current_name:
        # e.g., '  --> WER: 15.20% | CER: 8.40% | RTF: 0.150'
        match = re.search(r'WER:\s*([\d.]+)%\s*\|\s*CER:\s*([\d.]+)%\s*\|\s*RTF:\s*([\d.]+)', line)
        if match:
            wer, cer, rtf = match.groups()
            results.append({
                'Trailer': current_name[:40],
                'WER (%)': float(wer),
                'CER (%)': float(cer),
                'RTF': float(rtf)
            })
            current_name = None

if results:
    df = pd.DataFrame(results)
    
    # Save the dataframe for the user
    df.to_csv('evaluate_results/parsed_summary.csv', index=False)
    
    print('\n' + '='*90)
    print(f' EVALUATION SUMMARY (N={len(df)} successfully completed trailers)')
    print('='*90)
    print(df.to_string(index=False))
    print('-'*90)
    print(f"Mean WER: {df['WER (%)'].mean():.2f}%")
    print(f"Mean CER: {df['CER (%)'].mean():.2f}%")
    print(f"Mean RTF: {df['RTF'].mean():.3f}")
    print('='*90 + '\n')
else:
    print('No results found in log.')
