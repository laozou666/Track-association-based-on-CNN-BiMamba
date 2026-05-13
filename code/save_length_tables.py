"""Export length-stratified evaluation as CSV / Markdown / plain text tables."""
import json, os, csv

OUT = '/home/yangcq/track_association/output/tables'
os.makedirs(OUT, exist_ok=True)

with open('/home/yangcq/track_association/output/length_eval/test_length_eval.json') as f:
    results = json.load(f)

splits = [
    ('short',  'Short',  '[21, 79]',   50.5),
    ('medium', 'Medium', '[80, 180]',  115.8),
    ('long',   'Long',   '[181, 350]', 284.0),
]
N_map = {'short': 9836, 'medium': 7239, 'long': 2282}
pos_map = {'short': 7010, 'medium': 4717, 'long': 1417}
neg_map = {'short': 2826, 'medium': 2522, 'long': 865}

models = [
    ('cnn_mamba_tuned', 'CNN-Mamba tuned (Ours)'),
    ('cnn_bilstm',      'CNN-BiLSTM'),
    ('cnn_lstm',        'CNN-LSTM'),
]

# ---------- 1. Wide CSV ----------
wide_csv = os.path.join(OUT, 'length_eval_wide.csv')
with open(wide_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Model',
                'Short_Acc', 'Short_Prec', 'Short_Rec', 'Short_F1',
                'Medium_Acc', 'Medium_Prec', 'Medium_Rec', 'Medium_F1',
                'Long_Acc', 'Long_Prec', 'Long_Rec', 'Long_F1',
                'Overall_Acc_weighted'])
    for key, label in models:
        row = [label]
        total_n, total_correct = 0, 0
        for sk, _, _, _ in splits:
            r = results[key][sk]
            row += [f'{r["acc"]*100:.2f}',
                    f'{r["prec"]*100:.2f}',
                    f'{r["rec"]*100:.2f}',
                    f'{r["f1"]*100:.2f}']
            total_n += r['n']
            total_correct += r['acc'] * r['n']
        overall = total_correct / total_n * 100
        row.append(f'{overall:.2f}')
        w.writerow(row)
print(f'Saved: {wide_csv}')

# ---------- 2. Long CSV ----------
long_csv = os.path.join(OUT, 'length_eval_long.csv')
with open(long_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Model', 'Split', 'Length_Range', 'Mean_Length', 'N',
                'Accuracy_%', 'Precision_%', 'Recall_%', 'F1_%'])
    for key, label in models:
        for sk, sl, rng, mL in splits:
            r = results[key][sk]
            w.writerow([label, sl, rng, f'{mL:.1f}', r['n'],
                        f'{r["acc"]*100:.2f}',
                        f'{r["prec"]*100:.2f}',
                        f'{r["rec"]*100:.2f}',
                        f'{r["f1"]*100:.2f}'])
print(f'Saved: {long_csv}')

# ---------- 3. Markdown ----------
md_path = os.path.join(OUT, 'length_eval.md')
with open(md_path, 'w') as f:
    f.write('# Length-Stratified Test Evaluation\n\n')
    f.write('## Test Set Information\n\n')
    f.write('| Split | Length Range | Mean Length | N | Pos | Neg |\n')
    f.write('|---|---|:---:|:---:|:---:|:---:|\n')
    for sk, sl, rng, mL in splits:
        f.write(f'| {sl} | {rng} | {mL:.1f} | {N_map[sk]:,} | '
                f'{pos_map[sk]:,} | {neg_map[sk]:,} |\n')
    f.write('| **Total** | [21, 350] | 102.4 | **19,357** | 13,144 | 6,213 |\n\n')

    f.write('## Accuracy Summary (%)\n\n')
    f.write('| Model | Short | Medium | Long | Weighted Overall |\n')
    f.write('|---|:---:|:---:|:---:|:---:|\n')
    for key, label in models:
        row = f'| {label} |'
        tot_n, tot_c = 0, 0
        for sk, _, _, _ in splits:
            r = results[key][sk]
            row += f' {r["acc"]*100:.2f} |'
            tot_n += r['n']
            tot_c += r['acc'] * r['n']
        row += f' **{tot_c/tot_n*100:.2f}** |'
        f.write(row + '\n')
    f.write('\n')

    f.write('## Full Metrics per Split\n\n')
    for sk, sl, rng, _ in splits:
        f.write(f'### {sl} Test Set  (length {rng}, N = {N_map[sk]:,})\n\n')
        f.write('| Model | Accuracy | Precision | Recall | F1 |\n')
        f.write('|---|:---:|:---:|:---:|:---:|\n')
        for key, label in models:
            r = results[key][sk]
            f.write(f'| {label} | {r["acc"]*100:.2f}% | {r["prec"]*100:.2f}% | '
                    f'{r["rec"]*100:.2f}% | {r["f1"]*100:.2f}% |\n')
        f.write('\n')

    f.write('## Accuracy Advantage: CNN-Mamba tuned  vs  CNN-BiLSTM\n\n')
    f.write('| Split | CNN-Mamba tuned | CNN-BiLSTM | Gap (pp) |\n')
    f.write('|---|:---:|:---:|:---:|\n')
    for sk, sl, _, _ in splits:
        m = results['cnn_mamba_tuned'][sk]['acc'] * 100
        b = results['cnn_bilstm'][sk]['acc'] * 100
        gap = m - b
        sign = '+' if gap >= 0 else ''
        f.write(f'| {sl} | {m:.2f}% | {b:.2f}% | **{sign}{gap:.2f}** |\n')
print(f'Saved: {md_path}')

# ---------- 4. Plain text ----------
txt_path = os.path.join(OUT, 'length_eval.txt')
with open(txt_path, 'w') as f:
    f.write('Length-Stratified Test Evaluation\n')
    f.write('=' * 72 + '\n\n')
    f.write(f'{"Model":<24}{"Short":>12}{"Medium":>12}{"Long":>12}{"Overall":>12}\n')
    f.write('-' * 72 + '\n')
    for key, label in models:
        tot_n, tot_c = 0, 0
        row = f'{label:<24}'
        for sk, _, _, _ in splits:
            r = results[key][sk]
            row += f'{r["acc"]*100:>10.2f}%  '
            tot_n += r['n']
            tot_c += r['acc'] * r['n']
        row += f'{tot_c/tot_n*100:>10.2f}%'
        f.write(row + '\n')
    f.write('-' * 72 + '\n\n')

    f.write('Gap  =  CNN-Mamba tuned  -  CNN-BiLSTM\n')
    f.write(f'{"Split":<12}{"Mamba":>10}{"BiLSTM":>10}{"Gap (pp)":>14}\n')
    f.write('-' * 46 + '\n')
    for sk, sl, _, _ in splits:
        m = results['cnn_mamba_tuned'][sk]['acc'] * 100
        b = results['cnn_bilstm'][sk]['acc'] * 100
        f.write(f'{sl:<12}{m:>9.2f}%{b:>9.2f}%{m-b:>+13.2f}\n')
    f.write('-' * 46 + '\n\n')
    f.write('N per split:  Short = 9836  Medium = 7239  Long = 2282  Total = 19357\n')
print(f'Saved: {txt_path}')
print('\nAll 4 files saved to:', OUT)
