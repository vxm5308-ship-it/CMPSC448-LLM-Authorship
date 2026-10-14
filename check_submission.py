"""Quick non-destructive check for project submission readiness."""
from pathlib import Path
import csv
p=Path(__file__).resolve().parent
required=['results/metrics_summary.csv','results/baselines.csv','results/dataset_counts.csv','Project_Report.pdf']
for name in required:
    f=p/name
    print(('OK' if f.is_file() and f.stat().st_size else 'MISSING'),name)
metrics=p/'results/metrics_summary.csv'
if metrics.is_file():
    with metrics.open(newline='') as fh:
        rows=list(csv.DictReader(fh))
    found={(r.get('model'),r.get('features')) for r in rows}
    expected={(m,f) for m in ('CNN','BiLSTM') for f in ('input','output','both')}
    print('Primary experiment coverage:',len(found & expected),'/ 6')
else:
    print('Primary experiment coverage: not available')
print('Team names, reflections, GitHub URL and interpretation require manual review.')
