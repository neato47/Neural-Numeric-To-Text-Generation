import csv
from pathlib import Path

from reporter.core.train import RunResult


def export_results_to_csv(dest_dir: Path, result: RunResult, sax_bool: bool, sax_bool2: bool, full_data: bool, learn_templates: bool, cross_entropy: bool, mse: bool) -> None:

    header = ['summary_id',
              'gold tokens',
              'pred tokens']
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_file = 'reporter'
    if sax_bool:
        csv_file = 'reporter_sax'
    elif sax_bool2:
        csv_file = 'reporter_sax2'
        
    if full_data:
        csv_file += "_fd"
    if learn_templates:
        csv_file += "_td"
    if cross_entropy:
        csv_file += "_ce"
    elif mse:
        csv_file += "_mse"
        
    csv_file += ".csv"
    output_file = dest_dir / Path(csv_file)

    with output_file.open(mode='w') as w:
        writer = csv.writer(w, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for (summary_id, gold_sents, pred_sents) in \
                zip(result.summary_ids,
                    result.gold_sents,
                    result.pred_sents):
            writer.writerow([summary_id,
                             '|'.join(gold_sents),
                             '|'.join(pred_sents)])
