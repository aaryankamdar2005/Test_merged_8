import csv
from backend.model_accuracy import calculate_model_accuracies

def main():
    rows = []
    with open('../Final_Dataset.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # We need to ensure numeric types for columns that are numeric, though _notebook_dataframe handles some,
            # wait, _notebook_dataframe doesn't cast types. Let's cast them.
            processed_row = {}
            for k, v in row.items():
                try:
                    processed_row[k] = float(v) if '.' in v else int(v)
                except ValueError:
                    processed_row[k] = v
            rows.append(processed_row)
    
    metrics = calculate_model_accuracies(rows)
    for model_name, acc in metrics.items():
        print(f"{model_name}: {acc}")

if __name__ == '__main__':
    main()
