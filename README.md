# Z-Anonymity for Process Mining

This project implements z-anonymity for process mining event logs, focusing on directly-follows relations between activities.

## Project Structure

```
z_anonymity_pm/
├── data/
│   ├── input/         # Place your input event logs here
│   └── output/        # Anonymized logs and results will be stored here
├── src/
│   ├── anonymization/ # Z-anonymity implementation
│   ├── evaluation/    # Privacy and utility measurement
│   └── utils/        # Helper functions
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your event log files in the `data/input/` directory
2. Run the main script:
```bash
python src/main.py --log_file your_log.xes --z_value 5 --time_window 3600
```

## Parameters

- `z_value`: The minimum number of cases that should share the same directly-follows relation
- `time_window`: Time frame window in seconds for considering directly-follows relations

## Measurements

The implementation measures both privacy and utility:

### Privacy
- Z-anonymity compliance for directly-follows relations

### Utility
- Preservation of always-follows relations
- Preservation of sometimes-follows relations
- Preservation of never-follows relations

## Results

Results and analysis will be stored in the `data/output/` directory. 