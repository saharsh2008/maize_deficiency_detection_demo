# Maize Nutrient Deficiency Detection

Detect nutrient deficiencies in maize crops using deep learning and provides eco-friendly remedies.

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow scikit-learn pillow matplotlib flask opencv-python numpy --break-system-packages
```

### 2. Prepare Dataset

Place your dataset in this structure:
```
DATASET_AGRINET/DATASET_AGRINET/TRAINING/MAIZE/
├── HEALTHY/
├── NITROGEN/
├── PHOSPHOROUS/
├── POTASSIUM/
└── ZINC/
```

### 3. Train the Model (Required First)

```bash
python train_maize_deficiency_model.py
```

This creates:
- `maize_deficiency_model_final.keras`
- `maize_deficiency_classes.txt`
- Training curves and processed dataset

### 4. Run the Web App

```bash
python app.py
```

Access at: `http://localhost:5000`

## Files

- `train_maize_deficiency_model.py` - Model training script
- `app.py` - Flask web application
- `templates/index.html` - Web interface
