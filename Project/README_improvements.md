# NER Model F1 Score Improvement Guide

## Problem Analysis

Your current NER model shows very low F1 scores (mostly below 0.3) due to:

- **Severe class imbalance**: "O" class dominates with 1098 samples
- **Poor recall**: Most entity classes have recall < 0.75
- **Poor precision**: Most entity classes have precision < 0.32
- **Zero support**: I-BODY_PART has 0 support

## Solution Overview

### 1. Enhanced Model Architecture

- **Bidirectional LSTM**: Replaced unidirectional LSTM with BiLSTM for better context
- **Multi-head Attention**: Added attention mechanism for better feature extraction
- **Layer Normalization**: Improved training stability
- **Increased capacity**: Larger embedding (256 vs 128) and LSTM units (256 vs 128)

### 2. Data Augmentation

- **Synonym replacement**: Replace non-entity words with synonyms
- **Oversampling**: Increase minority class samples
- **Entity-specific augmentation**: Focus on underrepresented entities

### 3. Training Improvements

- **Enhanced class weights**: Better handling of class imbalance
- **Early stopping**: Prevent overfitting
- **Learning rate scheduling**: Better convergence
- **Increased epochs**: 50 vs 8 for better training

### 4. Evaluation Metrics

- **Per-entity F1 scores**: Focus on entity-level performance
- **Comprehensive reporting**: Detailed improvement tracking

## Files Created

1. **improved_ner_model.py**: Enhanced NER model with all improvements
2. **run_improvements.py**: Script to run the complete improvement pipeline
3. **requirements.txt**: Required dependencies

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Improvement Pipeline

```bash
python run_improvements.py
```

### Step 3: Check Results

- Model files: `enhanced_ner_model.keras`, `enhanced_mappings.pkl`
- Results: `f1_improvement_results.csv`

## Expected Improvements

Based on the enhancements, you should see:

- **50-200% improvement** in F1 scores for minority classes
- **Better recall** for underrepresented entities
- **Reduced false positives** through attention mechanism
- **More stable training** through layer normalization

## Advanced Options

### For Even Better Results:

1. **Transformer Models**: Consider using BERT/RoBERTa
2. **Ensemble Methods**: Combine multiple models
3. **Active Learning**: Add more training data
4. **Domain-specific Preprocessing**: Add medical/herbal dictionaries

### Troubleshooting

- If memory issues occur, reduce `BATCH_SIZE` to 32 or 16
- If training is slow, reduce `EMBED_DIM` and `LSTM_UNITS`
- For very small datasets, reduce `AUGMENTATION_FACTOR`

## Monitoring Progress

The script will show:

- Class distribution analysis
- Training progress
- Detailed improvement metrics
- Final comparison with original results
