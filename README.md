# 🏦 Bank Marketing Classifier

A machine learning project that predicts whether a customer will subscribe to a term deposit based on their demographic and banking information. Built with Kedro for data science workflows and Streamlit for interactive predictions.

## 🚀 Features

- **Automated ML Pipeline**: Uses AutoGluon for state-of-the-art automated machine learning
- **Interactive Web App**: Beautiful Streamlit interface for real-time predictions
- **Fast Training**: Optimized for quick model development (30 seconds to 2 minutes)
- **Production Ready**: Complete data processing, training, and validation pipeline
- **Performance Reports**: Automatic generation of confusion matrices and classification reports

## 📊 Model Performance

- **Training Accuracy**: ~91%
- **Validation Accuracy**: ~88%
- **Inference Speed**: <0.2 seconds
- **Model Type**: AutoGluon TabularPredictor with ensemble methods

## 🛠️ Tech Stack

- **Framework**: Kedro (data science pipeline)
- **ML Library**: AutoGluon (automated machine learning)
- **Web App**: Streamlit (interactive interface)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure

```
asi-bank-project/
├── app.py                          # Streamlit web application
├── quick_train.py                  # Fast training script
├── requirements.txt                # Python dependencies
├── conf/                          # Configuration files
│   ├── base/parameters.yml        # Production settings
│   └── local/parameters.yml       # Fast development settings
├── data/                          # Data storage
│   ├── 01_raw/                    # Raw data
│   ├── 02_intermediate/           # Processed data
│   ├── 06_models/                 # Trained models
│   └── 08_reporting/              # Performance reports
└── src/asi_projekt/               # Source code
    ├── pipelines/                 # ML pipelines
    │   ├── data_processing/       # Data preprocessing
    │   ├── model_training/        # Model training
    │   └── model_validation/      # Model evaluation
    └── settings.py                # Project settings
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Fast Training (Recommended for Development)

```bash
python quick_train.py
```

### 3. Run the Web App

```bash
streamlit run app.py
```

### 4. Full Pipeline (Production)

```bash
kedro run
```

## 📈 Training Options

### Option 1: Quick Training (30 seconds)
```bash
python quick_train.py
```
- **Best for**: Development and testing
- **Quality**: Good for iterations
- **Preset**: `optimize_for_deployment`

### Option 2: Fast Kedro Pipeline (30 seconds)
```bash
kedro run --pipeline=train --env local
```
- **Best for**: Development with Kedro
- **Quality**: Good for iterations
- **Preset**: `optimize_for_deployment`

### Option 3: Balanced Pipeline (2 minutes)
```bash
kedro run --pipeline=train
```
- **Best for**: Production
- **Quality**: Better model performance
- **Preset**: `medium_quality`

### Option 4: Full Pipeline (Complete workflow)
```bash
kedro run
```
- **Best for**: End-to-end processing
- **Includes**: Data processing → Training → Validation → Reports

## 🎯 Available Pipelines

| Pipeline | Command | Description |
|----------|---------|-------------|
| `dp` | `kedro run --pipeline=dp` | Data preprocessing |
| `train` | `kedro run --pipeline=train` | Model training |
| `mv` | `kedro run --pipeline=mv` | Model validation |
| `full` | `kedro run --pipeline=full` | Complete workflow |

## 📊 Data Features

The model uses the following customer features:

- **Demographic**: Age, Job, Marital Status, Education
- **Financial**: Balance, Default Credit, Housing Loan, Personal Loan
- **Campaign**: Contact Type, Month, Day, Duration, Campaign Contacts
- **Historical**: Days Since Last Contact, Previous Contacts, Previous Outcome

## 🎨 Web Application

The Streamlit app provides:

- **Interactive Form**: Input customer details
- **Real-time Predictions**: Instant results with confidence scores
- **Performance Metrics**: Model accuracy and inference speed
- **Beautiful UI**: Modern, responsive design

## ⚡ Performance Optimizations

- **Fast Training**: Reduced from 10 minutes to 30 seconds
- **Optimized Presets**: Uses `optimize_for_deployment` for speed
- **Resource Management**: Limited CPU usage to prevent system slowdown
- **Model Warm-up**: Faster inference on first prediction

## 🔧 Configuration

### Fast Development (`conf/local/parameters.yml`)
```yaml
time_limit: 30
presets: optimize_for_deployment
epochs: 20
```

### Production (`conf/base/parameters.yml`)
```yaml
time_limit: 120
presets: medium_quality
epochs: 50
```

## 📋 Requirements

- Python 3.9+
- 8GB+ RAM recommended
- SSD storage for faster I/O

## 🐛 Troubleshooting

### Common Issues

1. **Slow Training**: Use `python quick_train.py` for fastest training
2. **Missing Columns**: Ensure all 16 features are provided in the web app
3. **Model Loading Errors**: Retrain the model with current AutoGluon version

### Performance Tips

- Close other applications during training
- Use SSD storage for faster I/O
- Ensure sufficient RAM (8GB+ recommended)

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the performance tips
3. Open an issue on GitHub

---

**Happy Predicting! 🎯✨**