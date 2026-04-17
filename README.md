# AutoValuate — Car Price Prediction with Machine Learning

A production-ready web application that predicts Indian car prices using a trained Gradient Boosting Regression model. Deployed on Netlify with zero backend costs.

---

## Live Demo

Deploy to Netlify in one click and share the link with anyone for free.

---

## What This Does

You enter a car's specifications — brand, engine power, fuel type, body type, dimensions, and more — and the app instantly returns an estimated ex-showroom price with a confidence range. The model was trained on 1,276 real Indian automobile listings and achieves an R² score of 0.981.

---

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.9810 (98.1%) |
| Mean Absolute Error | Rs. 7,66,859 |
| Root Mean Square Error | Rs. 38,51,098 |
| 5-Fold CV R² | 0.96 ± 0.02 |
| Training Samples | 1,020 |
| Test Samples | 256 |

---

## Features Used for Prediction

The model uses 18 engineered features across two categories:

Numeric features: Power (PS), Torque (Nm), Displacement (cc), Length, Width, Wheelbase, Fuel Tank capacity, Kerb Weight, Ground Clearance, ARAI Mileage, Brand Goodwill Tier, Seating Capacity.

Categorical features (label-encoded): Make, Fuel Type, Body Type, Drivetrain, Transmission type.

---

## Tech Stack

**Frontend**: Plain HTML, CSS, JavaScript — no frameworks needed. Fonts from Google Fonts (Playfair Display + DM Sans). Hosted as a static site on Netlify.

**Machine Learning**: Python with Pandas for data wrangling, Scikit-learn for model training (GradientBoostingRegressor), NumPy for numerical operations, Matplotlib for evaluation plots.

**Deployment**: Netlify static hosting with optional serverless functions for server-side prediction.

---

## Project Structure

```
car-price-predictor/
├── public/
│   └── index.html          Main web app
├── netlify/
│   └── functions/
│       └── predict.py      Serverless prediction function
├── data/
│   ├── cars.csv            Training dataset (1,276 Indian cars)
│   ├── feature_medians.json  Default values for missing inputs
│   ├── label_encoders.json   Encoded category mappings
│   └── model_stats.json      Saved evaluation metrics
├── train_model.py          Full training script with visualizations
├── requirements.txt        Python dependencies
├── package.json            Node metadata for Netlify CLI
├── netlify.toml            Netlify deployment configuration
└── README.md               You are here
```

---

## Getting Started

### Run the Web App Locally

No installation needed. Just open `public/index.html` in your browser.

### Retrain the Model

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python train_model.py
```

This will print evaluation metrics, save `model.pkl`, and generate a `public/model_analysis.png` with four plots: predicted vs actual, residuals, feature importances, and price distribution.

### Deploy to Netlify

Option 1 — Drag and drop: Go to app.netlify.com, click "Add new site", and drag the entire project folder.

Option 2 — CLI:

```bash
npm install -g netlify-cli
netlify login
netlify deploy --prod --dir=public
```

---

## Dataset Details

The dataset contains 1,276 Indian car listings sourced from manufacturer specifications. It covers 40 brands from Bajaj and Datsun at the budget end to Ferrari, Lamborghini, and Bugatti at the ultra-luxury tier. The target variable (`Ex-Showroom_Price`) ranges from Rs. 2.36 lakh to Rs. 2.12 crore.

### Data Preprocessing Steps

- Price strings like `Rs. 2,92,667` were parsed to numeric using regex
- Power ratings like `38PS@5500rpm` were extracted to plain numeric PS values
- Torque like `51Nm@4000rpm` was extracted in Nm
- All dimension columns (mm) and weight columns (kg) were similarly parsed
- Missing values were imputed using column medians
- A brand goodwill tier (1–10 scale) was manually assigned based on market positioning
- The price target was log-transformed before training to reduce skewness

---

## Why Gradient Boosting

Random Forests were also evaluated. Gradient Boosting outperformed with a higher R² and lower MAE because:

- It builds trees sequentially, correcting prior errors
- It handles the wide price range (budget to ultra-luxury) better due to log-target training
- It captures non-linear interactions between brand tier and engine power that additive models miss

---

## Interpreting Predictions

The predicted price is an ex-showroom estimate for the Indian market. A confidence range of roughly plus or minus 12–15% is displayed alongside the point estimate. The model is most accurate for mainstream cars in the Rs. 4–40 lakh range. For ultra-luxury vehicles (Lamborghini, Ferrari, Bugatti), the confidence interval widens because fewer training samples exist in that range.

---

## License

MIT License. Free to use, modify, and deploy.

---

## Author

Built with love by **Muhammad Irtaza**

If this project was useful, give it a star on GitHub.
