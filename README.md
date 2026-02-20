# 🚗 Car Selling Price Prediction

A machine learning project that predicts the selling price of used cars using regression models, with a Streamlit web app for real-time predictions.

---

## 📁 Project Structure

```
├── Car details v3.csv          # Dataset
├── train.py                    # Model training script
├── app.py                      # Streamlit web app
├── RandomForest_tuned.pkl      # Saved trained model
└── README.md
```

---

## 📦 Requirements

Install dependencies with:

```bash
pip install pandas scikit-learn matplotlib streamlit joblib
```

---

## 🔧 How It Works

### 1. Training (`train.py`)

The training script does the following:

- **Loads** the dataset from `Car details v3.csv`
- **Feature engineering** — extracts numeric values from `mileage`, `engine`, and `max_power` string columns
- **One-hot encodes** categorical columns: `fuel`, `seller_type`, `transmission`, `owner`
- **Drops** `torque` and `name` columns
- **Trains and compares** 4 models:
  - Linear Regression
  - Ridge Regression (with cross-validated alpha)
  - Decision Tree
  - Random Forest ✅ (best performer)
- **Saves** the tuned Random Forest model as `RandomForest_tuned.pkl`

### 2. Web App (`app.py`)

- Built with **Streamlit**
- Loads the saved model
- Lets you enter car features manually or upload a CSV
- Outputs the **predicted selling price in ₹**

---

## 🚀 Running the App

```bash
streamlit run app.py
```

---

## 📊 Models Compared

| Model | Description |
|---|---|
| Linear Regression | Simple baseline |
| Ridge Regression | Regularized linear model |
| Decision Tree | Tree-based, no scaling needed |
| Random Forest | Ensemble of trees, best results |

---

## 🧾 Input Features

| Feature | Description |
|---|---|
| `year` | Manufacturing year |
| `km_driven` | Total kilometers driven |
| `mileage` | Fuel efficiency (kmpl) |
| `engine` | Engine displacement (cc) |
| `max_power` | Max power (bhp) |
| `seats` | Number of seats |
| `fuel_*` | Fuel type (Petrol, Diesel, CNG, LPG) |
| `transmission_*` | Manual or Automatic |
| `seller_type_*` | Individual, Dealer, Trustmark Dealer |
| `owner_*` | First, Second, Third, etc. |

> One-hot encoded columns: only one in each group should be `1`, rest `0`.

---

## 📈 Example Prediction

For a **2018 Maruti Swift** (Petrol, Manual, First Owner, 45,000 km):

```
Predicted Selling Price: ₹ 4,94,714
```

---

## ⚠️ Notes

- Tree-based models (Random Forest, Decision Tree) do **not** need feature scaling
- Linear and Ridge models use a `Pipeline` with `StandardScaler` and `SimpleImputer`
- The `engine` column is kept **numeric** — do not one-hot encode it
