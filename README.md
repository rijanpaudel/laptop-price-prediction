# 💻 Laptop Price Prediction App (Flask + ML)

This is a web app that predicts the price of a laptop based on its features using a machine learning model trained on real-world data.

## 🧠 Features

* Trained ML model using `scikit-learn`
* Flask backend API with CORS support
* HTML/JS frontend form for predictions
* Preprocessed features like brand, processor, RAM, etc.
* Log-transformed predictions reversed for real-world prices

---

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/rijanpaudel/laptop-price-prediction.git
cd laptop-price-prediction
```

---

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

---

### 3. Install Required Libraries

Make sure `pip` is up-to-date, then install:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ❗ If `pandas` gives errors, try:
>
> ```bash
> pip install pandas --no-cache-dir
> ```

---

### 4. Run the Backend Server

```bash
cd backend
python3 app.py
```

The Flask server will run at:
🌐 `http://localhost:4000`

---

### 5. Open the Frontend

* Go to the `frontend` folder (or where `prediction.html` is located).
* Open `prediction.html` using your browser:

  👉 Right-click → Open With → Chrome / Firefox

---

## 📦 Project Structure

```
laptop-price-prediction/
│
├── backend/
│   ├── app.py                  # Flask backend
│   ├── laptop_price_model.pkl  # Trained ML model
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   └── prediction.html         # HTML frontend form
│
└── README.md
```

---

## 🔄 API Info

**POST /predict**
Sends laptop specs in JSON format and returns predicted price.

### Request Example

```json
{
  "brand": "dell",
  "processor": "i5",
  "ram": "8",
  "storage": "512",
  "gpu": "integrated",
  "screenSize": "15.6"
}
```

### Response Example

```json
{
  "estimatedPrice": 72119.58
}
```

---

## 🧪 Trained On

* Dataset: Cleaned laptop specifications dataset from Kaggle
* Model: Linear Regression / Random Forest
* Preprocessing: Label encoding, ppi calculation, log transformation

---

## 🧑‍💻 Author

**Rijan Paudel**
📧 [rijanpaudel](https://github.com/rijanpaudel)
📍 Nepal

---

## ✅ TODO (Optional)

* [ ] Add dropdowns to frontend
* [ ] Improve error handling
* [ ] Deploy on Render/Heroku

---

## 📜 License

This project is open-source and free to use.
