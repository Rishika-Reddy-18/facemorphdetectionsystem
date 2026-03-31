# Face Morph Detection System

A web-based application to detect face morphing in images using Deep Learning. This project uses a trained TensorFlow/Keras model integrated with a Flask backend to classify whether an image is **genuine** or **morphed**, along with a visual heatmap.

---

## 🚀 Features

* Upload an image for analysis
* Detect whether the image is **Real** or **Morphed**
* Display prediction confidence
* Generate and display **heatmap visualization**
* User-friendly web interface
* (Optional) Login/Admin system

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Flask (Python)
* **Machine Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV, NumPy

---

## 📂 Project Structure

```
project-folder/
│
├── static/
│   ├── uploads/
│   ├── css/
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── login.html
│   └── admin.html
│
├── model/
│   └── morph_model.h5
│
├── utils/
│   ├── preprocess.py
│   ├── heatmap.py
│   └── chatbot.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create virtual environment (recommended)

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 📸 Usage

1. Open the web app
2. Upload an image
3. Click **Predict**
4. View:

   * Prediction (Real / Morph)
   * Confidence score
   * Heatmap output

---

## 📊 Model Information

* Model: CNN-based classifier
* Framework: TensorFlow/Keras
* Input: Face image
* Output: Binary classification (Real / Morph)

---

## ⚠️ Notes

* Ensure the model file (`morph_model.h5`) is placed inside the `model/` folder
* Uploaded images are stored temporarily in `static/uploads/`
* Large image sizes may affect performance

---

## 🔐 Future Improvements

* Improve model accuracy
* Add real-time webcam detection
* Enhance UI/UX
* Deploy to cloud (AWS / Heroku)

---

## 🤝 Contributing

Feel free to fork this repository and contribute by submitting pull requests.

---

## 📜 License

This project is for educational purposes.

---

## 👩‍💻 Author

Your Name
GitHub: https://github.com/Rishika-Reddy-18
