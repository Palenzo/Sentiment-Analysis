# 🎬 IMDB Sentiment Analysis with Machine Learning

An in-depth project analyzing movie reviews from IMDB to classify sentiments as positive or negative using machine learning. This project focuses on text preprocessing, feature engineering, and model evaluation to accurately assess user sentiment based on review text.

---

## 📄 Project Overview

This project leverages a dataset of 50,000 IMDB movie reviews, aiming to:

- Perform text preprocessing (e.g., cleaning, tokenization, stemming).
- Create a feature matrix using Count Vectorization.
- Train multiple machine learning models for sentiment classification.
- Deploy the final model with a frontend interface for user interaction.

---

## 📚 Dataset

We used the IMDB Large Movie Review Dataset, which contains 50,000 reviews labeled as positive or negative. Each review undergoes preprocessing steps to enhance model performance. Additional datasets like Sentiment140 and Twitter US Airline Sentiment were considered for model robustness testing.

**Source:** Stanford Large Movie Review Dataset

---

## 🛠️ Key Features

- **Data Cleaning:** Removing HTML tags, special characters, and stop words.
- **Feature Extraction:** Transforming text into a structured form using Count Vectorization.
- **Model Training:** Implementing and comparing multiple algorithms (e.g., Naive Bayes variants) for best performance.
- **Web Deployment:** Serving the model through a FastAPI backend with a React-based frontend for sentiment analysis on demand.

---

## 🔧 Tech Stack

- **Python:** For data processing and model training
- **Scikit-Learn:** For machine learning
- **FastAPI:** For API development
- **React:** For frontend

---

## 🧩 Project Structure


├── data/ # Dataset and preprocessing scripts 
├── notebooks/ # Jupyter notebooks for data exploration 
├── models/ # Trained model files 
├── api/ # FastAPI backend for model deployment 
├── frontend/ # React frontend for user interaction 
└── README.md # Project documentation


---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Node.js (for the frontend)
- FastAPI and Uvicorn

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis

Install backend dependencies:


pip install -r requirements.txt

Run the FastAPI server:


uvicorn api.app:app --reload
Start the frontend:


cd frontend
npm install
npm start

📝 Model Details and Results
The project explored various classifiers:

Gaussian Naive Bayes
Bernoulli Naive Bayes
Multinomial Naive Bayes (Best Performance)
Model	Accuracy
Gaussian NB	84.5%
Bernoulli NB	88.7%
Multinomial NB	90.3%
🌐 API Endpoints
POST /predict/: Accepts a JSON body with a text review and returns a prediction.
Example:

Request:

{
  "text": "The movie was fantastic! I loved it."
}
Response:

{
  "sentiment": "Positive"
}

🤖 Future Enhancements
Model Fine-Tuning: Use more advanced models like SVM or ensemble methods.
Data Augmentation: Incorporate more review data from sources like Sentiment140.
Live Deployment: Deploy the API and frontend to a cloud provider.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please open an issue to discuss your ideas or submit a pull request.
# Sentiment-Analysis
