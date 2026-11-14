# traffic-accident-prediction
# 🚦 AI-Based Traffic Accident Prediction System

An intelligent traffic accident prediction system that uses Artificial Intelligence and Machine Learning to predict potential road accident risks based on different traffic and environmental factors. The project aims to improve road safety by identifying high-risk conditions and providing explainable predictions using DeepSHAP.

## 📌 Project Overview

Road accidents are a major concern worldwide. Many accidents occur due to factors such as weather conditions, road characteristics, traffic density, lighting conditions, and other environmental factors.

This project focuses on developing an AI-powered prediction system that analyzes relevant data to estimate accident risk. It also uses **DeepSHAP (SHAP-based Explainable AI)** to explain how different input features influence the model's predictions.

## 🎯 Objectives

- Predict potential traffic accident risks using Machine Learning.
- Analyze important factors contributing to road accidents.
- Provide explainable AI predictions using DeepSHAP.
- Develop a scalable backend using Spring Boot.
- Integrate the trained AI model with the backend.
- Support data-driven road safety analysis and decision-making.

## ✨ Key Features

- **Accident Risk Prediction:** Predicts accident risk based on input parameters.
- **Machine Learning Integration:** Uses trained ML models for prediction.
- **Explainable AI:** Uses DeepSHAP to understand the factors influencing predictions.
- **REST API Integration:** Connects the AI model with the backend through APIs.
- **Backend Architecture:** Uses Spring Boot for backend development.
- **Data Analysis:** Processes accident-related data to identify patterns and risk factors.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Java | Backend programming |
| Spring Boot | REST API and backend development |
| Python | Machine Learning implementation |
| Machine Learning | Accident risk prediction |
| DeepSHAP / SHAP | Model explainability |
| Pandas & NumPy | Data processing |
| Scikit-learn | Machine Learning |
| MySQL | Database management |
| Postman | API testing |
| Git & GitHub | Version control |

## 🏗️ System Architecture

The project follows a modular architecture:

1. **Data Collection:** Collects and prepares traffic accident-related datasets.
2. **Data Preprocessing:** Cleans the dataset and handles missing values, encoding, and feature selection.
3. **Model Training:** Trains a Machine Learning model using relevant accident-related features.
4. **Accident Prediction:** Generates predictions based on user-provided input.
5. **DeepSHAP Explainability:** Explains the contribution of different features to the prediction.
6. **Spring Boot Backend:** Exposes REST APIs to communicate with the AI model.
7. **Database:** Stores relevant application data and prediction records.

## 🔄 Project Workflow

Traffic Data → Data Preprocessing → ML Model Training → Accident Risk Prediction → DeepSHAP Explanation → API Response

## ⚙️ Installation & Setup

### Prerequisites

- Java JDK 17 or above
- Python 3.10 or above
- Maven
- MySQL
- Git

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd traffic-accident-prediction
```

### 2. Set Up Python Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Database

Create a MySQL database and update the database credentials in the Spring Boot `application.properties` file.

### 5. Run the Spring Boot Application

```bash
mvn clean install
mvn spring-boot:run
```

### 6. Test APIs

Use Postman to test the available REST API endpoints.

## 📊 Explainable AI – DeepSHAP

DeepSHAP is used to interpret the Machine Learning model's predictions.

It helps identify which input features contribute positively or negatively to the predicted accident risk.

This makes the system more transparent and helps users understand the reasoning behind AI-generated predictions.

## 🔮 Future Enhancements

- Real-time traffic data integration.
- GPS-based accident risk prediction.
- Interactive accident hotspot visualization.
- Weather API integration.
- Advanced deep learning models.
- Real-time accident risk alerts.
- Web-based dashboard for prediction visualization.

## 🎓 Project Purpose

This project is developed as an academic and practical implementation of Artificial Intelligence, Machine Learning, Explainable AI, and Backend Development to explore how technology can contribute to improving road safety.

## 👩‍💻 Author

**Muskan Varshney**

B.Tech – Computer Science and Engineering

GitHub: Add your GitHub profile link here

---

⭐ If you find this project interesting, don't forget to star the repository!
