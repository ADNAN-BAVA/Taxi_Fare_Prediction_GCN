# 🚖 **Hybrid GCN-XGBoost Model for E-Hailing Price Prediction and Genetic Algorithm for Demand-Based Allocation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/) [![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](#)  [![Platform](https://img.shields.io/badge/Platform-Jupyter-orange.svg)](#)

---

## 📝 **Overview**  
This project addresses critical challenges in **e-hailing services**, including:  
- Inaccurate **fare predictions** due to dynamic urban patterns.  
- Inefficient **vehicle allocation** leading to increased wait times and driver fatigue.  
- Managing the **spatial-temporal complexities** of fluctuating demand.

To solve these challenges, a **hybrid model** combining **Graph Convolutional Networks (GCN)** and **XGBoost** for fare prediction, along with a **Genetic Algorithm (GA)** for vehicle allocation, has been implemented.

🚀 **Deployed via Streamlit**, the interactive web application enables:  
- Real-time **fare prediction**.  
- Optimized **vehicle allocation** for peak demand.  
- Visual analysis of **demand trends**.

---

## 🚀 **Key Features**  
✅ **Hybrid Fare Prediction Model**  
   - **GCN**: Captures spatial relationships between zones.  
   - **XGBoost**: Accurate regression-based fare predictions.  
   - **Metrics**:  
     - 🏆 **RMSE**: `2.13`  
     - 🏆 **MAE**: `0.56`  

✅ **Genetic Algorithm for Vehicle Allocation**  
   - Optimizes vehicle distribution based on **high-demand zones**.  
   - Reduces idle vehicle time and increases service efficiency.  

✅ **Streamlit Real-Time Deployment**  
   - 🎯 **Fare Prediction**: Input location & time for instant predictions.  
   - 📊 **Demand Analysis**: Visualize demand trends dynamically.  
   - 🛠️ **Resource Optimization**: Insights into vehicle placement efficiency.  

---

## 🎯 **Research Objectives**  
1. 🚗 Develop a **hybrid GCN-XGBoost** model for improved fare accuracy.  
2. 🗺️ Analyze **spatial dependencies** between urban zones.  
3. 🧬 Implement **Genetic Algorithms** for vehicle allocation optimization.  
4. 🌐 Deploy an **interactive web application** using Streamlit.  

---

## ⚙️ **Methodology**  
### 📥 **Data Collection**  
- **Sources**: NYC taxi trip data combined with **weather** and **geospatial** features.
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### 🛠️ **Data Preprocessing & Feature Engineering**  
- **Outlier Removal** & **Scaling** of continuous variables.  
- Geospatial Features: 📍 **Haversine formula** for accurate distance calculations.  
- Temporal Features: ⏰ Time of day, day of week, and seasonal data.

### 🧠 **Model Implementation**  
1. 🌍 **GCN**: Captures spatial dependencies for fare dynamics.  
2. 📈 **XGBoost**: Structured regression model for accurate fare prediction.  
3. 🧬 **Genetic Algorithm**: Optimizes resource allocation in high-demand zones.

---

## 📊 **Performance Metrics**  
| Metric           | Value       |  
|------------------|-------------|  
| **RMSE**         | `2.13`      |  
| **MAE**          | `0.56`      |  

- 🚀 Outperforms **Random Forest** and **Linear Regression** models.  
- 📉 **Idle Vehicle Time**: Reduced through optimized allocation strategies.  
- 💰 **Revenue**: Improved through targeted vehicle placement in demand zones.

---

## 💡 **Discussion**  
### ✅ **Strengths**  
- 🚀 Improved fare prediction accuracy.  
- 📊 Optimized vehicle allocation using GA.  
- 🌐 User-friendly real-time interface.  

### ⚠️ **Limitations**  
- Performance in **rural areas** is limited due to sparse spatial data.  
- Lack of **real-time traffic integration** may affect scalability.  

### 🔮 **Future Enhancements**  
- 🔗 Integrate **real-time traffic data** for improved accuracy.  
- 🏙️ Test adaptability across diverse urban settings.  
- 📲 Enhance **Streamlit application** features for better user experience.

---

## 🎯 **Conclusion**  
The hybrid **GCN-XGBoost** model and **Genetic Algorithm** effectively improve fare prediction and vehicle resource allocation for e-hailing platforms. This project demonstrates the potential for enhancing operational efficiency and user satisfaction in dynamic urban environments.

---

## 🌐 **Deployment**  
The project is deployed as a **Streamlit** application for real-time decision-making.  

- **Features**:  
   - 🎯 Instant Fare Predictions  
   - 🚗 Vehicle Allocation Optimization  
   - 📊 Dynamic Demand Analysis  

### **GUI**
Refer GUI Folder in Repo
---

## 📞 **Contact**  
**Mohammad Adnan**  
**Project Title**: *Hybrid GCN-XGBoost Model for E-Hailing Price Prediction and Genetic Algorithm for Demand-Based Allocation*  

🔗 For inquiries or collaborations, feel free to connect!

---

