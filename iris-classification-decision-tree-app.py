import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 讀取資料
iris = pd.read_csv("data/Iris.csv").dropna()

# Streamlit 應用標題
st.title("🌸 Iris 花卉分類決策樹模型")

# 頁面選項
tabs = st.tabs(["📊 資料預覽", "📈 特徵分佈圖", "🧊 相關矩陣", "🧠 模型訓練與評估", "🌳 決策樹視覺化"])

# 資料預覽
with tabs[0]:
    st.write("📊 Iris 資料集前五筆：")
    st.write(iris.head())
    st.write(f"資料集形狀：{iris.shape}")
    st.write(f"分類標籤：{iris['Species'].unique()}")

# 特徵分佈圖
with tabs[1]:
    st.subheader("📈 Sepal 長寬分佈圖")
    fig, ax = plt.subplots()
    sns.scatterplot(data=iris, x='SepalLengthCm',
                    y='SepalWidthCm', hue='Species', palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Petal 長寬分佈圖")
    fig, ax = plt.subplots()
    sns.scatterplot(data=iris, x='PetalLengthCm',
                    y='PetalWidthCm', hue='Species', palette='Set2', ax=ax)
    st.pyplot(fig)

# 相關矩陣
with tabs[2]:
    st.subheader("🧊 特徵相關矩陣")
    numeric_iris = iris.select_dtypes(include=[np.number])  # 選擇數值型欄位
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_iris.corr(), annot=True,
                cmap='Oranges', fmt='.2f', ax=ax)
    st.pyplot(fig)


# 模型訓練與評估
with tabs[3]:
    st.subheader("🧠 訓練決策樹模型")
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = iris['Species']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=100)

    clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=5, min_samples_leaf=3, random_state=100)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.write(f"🎉 模型準確度：{accuracy_score(Y_test, y_pred) * 100:.2f}%")
        st.text("📋 分類報告：")
        st.text(classification_report(Y_test, y_pred))
    
    with col2:
        cm = confusion_matrix(Y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3)) 
        sns.heatmap(cm, annot=True, cmap='Oranges', xticklabels=iris['Species'].unique(),
                    yticklabels=iris['Species'].unique(), fmt='d', ax=ax)
        st.pyplot(fig)


# 決策樹視覺化
with tabs[4]:
    st.subheader("🌳 決策樹結構")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(clf, feature_names=X.columns,
              class_names=iris['Species'].unique(), filled=True)
    st.pyplot(fig)
