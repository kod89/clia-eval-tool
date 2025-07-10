import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT


def generate_clia_pdf_report_streamlit(pdf_path, metrics, cm_img_path, roc_img_path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='English', fontName='Helvetica', fontSize=11, leading=14, spaceAfter=10, alignment=TA_LEFT))

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    story = []
    story.append(Paragraph("CLIA Evaluation Performance Report", styles["Title"]))
    story.append(Paragraph(f"Date: {datetime.today().strftime('%Y-%m-%d')}", styles["English"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>[1] Performance Metrics</b>", styles["English"]))
    story.append(Paragraph(f"- Accuracy: {metrics['accuracy']:.2f}", styles["English"]))
    story.append(Paragraph(f"- Precision: {metrics['precision']:.2f}", styles["English"]))
    story.append(Paragraph(f"- Recall: {metrics['recall']:.2f}", styles["English"]))
    story.append(Paragraph(f"- F1 Score: {metrics['f1_score']:.2f}", styles["English"]))

    story.append(PageBreak())
    story.append(Paragraph("<b>[2] Confusion Matrix</b>", styles["English"]))
    story.append(Image(cm_img_path, width=150*mm, height=100*mm))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This matrix shows predicted vs. actual classifications. High diagonal values indicate better performance.", styles["English"]))

    story.append(PageBreak())
    story.append(Paragraph("<b>[3] ROC Curve</b>", styles["English"]))
    story.append(Image(roc_img_path, width=150*mm, height=100*mm))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"AUC: {metrics['roc_auc']:.2f} — Higher values indicate better discrimination.", styles["English"]))

    story.append(PageBreak())
    story.append(Paragraph("<b>[4] Overall Assessment</b>", styles["English"]))
    story.append(Paragraph(f"Overall model performance is rated as: {metrics['overall']}", styles["English"]))

    doc.build(story)
    return pdf_path


# Streamlit 앱 UI
st.title("🔬 CLIA 분석 성능 평가 자동화 툴")

uploaded_file = st.file_uploader("CSV 파일 업로드 (마지막 열이 타겟값)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 업로드된 데이터")
    st.dataframe(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    overall = "Excellent" if roc_auc >= 0.9 else "Good" if roc_auc >= 0.8 else "Fair"

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "overall": overall
    }

    st.subheader("✅ 분석 성능 지표")
    st.json(metrics)

    # Confusion Matrix 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("output", exist_ok=True)
    cm_path = "output/conf_matrix.png"
    plt.savefig(cm_path)
    st.pyplot(plt.gcf())
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = "output/roc_curve.png"
    plt.savefig(roc_path)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("📄 PDF 보고서 생성")
    if st.button("PDF 생성"):
        report_path = "output/clia_eval_report.pdf"
        generate_clia_pdf_report_streamlit(report_path, metrics, cm_path, roc_path)
        with open(report_path, "rb") as f:
            st.download_button("📥 다운로드", f, file_name="clia_eval_report.pdf")
