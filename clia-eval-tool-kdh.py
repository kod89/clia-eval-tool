
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
import os

st.set_page_config(page_title="CLIA 분석 성능 평가 툴", layout="centered")
st.title("🔬 CLIA 분석 성능 평가 자동화 툴")

uploaded_file = st.file_uploader("📁 평가 결과 파일 업로드 (CSV 또는 Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        y_true = df["True_Label"]
        y_pred = df["Test_Result"]

        st.subheader("✅ 성능 지표 요약")
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall (Sensitivity)": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
        }
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        fig_cm.savefig(cm_path)
        st.pyplot(fig_cm)

        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        roc_path = "roc_curve.png"
        fig_roc.savefig(roc_path)
        st.pyplot(fig_roc)

        # PDF Report Generation
        st.subheader("📄 PDF 보고서 생성")
        pdf = FPDF()
        pdf.add_page()
        font_path = "assets/NanumGothic.ttf"
        if os.path.exists(font_path):
            pdf.add_font("Nanum", "", font_path, uni=True)
            pdf.set_font("Nanum", size=12)
        else:
            pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="CLIA 분석 성능 평가 보고서", ln=True, align='C')
        pdf.cell(200, 10, txt=f"작성일: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align='C')

        pdf.ln(10)
        pdf.set_font("Nanum" if os.path.exists(font_path) else "Arial", size=10)
        pdf.cell(200, 10, txt="[1] 성능 지표", ln=True)
        for metric, value in metrics.items():
            pdf.cell(200, 8, txt=f"- {metric}: {value:.4f}", ln=True)

        pdf.ln(5)
        pdf.cell(200, 10, txt="[2] Confusion Matrix", ln=True)
        pdf.image(cm_path, w=160)

        pdf.ln(5)
        pdf.cell(200, 10, txt="[3] ROC Curve", ln=True)
        pdf.image(roc_path, w=160)

        pdf_path = f"clia_eval_report_{datetime.today().strftime('%Y%m%d')}.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 PDF 보고서 다운로드", f, file_name=pdf_path, mime='application/pdf')

        st.success("✅ 분석 완료! 결과 및 보고서를 확인하세요.")

    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
