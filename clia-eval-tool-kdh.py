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
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall (Sensitivity)": recall,
            "F1 Score": f1,
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

        # PDF Report
        st.subheader("📄 PDF 보고서 생성")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="CLIA Evaluation Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Date: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, txt=f"[1] Performance Metrics and Interpretation\n"
                                 f"- Accuracy: {accuracy:.2f} — Proportion of correct predictions.\n"
                                 f"- Precision: {precision:.2f} — Among predicted positives, how many are truly positive.\n"
                                 f"- Recall (Sensitivity): {recall:.2f} — Among actual positives, how many were correctly identified.\n"
                                 f"- F1 Score: {f1:.2f} — Harmonic mean of precision and recall.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[2] Confusion Matrix", ln=True)
        pdf.image(cm_path, w=160)
        pdf.ln(5)
        pdf.multi_cell(0, 8, txt="The confusion matrix compares predicted and actual results. High false positives/negatives may indicate clinical risk.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[3] ROC Curve", ln=True)
        pdf.image(roc_path, w=160)
        pdf.multi_cell(0, 8, txt=f"Area Under Curve (AUC): {roc_auc:.2f}. A higher AUC indicates better diagnostic ability. This test maintains low false positive rate with good sensitivity.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[4] Final Evaluation Summary", ln=True)

        # 평가 레벨 판별
        if accuracy > 0.9 and roc_auc > 0.9:
            overall = "Excellent"
            advice = "The test kit shows excellent diagnostic performance with high reliability in clinical settings."
        elif accuracy > 0.8:
            overall = "Good"
            advice = "Performance is acceptable but may benefit from further optimization for critical applications."
        else:
            overall = "Needs Improvement"
            advice = "The test shows suboptimal results. Investigate possible causes such as sample quality, cutoff settings, or model calibration."

        pdf.multi_cell(0, 8, txt=(f"Metric-based evaluation:\n"
                                 f"- Accuracy level: {'High' if accuracy > 0.9 else 'Moderate' if accuracy > 0.8 else 'Low'}\n"
                                 f"- Precision level: {'High' if precision > 0.9 else 'Moderate' if precision > 0.8 else 'Low'}\n"
                                 f"- Recall level: {'High' if recall > 0.9 else 'Moderate' if recall > 0.8 else 'Low'}\n"
                                 f"- F1 Score level: {'High' if f1 > 0.9 else 'Moderate' if f1 > 0.8 else 'Low'}\n\n"
                                 f"Overall performance level: {overall}\n"
                                 f"Recommendation: {advice}"))

        pdf_path = f"clia_eval_report_{datetime.today().strftime('%Y%m%d')}.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 PDF 보고서 다운로드", f, file_name=pdf_path, mime='application/pdf')

        st.success("✅ 분석 완료! 결과 및 보고서를 확인하세요.")

    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
