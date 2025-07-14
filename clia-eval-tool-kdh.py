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

        def get_performance_level(acc, auc_score):
            if acc > 0.9 and auc_score > 0.9:
                return "Excellent", "The diagnostic performance is excellent. The test is suitable for real-world clinical applications.", "You may proceed with deployment. Further improvements are optional."
            elif acc > 0.8 and auc_score > 0.8:
                return "Good", "The test shows good performance. However, further evaluation may enhance reliability.", "Consider improving sensitivity and precision before full deployment."
            else:
                return "Needs Improvement", "The current test performance is insufficient for clinical use.", "Model improvement, retraining, and more data collection are strongly recommended."

        level, comment, recommendation = get_performance_level(accuracy, roc_auc)

        st.subheader("📄 PDF 보고서 생성")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="CLIA Evaluation Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Date: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, txt=f"[1] Performance Metrics\n"
                                 f"- Accuracy: {accuracy:.2f}\n"
                                 f"- Precision: {precision:.2f}\n"
                                 f"- Recall: {recall:.2f}\n"
                                 f"- F1 Score: {f1:.2f}")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[2] Confusion Matrix", ln=True)
        pdf.image(cm_path, w=160)
        pdf.multi_cell(0, 8, txt="- This matrix visualizes prediction vs. actual outcomes. A high false positive or false negative rate may indicate clinical risk.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[3] ROC Curve", ln=True)
        pdf.image(roc_path, w=160)
        pdf.multi_cell(0, 8, txt=f"- Area Under Curve (AUC): {roc_auc:.2f}. Higher values indicate better diagnostic ability.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[4] Final Evaluation Summary", ln=True)
        pdf.multi_cell(0, 8, txt=f"- Overall performance level: {level}\n- {comment}")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[5] Recommendation", ln=True)
        pdf.multi_cell(0, 8, txt=f"- {recommendation}")

        pdf_path = f"clia_eval_report_{datetime.today().strftime('%Y%m%d')}.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF Report", f, file_name=pdf_path, mime='application/pdf')

        st.success("✅ Analysis complete! Review the results and generated report.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
