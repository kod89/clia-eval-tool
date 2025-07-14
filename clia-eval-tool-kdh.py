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
        pdf.multi_cell(0, 8, txt=f"[1] Summary of Performance Metrics\n"
                                 f"- Accuracy: {accuracy:.2f}\n"
                                 f"- Precision: {precision:.2f}\n"
                                 f"- Recall: {recall:.2f}\n"
                                 f"- F1 Score: {f1:.2f}")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[2] Confusion Matrix", ln=True)
        pdf.image(cm_path, w=160)
        pdf.ln(5)
        pdf.multi_cell(0, 8, txt="- The confusion matrix compares predicted vs. actual labels.\n"
                                 "High false positives or negatives may indicate clinical risk.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[3] ROC Curve", ln=True)
        pdf.image(roc_path, w=160)
        pdf.multi_cell(0, 8, txt=f"- AUC (Area Under Curve): {roc_auc:.2f}.\n"
                                 "The closer to 1.0, the better the diagnostic performance.\n"
                                 "This test demonstrates a good trade-off between sensitivity and specificity.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[4] Final Evaluation Summary", ln=True)

        # 유니코드 회피용 텍스트 필터
        def sanitize_text(text):
            return text.replace("—", "-").replace("–", "-").replace("•", "-")

        # 평가 생성 함수
        def generate_evaluation_summary(acc, prec, rec, f1s, auc_score):
            summary = f"- Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f1s:.2f}, AUC: {auc_score:.2f}\n\n"

            if acc >= 0.9:
                summary += "- Excellent overall prediction accuracy.\n"
            elif acc >= 0.8:
                summary += "- Good accuracy, though improvement is possible.\n"
            else:
                summary += "- Low accuracy suggests inconsistent overall predictions.\n"

            if prec >= 0.9:
                summary += "- High precision: very few false positives.\n"
            elif prec >= 0.75:
                summary += "- Moderate precision: some false positives, caution required.\n"
            else:
                summary += "- Low precision: many false positives, may lead to unnecessary testing.\n"

            if rec >= 0.9:
                summary += "- High recall: most true positives detected.\n"
            elif rec >= 0.75:
                summary += "- Moderate recall: some true positives may be missed.\n"
            else:
                summary += "- Low recall: high risk of missing true cases.\n"

            if f1s >= 0.9:
                summary += "- F1 score indicates excellent balance of precision and recall.\n"
            elif f1s >= 0.75:
                summary += "- F1 score shows moderate balance, possible trade-offs.\n"
            else:
                summary += "- Low F1 score: model struggles to balance precision and recall.\n"

            if auc_score >= 0.9:
                summary += "- AUC indicates strong ability to distinguish between classes.\n"
            elif auc_score >= 0.75:
                summary += "- Moderate AUC: fair discrimination, could be improved.\n"
            else:
                summary += "- Low AUC: weak class separation, diagnostic confidence may be limited.\n"

            summary += "\nRecommendation: Consider improving data quality, adjusting decision thresholds, or retraining with more representative samples."
            return summary

        evaluation_summary = sanitize_text(generate_evaluation_summary(accuracy, precision, recall, f1, roc_auc))
        pdf.multi_cell(0, 8, txt=evaluation_summary)

        pdf_path = f"clia_eval_report_{datetime.today().strftime('%Y%m%d')}.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF Report", f, file_name=pdf_path, mime='application/pdf')

        st.success("✅ Analysis complete! Check the results and download the report.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
