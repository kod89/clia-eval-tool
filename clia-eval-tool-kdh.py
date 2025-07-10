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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os

st.set_page_config(page_title="CLIA Performance Evaluation Tool", layout="centered")
st.title("🔬 CLIA Performance Evaluation Tool")

uploaded_file = st.file_uploader("📁 Upload Evaluation File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        y_true = df["True_Label"]
        y_pred = df["Test_Result"]

        st.subheader("✅ Performance Metrics")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax_cm.set_title("Confusion Matrix")
        tick_marks = np.arange(2)
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_xticklabels(["Negative", "Positive"])
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_yticklabels(["Negative", "Positive"])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
        fig_cm.tight_layout()
        cm_path = "confusion_matrix_eng.png"
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
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc="lower right")
        roc_path = "roc_curve_eng.png"
        fig_roc.savefig(roc_path)
        st.pyplot(fig_roc)

        st.subheader("📄 Generate PDF Report")
        pdf_path = f"clia_eval_report_eng_{datetime.today().strftime('%Y%m%d')}.pdf"
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, height - 30 * 72 / 25.4, "CLIA Performance Evaluation Report")

        c.setFont("Helvetica", 10)
        c.drawString(20 * 72 / 25.4, height - 40 * 72 / 25.4, f"Date: {datetime.today().strftime('%Y-%m-%d')}")

        y = height - 55 * 72 / 25.4
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * 72 / 25.4, y, "[1] Summary of Performance Metrics")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(25 * 72 / 25.4, y, f"- Accuracy: {accuracy:.2f}")
        y -= 12
        c.drawString(25 * 72 / 25.4, y, f"- Precision: {precision:.2f}")
        y -= 12
        c.drawString(25 * 72 / 25.4, y, f"- Recall (Sensitivity): {recall:.2f}")
        y -= 12
        c.drawString(25 * 72 / 25.4, y, f"- F1 Score: {f1:.2f}")
        y -= 20

        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * 72 / 25.4, y, "[2] Confusion Matrix")
        y -= 180
        c.drawImage(ImageReader(cm_path), 35 * 72 / 25.4, y, width=100 * 72 / 25.4, preserveAspectRatio=True)
        y -= 70
        c.setFont("Helvetica", 10)
        c.drawString(25 * 72 / 25.4, y, "- The matrix shows the number of correct and incorrect predictions.")

        y -= 40
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * 72 / 25.4, y, "[3] ROC Curve")
        y -= 180
        c.drawImage(ImageReader(roc_path), 35 * 72 / 25.4, y, width=100 * 72 / 25.4, preserveAspectRatio=True)
        y -= 70
        c.setFont("Helvetica", 10)
        c.drawString(25 * 72 / 25.4, y, f"- AUC: {roc_auc:.2f} (Closer to 1.0 means better diagnostic performance)")

        y -= 40
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * 72 / 25.4, y, "[4] Final Evaluation")
        y -= 15
        c.setFont("Helvetica", 10)
        overall = "Excellent" if accuracy > 0.9 and roc_auc > 0.9 else "Good" if accuracy > 0.8 else "Needs Improvement"
        c.drawString(25 * 72 / 25.4, y, f"- Overall performance is evaluated as \"{overall}\".")
        c.save()

        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF Report", f, file_name=pdf_path, mime='application/pdf')

        st.success("✅ Analysis completed successfully!")

    except Exception as e:
        st.error(f"Error processing file: {e}")
