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
from datetime import datetime
import os

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT

st.set_page_config(page_title="CLIA 분석 성능 평가 툴", layout="centered")
st.title("🔬 CLIA 분석 성능 평가 자동화 툴")

uploaded_file = st.file_uploader("📁 평가 결과 파일 업로드 (CSV 또는 Excel)", type=["csv", "xlsx"])

def generate_reportlab_pdf(metrics, cm_path, roc_path, pdf_path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Korean', fontName='Helvetica', fontSize=10, leading=14, alignment=TA_LEFT))

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    story = []

    story.append(Paragraph("CLIA 분석 성능 평가 보고서", styles["Title"]))
    story.append(Paragraph(f"작성일: {datetime.today().strftime('%Y-%m-%d')}", styles["Korean"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>[1] 성능 지표 요약 및 해석</b>", styles["Korean"]))
    story.append(Paragraph(f"- 정확도(Accuracy): {metrics['accuracy']:.2f}", styles["Korean"]))
    story.append(Paragraph(f"- 정밀도(Precision): {metrics['precision']:.2f}", styles["Korean"]))
    story.append(Paragraph(f"- 민감도(Recall): {metrics['recall']:.2f}", styles["Korean"]))
    story.append(Paragraph(f"- F1 Score: {metrics['f1_score']:.2f}", styles["Korean"]))

    story.append(PageBreak())
    story.append(Paragraph("<b>[2] Confusion Matrix</b>", styles["Korean"]))
    story.append(Image(cm_path, width=150*mm, height=100*mm))
    story.append(Spacer(1, 6))
    story.append(Paragraph("- 혼동 행렬은 예측값과 실제 라벨 간의 비교입니다. "
                            "대각선 값이 높을수록 모델 성능이 우수합니다.", styles["Korean"]))

    story.append(PageBreak())
    story.append(Paragraph("<b>[3] ROC Curve</b>", styles["Korean"]))
    story.append(Image(roc_path, width=150*mm, height=100*mm))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"- AUC 값: {metrics['roc_auc']:.2f}. 1에 가까울수록 분류 성능이 우수합니다.", styles["Korean"]))

    story.append(PageBreak())
    story.append(Paragraph("<b>[4] 최종 평가 요약</b>", styles["Korean"]))
    story.append(Paragraph(f"- 전체적인 평가 결과는 \"{metrics['overall']}\" 수준으로 판단됩니다.", styles["Korean"]))

    doc.build(story)

if uploaded_file:
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
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        overall = "우수" if accuracy > 0.9 and roc_auc > 0.9 else "양호" if accuracy > 0.8 else "개선 필요"

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "overall": overall
        }

        # 숫자 지표만 DataFrame으로 표시
        numeric_metrics_df = pd.DataFrame(
            [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))],
            columns=["Metric", "Value"]
        )
        st.dataframe(numeric_metrics_df, use_container_width=True)

        # overall은 따로 출력
        st.markdown(f"**📌 최종 평가 요약:** `{metrics['overall']}` 수준")

        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'], ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        cm_path = "conf_matrix.png"
        fig_cm.savefig(cm_path)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # ROC Curve
        st.subheader("📈 ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        roc_path = "roc_curve.png"
        fig_roc.savefig(roc_path)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

        # PDF 생성
        st.subheader("📄 PDF 보고서 생성")
        pdf_path = f"CLIA_Evaluation_Report_{datetime.today().strftime('%Y%m%d')}.pdf"
        if st.button("PDF 보고서 생성"):
            generate_reportlab_pdf(metrics, cm_path, roc_path, pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 PDF 다운로드", f, file_name=pdf_path, mime="application/pdf")
            st.success("✅ PDF 보고서가 생성되었습니다!")

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
