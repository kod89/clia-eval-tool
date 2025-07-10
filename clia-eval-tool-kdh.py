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

        st.subheader("📄 PDF 보고서 생성")
        pdf = FPDF()
        pdf.add_page()

        font_path = "assets/NanumGothic.ttf"
        if os.path.exists(font_path):
            try:
                pdf.add_font("Nanum", "", font_path, uni=True)
                pdf.set_font("Nanum", size=12)
                font_ok = True
            except:
                st.error("❌ 폰트 로딩 오류: NanumGothic.ttf")
                pdf.set_font("Arial", size=12)
                font_ok = False
        else:
            st.error("❌ NanumGothic.ttf 폰트가 누락되었습니다.")
            pdf.set_font("Arial", size=12)
            font_ok = False

        pdf.cell(200, 10, txt="CLIA 분석 성능 평가 보고서", ln=True, align='C')
        pdf.cell(200, 10, txt=f"작성일: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align='C')
        pdf.ln(10)

        if font_ok:
            pdf.set_font("Nanum", size=10)
        else:
            pdf.set_font("Arial", size=10)

        # 성능 지표 해석 (줄바꿈 분리 + 특수문자 제거)
        pdf.multi_cell(0, 8, "[1] 성능 지표 해석")
        pdf.multi_cell(0, 8, f"- 정확도 (Accuracy): {accuracy:.2f}")
        pdf.multi_cell(0, 8, "  전체 샘플 중 예측이 맞은 비율로, 전체적인 모델의 성능을 나타냅니다.")
        pdf.multi_cell(0, 8, f"- 정밀도 (Precision): {precision:.2f}")
        pdf.multi_cell(0, 8, "  양성으로 예측한 것들 중 실제 양성의 비율입니다. 위양성이 적을수록 높습니다.")
        pdf.multi_cell(0, 8, f"- 민감도 (Recall): {recall:.2f}")
        pdf.multi_cell(0, 8, "  실제 양성 중에서 모델이 양성으로 잘 맞춘 비율입니다. 누락된 양성(FN)이 적을수록 좋습니다.")
        pdf.multi_cell(0, 8, f"- F1 점수 (F1 Score): {f1:.2f}")
        pdf.multi_cell(0, 8, "  정밀도와 민감도의 조화 평균으로 두 지표 간 균형을 보여줍니다.")

        # 혼동 행렬
        pdf.ln(5)
        pdf.cell(200, 10, txt="[2] Confusion Matrix 해석", ln=True)
        pdf.image(cm_path, w=160)
        pdf.ln(3)
        pdf.multi_cell(0, 8, "- Confusion Matrix는 예측과 실제 간의 관계를 시각화한 것입니다.")
        pdf.multi_cell(0, 8, "- True Positive, True Negative가 높을수록 모델의 분류 성능이 우수합니다.")
        pdf.multi_cell(0, 8, "- False Positive가 많으면 위양성 발생, False Negative가 많으면 실제 환자 놓침을 의미합니다.")

        # ROC Curve
        pdf.ln(5)
        pdf.cell(200, 10, txt="[3] ROC Curve 해석", ln=True)
        pdf.image(roc_path, w=160)
        pdf.ln(3)
        pdf.multi_cell(0, 8, f"- ROC 곡선은 민감도(TPR)와 위양성률(FPR) 간의 관계를 시각화합니다.")
        pdf.multi_cell(0, 8, f"- AUC 값이 {roc_auc:.2f}로 높을수록 분류기의 성능이 우수함을 의미합니다.")
        pdf.multi_cell(0, 8, "- 곡선이 왼쪽 위로 치우칠수록 좋은 모델입니다.")

        # 최종 요약
        pdf.ln(5)
        pdf.cell(200, 10, txt="[4] 종합 평가 요약", ln=True)
        overall = "우수" if accuracy > 0.9 and roc_auc > 0.9 else "양호" if accuracy > 0.8 else "개선 필요"
        pdf.multi_cell(0, 8, f"- 분석 결과, 전체적인 성능은 '{overall}'으로 평가됩니다.")
        pdf.multi_cell(0, 8, "- 실험 데이터를 기반으로 본 시스템은 현장 진단용으로 활용 가능성이 높습니다.")

        # 파일 출력
        pdf_path = f"clia_eval_report_{datetime.today().strftime('%Y%m%d')}.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 PDF 보고서 다운로드", f, file_name=pdf_path, mime='application/pdf')

        st.success("✅ 분석 완료! 해석이 포함된 PDF 보고서를 다운로드할 수 있습니다.")

    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
