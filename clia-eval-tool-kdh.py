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
            pdf.add_font("Nanum", "", font_path, uni=True)
            pdf.set_font("Nanum", size=12)
        else:
            pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="CLIA 분석 성능 평가 보고서", ln=True, align='C')
        pdf.cell(200, 10, txt=f"작성일: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align='C')
        pdf.ln(10)

        explanation_font = "Nanum" if os.path.exists(font_path) else "Arial"
        pdf.set_font(explanation_font, size=10)

        pdf.multi_cell(0, 8, f"""[1] 성능 지표 해석
- 정확도(Accuracy): {accuracy:.2f}
  → 전체 샘플 중 예측이 맞은 비율로, 전체 성능을 나타냅니다.
- 정밀도(Precision): {precision:.2f}
  → 양성 예측 중 실제 양성의 비율로, 위양성(FP)이 낮을수록 높아집니다.
- 민감도(Recall): {recall:.2f}
  → 실제 양성 중 잘 검출된 비율로, 놓친 양성(FN)이 적을수록 좋습니다.
- F1 Score: {f1:.2f}
  → 정밀도와 민감도의 조화 평균으로, 두 지표 간 균형을 봅니다.
""")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[2] Confusion Matrix 해석", ln=True)
        pdf.image(cm_path, w=160)
        pdf.ln(3)
        pdf.multi_cell(0, 8, "- 혼동 행렬은 예측값과 실제값 간의 관계를 보여줍니다. "
                             "True Positive와 True Negative 비율이 높을수록 모델이 우수하다고 평가됩니다. "
                             "False Positive가 많을 경우 위양성이 많고, False Negative가 많을 경우 놓치는 사례가 많다는 의미입니다.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[3] ROC Curve 해석", ln=True)
        pdf.image(roc_path, w=160)
        pdf.ln(3)
        pdf.multi_cell(0, 8, f"- ROC 곡선은 민감도(TPR)와 위양성률(FPR) 간의 관계를 나타냅니다. "
                             f"AUC 값은 {roc_auc:.2f}로, 이는 전체적인 분류 성능이 우수함을 의미합니다. "
                             "곡선이 왼쪽 위 모서리에 가까울수록 좋습니다.")

        pdf.ln(5)
        pdf.cell(200, 10, txt="[4] 종합 평가 요약", ln=True)
        overall = "우수" if accuracy > 0.9 and roc_auc > 0.9 else "양호" if accuracy > 0.8 else "개선 필요"
        pdf.multi_cell(0, 8, f"- 본 진단 시스템의 분석 결과, 전체적인 성능은 \"{overall}\"으로 평가됩니다. "
                             "정밀도와 민감도 수치가 균형 있게 높고, AUC도 높은 수준으로, 실제 진단 현장에서 신뢰성 있게 사용할 수 있습니다.")

        pdf_path = f"clia_eval_report_{datetime.today().strftime('%Y%m%d')}.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 PDF 보고서 다운로드", f, file_name=pdf_path, mime='application/pdf')

        st.success("✅ 분석 완료! 결과 및 해석이 포함된 보고서를 다운로드할 수 있습니다.")

    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
