# 🧪 CLIA 분석 성능 평가 자동화 툴

본 프로젝트는 진단 제품의 분석 성능 평가를 자동화하기 위해 제작된 **Streamlit 기반 웹 애플리케이션**입니다.

## 🚀 주요 기능

- CSV 또는 Excel 데이터 업로드
- 민감도, 특이도, 정밀도, 정확도, F1-score 자동 계산
- Confusion Matrix 및 ROC Curve 시각화
- 📄 PDF 보고서 자동 생성 및 다운로드 기능

## 📁 입력 파일 형식

| Sample_ID | True_Label | Test_Result |
|-----------|------------|-------------|
| 001       | 1          | 1           |
| 002       | 0          | 0           |

## 🛠 사용 방법

```bash
git clone https://github.com/your-username/clia-eval-tool.git
cd clia-eval-tool
pip install -r requirements.txt
streamlit run app.py
```

## 📎 샘플 데이터

`assets/sample_data.csv` 파일 포함 (예시 데이터)

---

이 프로젝트는 CLIA 평가, 생산이관, 문서화 업무에 특화된 자동화 도구로, 실무에서 효율성과 일관성을 크게 높일 수 있습니다.
