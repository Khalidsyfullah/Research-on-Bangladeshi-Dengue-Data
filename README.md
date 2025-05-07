## Research on Bangladeshi Dengue Data

This project focuses on early dengue detection using Bangladeshi data, including dengue case reports from 2016–2024 across 64 districts and a symptom-based survey. A two-step machine learning framework is proposed to enhance early, non-invasive diagnosis based on both seasonal trends and clinical symptoms.

### Dataset Description

**1. Dengue Symptoms Dataset**
Collected symptom data from dengue patients across Bangladesh using a structured Google Form based on WHO guidelines ([WHO Factsheet](https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue)). The dataset underwent multiple preprocessing steps to ensure quality and consistency.
Form Link: [Google Form](https://forms.gle/WhkjRrua6Z4wPEpTA)

**2. Location\_Dataset**
District-wise monthly dengue case counts in Bangladesh from 2016 to 2023, retrieved from the Directorate General of Health Services ([DGHS](https://old.dghs.gov.bd/index.php/bd/home/5200-daily-dengue-status-report)).

**3. Location\_data**
National dengue case data, used primarily for visualization and plotting purposes.

### Methodology and Implementation

Dengue is a critical health issue in Bangladesh, demanding timely diagnosis to reduce fatalities. This study proposes a two-stage machine learning pipeline:

1. **Trend Forecasting:**

   * Used LSTM and Prophet models to forecast dengue trends based on historical district-wise case data.
   * The Prophet model outperformed LSTM with lower MAE and RMSE.

2. **Symptom-Based Prediction:**

   * The forecasted trend data was merged with the symptom dataset (month and district included).
   * Random Forest and XGBoost were applied for dengue case prediction.
   * The symptom-only model achieved **92.5% accuracy**, while the merged dataset achieved **82.3%**.

This approach demonstrates that integrating external symptoms with regional case trends can enable faster, more accurate, and non-invasive dengue diagnosis—helping improve public health response and resource allocation.

🔗 [Full Paper (IEEE Xplore)](https://ieeexplore.ieee.org/abstract/document/10800244)


### Citation

Please cite the work as:

```bibtex
@INPROCEEDINGS{10800244,
  author={Syfullah, Md. Khalid and Ali, Md. Santo and Oishy, Asfia Moon and Hossain, Md. Sozib},
  booktitle={2024 IEEE International Conference on Power, Electrical, Electronics and Industrial Applications (PEEIACON)}, 
  title={Towards Early Dengue Diagnosis in Bangladesh: A Non-Invasive Prediction Model Based on Symptoms and Local Trends}, 
  year={2024},
  pages={833-838},
  doi={10.1109/PEEIACON63629.2024.10800244}
}
```
