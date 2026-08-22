# Student Habits & Exam Score Prediction

Predicts exam scores from student lifestyle/habit data (study hours, sleep, social media use, etc.) using a Random Forest regression pipeline.

## Setup
pip install -r requirements.txt

## Run
python src/evaluate.py
## Model Comparison (7-fold CV)
| Model | RMSE | R² |
|---|---|---|
| Random Forest | 6.54 | 0.847 |
| Gradient Boosting | 5.89 | 0.876 |
| Linear Regression | 5.51 | 0.891 |

## Key findings
- Linear Regression outperformed both tree-based ensembles, suggesting the relationship between habits and exam performance is largely linear
- `study_hours_per_day` dominates feature importance (~69%)
- `part_time_job` has negligible measured effect on predicted scores
- Baseline (predict mean) RMSE: 16.69, R² ≈ 0.00 — all three models substantially outperform this
