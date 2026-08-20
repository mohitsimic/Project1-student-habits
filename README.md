# Student Habits & Exam Score Prediction

Predicts exam scores from student lifestyle/habit data (study hours, sleep, social media use, etc.) using a Random Forest regression pipeline.

## Setup
pip install -r requirements.txt

## Run
python src/evaluate.py

## Results
- RMSE: ~6.4 points (0-100 scale)
- R²: ~0.86
- Baseline (predict mean) RMSE: 17.2636001209497
-                         R² : -0.0007215789431849196

## Key findings
- `study_hours_per_day` dominates feature importance (~69%)
- `part_time_job` has negligible measured effect on predicted scores
