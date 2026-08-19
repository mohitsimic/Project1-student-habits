# Student Habits & Exam Score Prediction

Predicts exam scores from student lifestyle/habit data (study hours, sleep, social media use, etc.) using a Random Forest regression pipeline.

## Setup
pip install -r requirements.txt

## Run
python src/evaluate.py

## Results
- RMSE: ~6.4 points (0-100 scale)
- R²: ~0.86
- Baseline (predict mean) RMSE: [fill in once you run the dummy comparison]

## Key findings
- `study_hours_per_day` dominates feature importance (~69%)
- `part_time_job` has negligible measured effect on predicted scores
