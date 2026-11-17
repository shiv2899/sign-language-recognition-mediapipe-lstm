# Sign Language Recognition - MediaPipe + LSTM

Real-time sign language recognition using MediaPipe landmarks + LSTM.

## Status: Work in Progress - Paused

### What is Done:
- Data collected for 3 signs: Hello, Yes, No (50 videos each, 30 frames each = 4500 files)
- Training V1: Full 1662 features - Test Accuracy 44.44%
- Training V2: Full 1662 features + Dropout - Test Accuracy 53.33%
- Training V3: Hands-Only 258 features - Test Accuracy 53.33%

### Key Learning:
- Yes vs No are confusing the model (Yes recall only 0.20) because both are head movements.
- Next step: Replace Yes/No with Thank You / Help (hand-based signs, more distinct).

### Next:
- Collect Thank You, Help data (50 videos each)
- Retrain with Hello, Thank You, Help
- Live webcam demo with predict_live.py

## Files:
- `collect_data.py` - Webcam data collection with MediaPipe Holistic
- `01_sign_language_training_v1.ipynb` - Training experiments (V1, V2, V3)