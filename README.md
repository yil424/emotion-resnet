# Facial Emotion Recognition with CNN and SE-ResNet18

- Dataset: 5 emotions (Angry, Fear, Happy, Sad, Surprise), ~59k images.
- Baseline: Custom CNN with data augmentation (flip + rotation).
- Advanced model: SE-ResNet18 (ImageNet pretrained + Squeeze-and-Excitation).
- Interpretability: t-SNE on penultimate features, Grad-CAM heatmaps.
- Demo: Streamlit app + Qwen3: 8b for emotion-aware responses.

> Note: The dataset is not included in this repo due to size.
Web Link: https://emotion-resnet.streamlit.app
