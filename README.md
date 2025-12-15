# Facial Emotion Recognition with CNN and SE-ResNet18

- Dataset: 5 emotions (Angry, Fear, Happy, Sad, Surprise), ~59k images.
- Baseline: Custom CNN with data augmentation (flip + rotation).
- Advanced model: SE-ResNet18 (ImageNet pretrained + Squeeze-and-Excitation).
- Self-designed advanced model: SE-ResNet18 + SpatialAttention + label smoothing
- Interpretability: t-SNE on penultimate features, Grad-CAM heatmaps.
- Demo: Streamlit app + Qwen2.5:1.5B-Instruct for emotion-aware responses.

- Web Link: https://emotion-resnet.streamlit.app

> Note: The dataset is not included in this repo due to size.
