# Presentation outline

Use this validated outline to build the final PPT/PDF:

1. **Question:** Does pose fail before detection under surveillance degradation,
   and can enhancement or corruption-aware fine-tuning recover performance?
2. **Dataset:** deterministic 150-image COCO-person validation subset, seed 42.
3. **Three tasks:** Canny boundary F1, YOLOv8n detection F1/recall, and
   YOLOv8n-pose PCK@0.2/OKS.
4. **Three distortions:** low light, motion blur, JPEG; mild/medium/severe.
5. **Enhancement:** CLAHE+gamma, stable unsharp masking, bilateral filtering.
6. **Clean baseline:** boundary F1 0.547; detection F1 0.770; pretrained pose
   PCK@0.2 0.931.
7. **Main failure:** severe motion blur reduced boundary F1 to 0.298,
   detection F1 to 0.406, and pose PCK@0.2 to 0.745.
8. **Best recovery:** severe JPEG bilateral filtering raised detection F1
   0.431→0.600 and pose PCK@0.2 0.770→0.872.
9. **Cross-level result:** unsharp masking improved severe-blur boundary F1
   0.298→0.345 while detection and pose became slightly worse.
10. **Fine-tuning result:** fine-tuned pose underperformed pretrained pose in
    all 19 matched conditions; clean PCK@0.2 was 0.825 vs 0.931.
11. **Limitations:** subset scale, no official detection AP/pose AP, no complete
    per-joint/person-size final aggregate.
12. **Conclusion:** classical enhancement is distortion-specific; stronger
    low-level edges do not guarantee high-level recovery; small corruption-aware
    fine-tuning can reduce localization accuracy.

All numbers must be sourced from
`results/coco_person/tables/val_aggregate.csv`.
