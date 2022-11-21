# TensorRT model Evaluation
## mAP 측정 방법
1. ```convert_gt2eval.py``` : ground truth label(*.txt) -> evaluation form 변환
2. ```get_result_evel.py``` : tensorrt model로 image inference 결과 추출
3. mAP 추출
```shell
# in /workspace
python3 tools/eval/convert_gt2eval.py --dataset=obstacle --source=${TEST_LABEL_DIR} --target=tools/evel/input/ground-truth
python3 tools/eval/get_result_eval.py --dataset=obstacle --image_dir=${TEST_IMAGE_DIR} --target=tools/evel/input/detection-results
python3 tools/eval/evaluation.py --iou=0.5 --gt=/workspace/tools/eval/input/ground-truth --det=/workspace/tools/eval/input/detection-results
```