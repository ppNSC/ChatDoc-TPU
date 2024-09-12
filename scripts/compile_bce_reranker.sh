mkdir -p bmodel
pushd bmodel 

model_transform.py \
  --model_name bce-reranker-base_v1 \
  --model_def ../onnx/bce-reranker-base_v1.onnx \
  --input_shapes [[3,512],[3,512]] \
  --mlir bce-reranker-base_v1.mlir

  model_deploy.py \
  --mlir bce-reranker-base_v1.mlir \
  --quantize F16 \
  --chip bm1684x \
  --model bce-reranker-base_v1.bmodel \
  --compare_all \
  --debug

  popd