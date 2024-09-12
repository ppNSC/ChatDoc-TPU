mkdir -p bmodel
pushd bmodel 

model_transform.py \
  --model_name bce-embedding-base_v1 \
  --model_def ../onnx/bce-embedding-base_v1.onnx \
  --input_shapes [[4,512],[4,512]] \
  --mlir bce-embedding-base_v1.mlir

  model_deploy.py \
  --mlir bce-embedding-base_v1.mlir \
  --quantize F16 \
  --chip bm1684x \
  --model bce-embedding-base_v1.bmodel \
  --compare_all \
  --debug

  popd