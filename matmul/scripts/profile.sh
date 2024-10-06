sudo nsys profile --stats=true --trace=cuda,nvtx --cuda-memory-usage=true --gpu-metrics-device=all ./main 3
sudo $(which ncu) -o sgemm3_profile --set full --kernel-name "sgemm3" ./main 3