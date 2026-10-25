Loop for each optimization:
Edit Solver_GPU_GlobalMemory_fuse/compressed.cu, then run bash build.sh.
bash gpu_compare.sh fire fuse tiny must print identical.
bash gpu_compare.sh fire fuse times it on long.
bash gpu_profile.sh nsys fire long SOR gpu-optimized fuse - Compare its summary.txt with the original's.