# ggml-speech GPU structural facts ledger

Cross-task facts about GPU backends in this tree and the engines that consume it.
Check each fact against the current target before relying on it; append new facts
with evidence (file:line or measured output). Do not record hypotheses here — only
verified mechanisms.

## Engine dispatch (qvac-ext-lib-whisper.cpp/engines/audiogen)

- All AceSTEP stages run DIRECT `ggml_backend_graph_compute` on one backend
  (dit_ggml.cpp:743, lm_ggml.cpp:446/590, textenc_ggml.cpp:186, cond_ggml.cpp:353,
  detok_ggml.cpp:222). `supports_op` is consulted only by the flash-attention probe
  (dit_ggml.cpp:305-344) and by the VAE scheduler ([GPU,CPU], op_offload=false,
  vae_ggml.cpp:388-421) which exists for progress/cancel callbacks. Scheduler-based
  fallback reasoning applies ONLY to the VAE stage.
- Stage placement is an allowlist (stage_placement.h:76-95): under Vulkan the LM
  runs on CPU (Mali-G715 observation), detok/DiT/VAE/encoders on GPU. ROCm/MUSA are
  not validated backends (backend_registry.h:63-70); unit tests pin these exclusions
  (test_acestep_units.cpp:316-319, :380-381).
- Default memory mode loads and frees stage weights per generation
  (engine.cpp:1114); ACESTEP_KEEP_STAGES=1 keeps them resident.

## ggml-vulkan on AMD Strix Halo (Radeon 8060S, RADV GFX1151, Mesa 25.2.8)

- Runtime device line: `uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory:
  65536 | int dot: 0 | matrix cores: KHR_coopmat`. Reported device memory = GTT
  (~116 GiB); VRAM carve-out is 2 GiB.
- Arch detection maps GFX1151 to AMD_RDNA3 (ggml-vulkan.cpp:343-345); there is no
  RDNA3.5/RDNA4 distinction.
- RADV gets KHR coopmat unconditionally (ggml-vulkan.cpp:17317-17322); coopmat2 is
  VK_NV-only. Default subgroup size reported by RADV is 64 (wave64).
- gpu_pipeline_configs (:3400-3415) pins subgroup sizes only for RDNA1/RDNA2;
  AMD_RDNA3 is unpinned (get_subgroup_size returns 0).
- The RDNA occupancy-limiting shmem workaround (:3151-3165) is annotated "guessed,
  tested on RDNA2".
- UMA devices force prefer_host_memory=true AFTER the env read (:5487-5492); the
  GGML_VK_PREFER_HOST_MEMORY env is presence-checked, so "off" requires a code
  change. Buffer allocation order for prefer_host_memory: HostVisible|HostCoherent
  first, DeviceLocal fallback (:2963-2975). Measured only on Samsung Xclipse 920
  before this campaign.
- The DiT sets GGML_PREC_F32 only on flash attention (engines/audiogen
  dit_ggml.cpp:340, 356). On coopmat devices, FA_COOPMAT1 requires f32acc coopmat
  support or the FA path silently becomes FA_SCALAR (:3244-3252).
- ggml_vk_buffer_from_host_ptr is implemented (:17101-17124, requires
  VK_EXT_external_memory_host) but device caps advertise buffer_from_host_ptr=false
  (:16417).
- Ubuntu 25.10 system glslc does not support GL_EXT_integer_dot_product (cmake
  feature probe), so integer-dot Vulkan shaders are not built and the device line
  shows `int dot: 0` even though the hardware is RDNA3.
- Verified coopmat configuration table (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR,
  Mesa 25.2.8): 16x16x16 subgroup scope only; f16xf16 with C/R f16 AND C/R f32
  (f32acc IS supported, so GGML_PREC_F32 flash attention can use FA_COOPMAT1);
  full int8 set (u8/s8 x u8/s8 -> s32/u32, with saturating variants). No bf16
  coopmat. The int8 coopmat hardware capability is currently unused by
  ggml-vulkan shaders.
- FIXED on this branch: test-backend-ops IM2COL_3D used to crash with
  GGML_ASSERT (compute workgroup count exceeded maxComputeWorkGroupCount[1] =
  65535 on RADV for large-OW cases) because the elements switch clamped only
  the z dimension and the shader had no y grid-stride loop. ggml-cuda received
  grid-striding fixes for the analogous overflow (im2col/pad); ggml-vulkan's
  im2col_3d dispatch was not covered until now. Was not part of any AceSTEP
  graph (VAE is 1D).
- Per-op GPU timing: GGML_VK_PERF_LOGGER=1 (+_CONCURRENT, _FREQUENCY). Memory
  placement audit: GGML_VK_MEMORY_LOGGER=1. Persistent pipeline cache:
  GGML_VK_PIPELINE_CACHE_DIR.
- SNAKE and COL2IM_1D have implementations on CPU, CUDA (inherited by HIP), Vulkan,
  Metal, OpenCL. No ACE-Step op participates in Vulkan fusion rules.

## Strix Halo memory-access facts (measured on the AceSTEP optimization campaign)

- Linear streaming (big elementwise ADD, 393 MB): ~208 GB/s. Small-segment
  strided access collapses: 128-256 B segments at multi-KiB stride run at
  2.6-22 GB/s — roughly 10-40x below linear streaming. Any kernel whose
  per-workgroup global accesses are narrow strided segments is
  pattern-bound, not bandwidth-bound, once data spills the ~32 MiB MALL.
- Fixes that worked: LDS-staged tiles with contiguous slab loads
  (col2im_1d_tiled.comp: 5-22 -> 88-146 GB/s; copy_transpose_large.comp:
  2.6-5 -> 150-195 GB/s). Thread count matters as much as segment width
  (256 -> 512 -> 1024 threads roughly doubled throughput twice).
- prefer_host_memory ON vs OFF made NO difference to these patterns on Strix
  (H3b measured); the Xclipse-motivated UMA default stays.
- A slow kernel batched by test-backend-ops perf can exceed the ~10 s
  watchdog and kill the GPU context ("context is lost... hard recovery") —
  perf-harness artifact, distinguish from a real hang (R13).
- Vulkan per-graph submit overhead: ~4.5 ms host per ~1100-node LM decode
  graph (encode + fence) vs ~5.3 ms GPU compute; graph/gallocr REBUILD cost
  is minor by comparison (LMGraphCache in the audiogen engine removed it;
  wall barely moved). Command-buffer replay would be the next lever.
- IM2COL dst-binding trap: on devices with BDA + int64, ggml_vk_op_f32 binds
  the im2col dst descriptor as a 1-BYTE DUMMY (the stock shaders write via
  buffer device address). Any new non-BDA pipeline dispatched under
  GGML_OP_IM2COL/IM2COL_3D silently writes into the dummy and produces
  garbage with no error; the real dst must be bound for such pipelines
  (see the tiled_1d exception at the IM2COL dispatch branch).
- PRE-EXISTING switch-label hazard in ggml_vk_op_f32's elements switch:
  splitting an op out of a shared case-label list silently reroutes the
  remaining labels if the new case is appended at the list tail (cost: lost
  element overrides, e.g. cpy_transpose tiles). Verify the intended branch
  fires with a print after any such split.

## HIP/ROCm on this machine

- ROCm 7.2 (HIP 7.2.53211) targets gfx1151 natively; 40 CUs. ggml-hip globs the
  whole ggml-cuda source tree (src/ggml-hip/CMakeLists.txt:63) including the
  ACE-Step kernels. gfx1151 is GGML_CUDA_CC_RDNA3_5 (common.cuh:76);
  amd_wmma_available=true (:315), amd_mfma_available=false.
- LavaSR ops (GRU, ZERO_UPSAMPLE, CHANNEL_SHUFFLE, AFFINE_PRELU) and the five
  SUPERTONIC ops have no CUDA/HIP kernels -> CPU fallback on a HIP build. Not in
  ACE-Step graphs.

## Build/runtime environment traps

- Shared-lib install prefix (~/ggml-install/<flavor>/lib) must be on
  LD_LIBRARY_PATH for audiogen binaries and ctest; missing path fails with
  "libqvac-speech-ggml-cpu.so.0: cannot open shared object file".
