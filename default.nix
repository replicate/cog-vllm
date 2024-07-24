{ pkgs, config, lib, ... }:
let
  inherit (config.cognix) cudaPackages;
in
{
  cog.build.cog_version = "0.10.0-alpha17";

  # workaround vllm setup.py looking for cuda, doing native imports
  python-env.pip.env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    CUDA_HOME = "${cudaPackages.cuda_nvcc.bin}";
  };
  python-env.pip.overridesList = [
    "pydantic<2"
    "starlette<0.28.0"
  ];
  cognix.merge-native = {
    cublas = true;
    cudnn = "force";
  };

  python-env.pip.drvs = {
    # TODO: "cuBLAS disabled", "CUDART: not found", "CUDA Driver: Not Found", "NVRTC: Not Found"
    vllm = { config, ... }: {
      env.CUDA_HOME = "${cudaPackages.cuda_nvcc.bin}";
      # work around 9.0a not being supported
      env.TORCH_CUDA_ARCH_LIST="8.6;9.0";
      # cmake called from setup.py
      env.dontUseCmakeConfigure = true;

      # patch pydantic req
      mkDerivation.postPatch = ''
        sed -i "s/from vllm.model_executor.layers.quantization.schema import QuantParamSchema/# from vllm.model_executor.layers.quantization.schema import QuantParamSchema/" vllm/model_executor/model_loader/weight_utils.py
      '';
      mkDerivation.buildInputs = with cudaPackages; [
        # not all of these are necessary, but having extra doesn't affect runtime closure size
        cuda_cudart.lib cuda_cudart.static cuda_cudart.dev

        cudnn.lib cudnn.dev
        nccl

        cuda_nvtx.lib cuda_nvtx.dev
        cuda_nvcc.dev
        cuda_nvrtc.dev cuda_nvrtc.lib
        cuda_cccl
        libcublas.lib libcublas.dev
        libcusolver.lib libcusolver.dev
        libcusparse.lib libcusparse.dev
      ];
      mkDerivation.nativeBuildInputs = [
        cudaPackages.cuda_nvcc
        # the horrible hack:
        # setup.py calls cmake, but doesn't have the flags I need
        # but only once
        (pkgs.writeShellScriptBin "cmake" ''
        if [ -e configured ]; then
          echo "cmake already configured"
          exec ${pkgs.cmake}/bin/cmake "$@"
        fi
        touch configured
        echo "cmake flags: $cmakeFlags ''${cmakeFlagsArray[@]}" "$@"
        exec ${pkgs.cmake}/bin/cmake $cmakeFlags "''${cmakeFlagsArray[@]}" "$@"
      '') pkgs.cmake ];

      # cmake wants to fetch cutlass over git
      mkDerivation.cmakeFlags = [
        "-DFETCHCONTENT_SOURCE_DIR_CUTLASS=${config.deps.cutlass}"
      ];
      deps.cutlass = pkgs.fetchFromGitHub {
        owner = "nvidia";
        repo = "cutlass";
        rev = "v3.5.0";
        hash = "sha256-D/s7eYsa5l/mfx73tE4mnFcTQdYqGmXa9d9TCryw4e4=";
      };
    };
  };
}
