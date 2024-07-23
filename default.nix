{ pkgs, config, lib, ... }:
let
  cudaPackages = pkgs.cudaPackages_12_1;
  python3 = config.python-env.deps.python;
in
{
  cog.build.cog_version = "0.10.0-alpha17";
  python-env.pip.uv.enable = true;
  # workaround setup.py looking for cuda, doing native imports
  python-env.pip.env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    CUDA_HOME = "${cudaPackages.cuda_nvcc.bin}";
  };
  # workaround uv not having git hashes yet
  python-env.deps.fetchgit = {
    url, rev, sha256
  }: builtins.fetchGit { inherit url rev; }; # allRefs = true; };

  python-env.pip.overridesList = [
    "pydantic<2"
    "starlette<0.28.0"
    "nvidia-cudnn-cu12==${cudaPackages.cudnn.version}"
  ];
  python-env.pip.constraintsList = [
    # "nvidia-cudnn-cu12==${cudaPackages.cudnn.version}"
    "nvidia-cublas-cu12==${cudaPackages.libcublas.version}"
  ];

  python-env.pip.drvs = {
    # https://github.com/vllm-project/vllm/issues/4201
    # https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/python-modules/torch/fix-cmake-cuda-toolkit.patch
    torch.mkDerivation.postInstall = ''
      sed -i 's|caster.operator typename make_caster<T>::template cast_op_type<T>();|caster;|g' $out/${python3.sitePackages}/torch/include/pybind11/cast.h
      rm $out/${python3.sitePackages}/torch/share/cmake/Caffe2/FindCUDAToolkit.cmake
    '';
    # TODO: should this have hwloc (and with enableCUDA)?
    numba.mkDerivation.buildInputs = [ pkgs.tbb_2021_11 ];
    # TODO: "cuBLAS disabled", "CUDART: not found", "CUDA Driver: Not Found", "NVRTC: Not Found"
    vllm = { config, ... }: {
      env.CUDA_HOME = "${cudaPackages.cuda_nvcc.bin}";
      # work around 9.0a not being supported
      env.TORCH_CUDA_ARCH_LIST="8.6;9.0";
      # cmake called from setup.py
      env.dontUseCmakeConfigure = true;

      env.autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
      env.appendRunpaths = [ "/run/opengl-driver/lib" "/usr/lib64" "$ORIGIN" ];
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
        cuda_cudart
        cuda_nvcc.dev
        cuda_nvrtc.dev cuda_nvrtc.lib
        cuda_nvml_dev.lib cuda_nvml_dev.dev
        cuda_cccl
        libcublas.lib libcublas.dev
        libcurand.dev
        cuda_profiler_api
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

    # patch in cuda packages from nixpkgs
    nvidia-cublas-cu12.mkDerivation.postInstall = ''
      pushd $out/${python3.sitePackages}/nvidia/cublas/lib
      for f in ./*.so.12; do
        chmod +w "$f"
        rm $f
        ln -s ${cudaPackages.libcublas.lib}/lib/$f ./$f
      done
      popd
    '';
    nvidia-cudnn-cu12.mkDerivation.postInstall = ''
      pushd $out/${python3.sitePackages}/nvidia/cudnn/lib
      for f in ./*.so.8; do
        chmod +w "$f"
        rm $f
        ln -s ${cudaPackages.cudnn.lib}/lib/$f ./$f
      done
      popd
    '';
  };
}
