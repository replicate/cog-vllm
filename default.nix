{ pkgs, config, lib, ... }:
let
  cudaPackages = pkgs.cudaPackages_12_1;
in
{
  # todo: give this a place in cog.yaml
  python-env.pip.overrides = [
    "cog @ http://r2.drysys.workers.dev/tmp/cog-0.10.0a6-py3-none-any.whl"
    "pydantic==1.10.9"
  ];
  python-env.pip.uv.enable = true;
  # workaround setup.py looking for cuda, doing native imports
  python-env.pip.env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    CUDA_HOME = "${cudaPackages.cuda_nvcc.bin}";
  };
  # workaround uv not having git hashes yet
  python-env.deps.fetchgit = {
    url, rev, sha256
  }: builtins.fetchGit { inherit url rev; allRefs = true; };

  python-env.pip.drvs = {
    numba.env.autoPatchelfIgnoreMissingDeps = [ "libtbb.so.12" ];
    deepspeed.env.HOME = "/tmp";
    # https://github.com/vllm-project/vllm/issues/4201
    torch.mkDerivation.postInstall = ''
      sed -i 's|caster.operator typename make_caster<T>::template cast_op_type<T>();|caster;|g' $out/lib/python3.10/site-packages/torch/include/pybind11/cast.h
    '';
    vllm = {
      env.CUDA_HOME = "${cudaPackages.cuda_nvcc.bin}";
      # work around 9.0a not being supported
      env.TORCH_CUDA_ARCH_LIST="9.0";
      # work around 'transformers>=4.39.1 not satisfied by version 4.40.0.dev0':
      env.dontCheckRuntimeDeps = true;
      # cmake called from setup.py
      env.dontUseCmakeConfigure = true;

      env.autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
      env.appendRunpaths = [ "/run/opengl-driver/lib" "/usr/lib64" "$ORIGIN" ];
      mkDerivation.buildInputs = [
        cudaPackages.cuda_cudart.lib
        cudaPackages.cuda_cudart.static
      ];
      mkDerivation.nativeBuildInputs = [
        # todo: replace with modular cuda
        cudaPackages.cudatoolkit
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
    };
    # https://github.com/vllm-project/vllm/issues/4360
    vllm-nccl-cu12 = { config, ... }: {
      env.HOME = "/tmp";

      mkDerivation.postUnpack = ''
        mkdir -p /tmp/.config/vllm/nccl/cu12
        cp ${config.deps.vllm-nccl} /tmp/.config/vllm/nccl/cu12/libnccl.so.2.18.1
      '';
      deps.vllm-nccl = pkgs.fetchurl {
        url = "https://github.com/vllm-project/vllm-nccl/releases/download/v0.1.0/cu12-libnccl.so.2.18.1";
        hash = "sha256-AFUDtiuf5Ga2svYLoq0dqkVxGW/8uCo9d6PHdb/NWsg=";
      };
    };
  };
}
