{ pkgs, config, lib, ... }:
let
  cudaPackages = pkgs.cudaPackages_12_1;
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
  ];

  python-env.pip.drvs = {
    # deepspeed.env.HOME = "/tmp";
    # https://github.com/vllm-project/vllm/issues/4201
    torch.mkDerivation.postInstall = ''
      sed -i 's|caster.operator typename make_caster<T>::template cast_op_type<T>();|caster;|g' $out/${config.python-env.deps.python.sitePackages}/torch/include/pybind11/cast.h
    '';
    tbb = {
      mkDerivation.buildInputs = [ pkgs.hwloc.lib ];
      env.autoPatchelfIgnoreMissingDeps = [ "libhwloc.so.5" ]; # hwloc has 15, but not 5?
    };
    numba.mkDerivation.buildInputs = [
      config.python-env.pip.drvs.tbb.public
    ];
    vllm = { config, ... }: {
      env.CUDA_HOME = "${cudaPackages.cuda_nvcc.bin}";
      # work around 9.0a not being supported
      env.TORCH_CUDA_ARCH_LIST="8.6;9.0";
      # work around 'transformers>=4.39.1 not satisfied by version 4.40.0.dev0':
      # env.dontCheckRuntimeDeps = true;
      # cmake called from setup.py
      env.dontUseCmakeConfigure = true;

      env.autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
      env.appendRunpaths = [ "/run/opengl-driver/lib" "/usr/lib64" "$ORIGIN" ];
      mkDerivation.buildInputs = [
        cudaPackages.cuda_cudart.lib
        cudaPackages.cuda_cudart.static
        cudaPackages.cuda_cudart.dev
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
