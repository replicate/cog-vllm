{ pkgs, config, lib, ... }:
let
  inherit (config.cognix) cudaPackages;
  mkForce = lib.mkForce;
  mkMoreForce = lib.mkOverride 49;
in
{
  cog.build.cog_version = "0.10.0-alpha21";

  python-env.pip = {
    overridesList = [
      "pydantic>=2.0"
      "fastapi>0.99.0"
      "starlette<0.28.0"
    ];
    drvs.pydantic = {
      version = mkForce "1.10.17";
      mkDerivation.src = pkgs.fetchurl {
        sha256 = "371dcf1831f87c9e217e2b6a0c66842879a14873114ebb9d0861ab22e3b5bb1e";
        url = "https://files.pythonhosted.org/packages/ef/a6/080cace699e89a94bd4bf34e8c12821d1f05fe4d56a0742f797b231d9a40/pydantic-1.10.17-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      };
    };
    drvs.fastapi = {
      version = mkForce "0.98.0";
      mkDerivation.src = pkgs.fetchurl {
        sha256 = "f4165fb1fe3610c52cb1b8282c1480de9c34bc270f56a965aa93a884c350d605";
        url = "https://files.pythonhosted.org/packages/50/2c/6b94f191519dcc8190e78aff7bcb12c58329d1ab4c8aa11f2def9c214599/fastapi-0.98.0-py3-none-any.whl";
      };
    };
    rootDependencies = mkForce {
      cog = true;
      openai = true;
      huggingface-hub = true;
      vllm = false;
      torch = true;
    };
  };
  deps.vllm-env = config.python-env.public.extendModules {
    modules = [
      {
        _file = ./.;
        pip = {
          rootDependencies = mkMoreForce { vllm = true; };
          drvs.pydantic = {
            version = mkMoreForce "2.9.2";
            mkDerivation.src = mkMoreForce (
              pkgs.fetchurl {
                sha256 = "f048cec7b26778210e28a0459867920654d48e5e62db0958433636cde4254f12";
                url = "https://files.pythonhosted.org/packages/df/e4/ba44652d562cbf0bf320e0f3810206149c8a4e99cdbf66da82e97ab53a15/pydantic-2.9.2-py3-none-any.whl";
              }
            );
          };
          # todo starlette?
          drvs.fastapi = {
            version = mkMoreForce "0.115.0";
            mkDerivation.src = mkMoreForce (
              pkgs.fetchurl {
                sha256 = "17ea427674467486e997206a5ab25760f6b09e069f099b96f5b55a32fb6f1631";
                url = "https://files.pythonhosted.org/packages/06/ab/a1f7eed031aeb1c406a6e9d45ca04bff401c8a25a30dd0e4fd2caae767c3/fastapi-0.115.0-py3-none-any.whl";
              }
            );
          };
        };

      }
    ];
  };
  cognix.environment.PYTHON_VLLM = config.deps.vllm-env.config.public.pyEnv;

}
