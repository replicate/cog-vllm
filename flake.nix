{
  inputs = {
    cognix.url = "github:datakami/cognix/yorickvp/uv";
  };

  outputs = { self, cognix }@inputs: cognix.lib.singleCognixFlake inputs "cognix-vllm";
}
