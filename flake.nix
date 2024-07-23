{
  inputs = {
    cognix.url = "github:datakami/cognix/24.07";
  };

  outputs = { self, cognix }@inputs: cognix.lib.singleCognixFlake inputs "cognix-vllm";
}
