# Optimize your model with Torch-TensorRT (optional → I used Vertex Notebook)

- Download NGC CLI
    - https://ngc.nvidia.com/setup/installers/cli
- Sign up for NGC account
- Login with API key
    - https://ngc.nvidia.com/setup/api-key
- Pull the docker container
    - https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

```
mkdir test
docker run -it --gpus all -v ${PWD}:/test nvcr.io/nvidia/pytorch:<23.01>-py3
cd /test
```

*Remove `gpus` flag if no gpu

<br/><br/>
# Optimize Model and Download

- Using Vertex AI notebook to convert the models

```
%%bash
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 torch-tensorrt tensorrt
```

```
import torch
import torchvision
import torch_tensorrt

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
```

```
# Load pretrained model

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()
```

```
# Save original FP32 model
torch.save(model.state_dict(), "resnet50_pretrained.pth")
```

```
# Compile and save with Torch TensorRT and save

trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float16)],
    enabled_precisions = {torch.half}, # Run with FP16
    workspace_size = 1 << 22
)

torch.jit.save(trt_model_fp16, "model.pt")
```
<br/><br/>

# Create Model Repository


- Create `model_repository`

```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.pt
```

`config.pbtxt`

```
name: "resnet50"
platform: "pytorch_libtorch"
max_batch_size : 0
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
    reshape { shape: [ 1, 3, 224, 224 ] }
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1, 1000 ,1, 1]
    reshape { shape: [ 1, 1000 ] }
  }
]
```

- Run server (CPU)

```
$ docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
```
<br/><br/>
# Building a Triton Client to Query the Server

- Get an example image
    
    ```
    wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
    ```
    
- Install packages

```
pip3 install torchvision attrdict nvidia-pyindex tritonclient
```

- Write a small preprocessing step

```
import numpy as np
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

# preprocessing function
def rn50_preprocess(img_path="img1.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).numpy()

transformed_img = rn50_preprocess()
```

```
# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")
```

```
inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=1000)
```

```
# Querying the server
results = client.infer(model_name="resnet50", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('output__0')
print(inference_output[:5])
```