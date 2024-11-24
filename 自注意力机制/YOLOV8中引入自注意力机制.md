深度学习中的注意力机制`（Attention Mechanism）`和人类视觉的注意力机制类似，就是在众多信息中把注意力集中放在重要的点上，选出关键信息，而忽略其他不重要的信息

## SEAM简介

`SEAM（Self-Ensembling Attention Mechanism）`是一种自集成注意力机制，通过多视角特征融合和一致性正则化来增强模型的鲁棒性和泛化能力，特别适用于**处理遮挡问题**和**多尺度特征融合**。`SEAM` 模块的核心目标是通过增强未遮挡区域的特征响应，来弥补遮挡区域的响应损失。

## 操作步骤

### seam.py

我们将注意力机制模块的代码文件存放在`ultralytics-main/ultralytics/nn/modules`路径下，新建文件`seam.py`，复制粘贴下方代码。



```Python
import torch
import torch.nn as nn

__all__ = ['SEAM', 'MultiSEAM']


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SEAM(nn.Module):
    def __init__(self, c1, n=1, reduction=16):
        super(SEAM, self).__init__()
        c2 = c1
        self.DCovN = nn.Sequential(
            # nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1),
            # nn.GELU(),
            # nn.BatchNorm2d(c2),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)


def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.SiLU(),
                nn.BatchNorm2d(c2)
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return dcovn


class MultiSEAM(nn.Module):
    def __init__(self, c1, depth=1, kernel_size=3, patch_size=[3, 5, 7], reduction=16):
        super(MultiSEAM, self).__init__()
        c2 = c1
        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)
```




![image.png](https://tc-cdn.flowus.cn/oss/bcc1d184-f36e-4989-a360-4081980783b5/image.png?time=1732430700&token=eeb9ac1e9ea9ed8941489df8e3867e0d4565b1630d39ecab98f9032d11ee9d58&role=free)



### tasks.py

打开`ultralytics-main/ultralytics/nn/tasks.py`中的`tasks.py`，添加代码进行导入。

```Python
from .modules.seam import SEAM, MultiSEAM
```


![image.png](https://tc-cdn.flowus.cn/oss/ca50bac2-d2ff-49e8-93ca-cc6fd6d590a0/image.png?time=1732430700&token=9c13c131848607f02a0304e6ecba7deacb22e911677c137d2c0bf3d6429dd13d&role=free)

继续增加代码，在如图所示位置，可直接`ctrl+f`检索`elif m in` 快速定位

```Python
        elif m in {SEAM,MultiSEAM}:
            c2 = ch[f]
            args = [c2,*args]
```


![image.png](https://tc-cdn.flowus.cn/oss/9a77fdee-33f2-4739-b680-0a5d54dbb2eb/image.png?time=1732430700&token=6a2969deed61cbd1db44b9ab06e7adddb708f367785ef0930608ccae7375f1d0&role=free)



### yolov8n_SEAM.yaml

本例中，是将`SEAM`添加到了`head`中，添加到不同的网络效果不同，同时注意层数的变化。

在`ultralytics-main/ultralytics/cfg/models/v8`下新建`yaml`文件命名为`yolov8n_SEAM.yaml`（根据自己所需网络的大小更换`n、s、l`），将以下内容复制进去，自行修改`nc`、`names`、`scales`。

```Python
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 1  # number of classes
#train: "E:/github/ultralytics-main/datasets2/images/train"
#val: "E:/github/ultralytics-main/datasets2/images/val"
names: ["things"]
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOP

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)
  - [-1, 1, SEAM, []]  # 16

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 19 (P4/16-medium)
  - [-1, 1, SEAM, []]  # 20

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 23 (P5/32-large)
  - [-1, 1, SEAM, []]  # 24

  - [[16, 20, 24], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```




![image.png](https://tc-cdn.flowus.cn/oss/7162da5a-e547-41df-b84a-b0fb8343f520/image.png?time=1732430700&token=7eca364a40ec889b84a4fa9300c09ff5255867483d3e9aa067697daafa93c96e&role=free)

下面是结构的对比

![image.png](https://tc-cdn.flowus.cn/oss/208c0f2a-5e05-4788-996f-11b83475d2b9/image.png?time=1732430700&token=4daaa31018c0312d972fb62449c8c50abba36079d32b7b99276eaa449e804818&role=free)

以上准备工作完成后，填写好训练代码中`yaml`的路径（`data`和`model`的）以及`bt`文件，点击`train.py`就可以正常训练了。

但是，我却遇到了不少问题。

## 运行与错误处理

### 未引入

点击`train.py`，在命令行界面可以看到，`seam`模块并未引入。

![image.png](https://tc-cdn.flowus.cn/oss/50201f2a-0150-4c9e-bd37-f27e55035073/image.png?time=1732430700&token=126f3563e5b9279120302c2494e300b892156c5ac5b969ef901e91403f553a48&role=free)

重新检查一遍前面的配置发现没有错误，又查看了`train.py`后发现`train.py`中`model` 和 `data`的配置文件设置的有误，没有正确传给`model`

> 这源于我最初训练用的代码传递参数设置不合理，起初我是在`model`文件夹下的`yolov8_SEAM.yaml`文件中直接配置了数据的路径，将其传递到了`data`。这样使用我原来的错误代码就会导致，在引入注意力机制等外部模块时，由于没有用到`model`而不能成功导入，从而一直使用的还是默认的模型结构。

错误代码

```Python
from ultralytics import YOLO
#疑问 模型配置文件的作用是什么，我这里yaml是包含了数据配置和模型配置吗
if __name__ == '__main__':
    model = YOLO("yolov8l.pt")
    model.train(data=r"E:\github\ultralytics-main\ultralytics\cfg\models\v8\yolov8_SEAM2.yaml",
                epochs=50,
                batch=4,
                workers=0,
                imgsz=640)
```


更换训练代码，将`model`和`data`分别传入。

```Python
#coding:utf-8
from ultralytics import YOLO

# 加载预训练模型
# 添加注意力机制，yolov8_pinecone_SEAM2.yaml 默认使用的是n。
model = YOLO(r"E:\github\ultralytics-main\ultralytics\cfg\models\v8\yolov8l_SEAM.yaml").load('yolov8l.pt')

# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data=r"E:\github\ultralytics-main\ultralytics\cfg\models\v8\yolov8l_SEAM.yaml", epochs=50, batch=4)  # 训练模型
    # 将模型转为onnx格式
    # success = model.export(format='onnx')

```




这里提供给大家一个训练的模板代码

```Python
#coding:utf-8
from ultralytics import YOLO

# 加载预训练模型
# 添加注意力机制，yolov8.yaml 默认使用的是n
#选择你的yaml文件、模型大小
model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml").load('yolov8n.pt')

# Use the model
if __name__ == '__main__':
    # Use the model 加载你的数据
    results = model.train(data='datasets/MyData/data.yaml', epochs=250, batch=4，workers=0)  # 训练模型
    # 将模型转为onnx格式
    # success = model.export(format='onnx')
```




### yaml命名问题

一波三折，再次启动发现模块已经成功导入，但是发生报错

```Python
WARNING ⚠️ no model scale passed. Assuming scale='n'. Transferred 35/403 items from pretrained weights
```


解决办法，二者选一个

- 在`yaml`文件里面的`scales`里面写上`n,s,m,l`

- `yaml`文件命名里面加上`n,s,m,l`…，不写的话默认用n模型，例如`yolov8n_SEAM.yaml`

然后，运行`train.py`即可，`SEAM`成功导入，也开始训练啦

![image.png](https://tc-cdn.flowus.cn/oss/79cf4873-bd02-4450-a199-db8c1a3de7a1/image.png?time=1732430700&token=ec7846ea8fcde70adb61d67c9f8b67d795233680e5bd18db75c915618f5772ec&role=free)

![image.png](https://tc-cdn.flowus.cn/oss/7c2b012f-7e30-4105-8d4d-0ead518184d0/image.png?time=1732430700&token=295b93ef38804ac3839e1030622add352df4d9ff7ad227400541d2916ed366a6&role=free)



### 传递问题

在训练过程中，出现了以下内容。

```Python
Transferred 402/643 items from pretrained weights
```


搜索过后，从官方的issues中得到答案

![image.png](https://tc-cdn.flowus.cn/oss/5e3ce7c8-8d7e-4e78-a106-8b6db734b789/image.png?time=1732430700&token=9f6cfc93628e7fa545d977487780d991db4664a71f3e012875bc3d98ecf852e0&role=free)

> 如果数据中的类数与预训练权重不同，则并非所有参数组都可以传输。这种大小的差异可能会导致某些权重未使用且未转移。

![image.png](https://tc-cdn.flowus.cn/oss/e6c6f8a5-849c-4ec1-96d8-f08f8a1fe37b/image.png?time=1732430700&token=a7ac0a3f6faa445b6a958f20017b291b1ccff7ac839e5ff56ad3fa501a4bf00c&role=free)


> 参考链接：https://github.com/ultralytics/ultralytics/issues/2741

由于公众号文章无法大范围多次修改，文章同步更新到了我的github上






