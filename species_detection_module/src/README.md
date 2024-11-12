# Training the model

The `train.py` contains the training code for YOLOv11 classification model. The training code initializes the training progress and stores the run in 
`runs/classify/train1`. 

``` bash
model.train(
    **args
)
```

The train function takes in different arguments for training the model like   `batch=16`, `epochs=10` etc.
