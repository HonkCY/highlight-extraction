from train_va import training_VAModel
from train_val import training_VALModel

training_VALModel(opt_type='sgd',steps=1000,epochs=10,batch_size=4,depth='v1')