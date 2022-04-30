 Federated Learning Simulator
===
PyTorch implementation for federated learning simulation and evaluation. The simultor aims to evaluate federated learning proposals.

# Prerequisites

To install all the prerequisites, you can use the following command:

~~~
pip install requirements.txt 
~~~

You also can install prerequisites via Anaconda

~~~
conda env create -f fl_simulator_env.yaml
~~~

The command for the global training:
~~~
python federated_train.py --cuda_visible_device=1 --method=FedAGM --global_method=FedAGM --config=configs/cifar_actl2.yaml --arch=ResNet18 --weight_decay=1e-3 --gr_clipping_max_norm=10 --global_epochs=3 --alpha=1 --mu=0.001 --gamma=0.9 --momentum=0.0 --tau 1.0 --lr=1e-1 --mode=dirichlet --dirichlet_alpha=0.3  --participation_rate=0.05 --learning_rate_decay 0.995 --entity=helioncneto --set CICIDS2017
~~~
