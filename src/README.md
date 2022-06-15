# Usage
`python main.py`

`python main.py -b split_inference.json`

`python main.py -s system_config.json`

`python main.py -b split_inference.json -s system_config.json`

# Components
Like almost any other machine learning project, this benchmark also has standard components like data processing pipeling, models and etc. There are some components that are uniquely designed for this project. Such components include algorithms and evaluation metrics. All components are tied up together with the interface.
## Benchmark specification
There are two separate configuration files that are used to specify a benchmark. One is a system level configuration file used for specifying file paths and other infrastructure related information such as GPUs. The benchmark config consists of benchmark specific and system independent parameters. The `method` key indicates which technique is used to run the benchmark. The `exp_keys` key is a list of multiple nested keys that is used to have dynamic control on the experiments folder name which could be useful for experiments logging and comparison using either logs or tensorboard.
There are three different modes in which these experiments can be performed - "defense", "challenge", "attack". This has to be specified in the key `experiment_type`. By default, the `experiment_type` is set as "defense".
## Models
Currently we support different families of ResNet for all defense mechanisms with varying `split_layer`.

## Datasets
1. CelebA
2. FairFace
3. CIFAR10
4. UTKFace
5. Labeled Faces in the Wild (LFW)


## Algorithms
Currently supported algorithms on the defense side include. These algorithms work for variable number of client side `split_layer`.
1. Split Inference `split_inference.json`
2. Uniform noise (currently supports Laplace and Gaussian distribution) `uniform_noise.json`
3. NoPeek (Face and Gesture'21)`nopeek.json`
4. Siamese Embedding (IEEE Internet of Things Journal' 20) `siamese_embedding.json`
5. PCA Embedding `pca_embedding.json`
6. PAN (ACM Interactive, Mobile, Wearable and Ubiquitous Technologies'19) `pan.json`
7. Complex neural networks (ICLR'20) `complex_nn.json`
8. Deep Obfuscator (ACM IOTDI'21) `deep_obfuscator.json`
9. Gaussian Blur (ECCV'16) `gaussian_blur.json`
10. Linear Correlation `linear_correlation.json`
11. Adversarially Learned Representations for Information Obfuscation and Inference (ICML'19) `aioi.json`
12. Cloak (WWW'21) `cloak.json`
13. DISCO (CVPR'21) `disco.json`
14. Shredder (ACM Architectural Support for Programming Languages and Operating Systems'20) `shredder.json`

1. Decoder Attack `decoder_attack.json`
2. Input Optimization Attack `input_optimization_attack.json`
3. Model Optimization Attack `input_model_optimization_attack.json`
4. Discriminator Attack `discriminator_attack.json`

## Config Files
There are primarily 2 config files -> system configs (-s) and experiment config (-b)

#### Experiment config
Here is a typical example of an experiment config with various parameters that can be adjusted based on the benchmark.
`method` : `split_inference`,   
`experiment_type` : We can use one of "attack", "defence" and "challenge" based on the type of experiment.  
`client`:  
      {"model_name": "resnet34", #replace this with any available model - restnet18, resnet50 etc.  
      "split_layer": 6, # change the number here to determine how many layers remain with the client.  
      "pretrained": false,   
      "optimizer": "adam",  
      "lr": 3e-4},     
`server`:   
      {"model_name": "resnet34",  
      "split_layer":6,  
      "logits": 2, # update this based on the dataset used.  
      "pretrained": false,  
      "lr": 3e-4,  
      "optimizer": "adam"},      
 `learning_rate`: 0.01,  
 `total_epochs`: 150,  
 `training_batch_size`: 128,  
 `dataset`: "cifar10",  
 `protected_attribute`: "data", # here the entire data is considered private, that is, we are looking at data reconstruction attacks. We can also assign other classes as private/protected attribute as well.  
 `prediction_attribute`: "class",  
 `exp_id`: "1",  
 `img_size`: 128,  
 `split`: false,  
 `test_batch_size`: 64,  
 `exp_keys`: []. 

### Writing your own algorithm
Most of the code in the algos is modular enough for a user to only focus on writing the important part of the mechanism and rest all functions automatically. Implementor of a mechanism just needs to inherit the `SimbaDefense` class from `algos.simba_algo`. A user can also build upon existing mechanisms by inheriting them and overriding a particular function of the algorithm.
## Evaluation
Evaluation involves two category of metrics. One category of metric is specific to the algorithm and the other category of metric is generic for all of the algorithms and used for benchmarking.
## Scheduler
