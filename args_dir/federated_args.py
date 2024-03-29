import argparse
import sys
import yaml
#from configs import parser as _parser


USABLE_TYPES = set([float, int])


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def produce_override_string(args, override_args):
    lines = []
    for v in override_args:
        if v != "multigpu":
            v_arg = getattr(args, v)
            if type(v_arg) in USABLE_TYPES:
                lines.append(v + ": " + str(v_arg))
            else:
                lines.append(v + ": " + f'"{str(v_arg)}"')
        else:
            lines.append("multigpu: " + str(args.multigpu))

    return "\n# ===== Overrided ===== #\n" + "\n".join(lines)


args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--mode", help="data setting to use", default="iid"
    )
    parser.add_argument(
        "--data_unbalanced", action='store_true', help="create unbalanced client data"#
    )
    parser.add_argument(
    "--t_sne", action='store_true', help="save t_sne graph"
    )
    parser.add_argument(
    "--umap", action='store_true', help="save umap"
    )
    parser.add_argument(
        "--umap_dim", default=2, type=int, help="umap figure dimensions "
    )
    parser.add_argument(
        "--project", help="data setting to use", default="federated_learning"
    )
    parser.add_argument(
        "--method", help="method to use", default="Fedavg"
        #'Fedavg''l2_act_reg''FedProx''PodNet'
    )  
    parser.add_argument(
        "--loss", help="loss to use", default="CE"
        #'CE','IL'
    )  
    parser.add_argument(
    "--return_feature", action='store_true', help="Do you need to make model forward return feature?"
    )
    parser.add_argument(
    "--hard_aug", action='store_true', help="use hard data augmentation?"
    )    
    
    parser.add_argument(
        "--global_method", help="aggregation method to use", default="base_avg"
    )
    parser.add_argument(
        "--additional_experiment_name", help="specific information for logging", default=""
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training."
    )
    parser.add_argument(
        "--temperature", default=20.0, type=float, help="Temperature for selection on blocked participants."
    )
    parser.add_argument(
        "--cool", default=0.97, type=float, help="Temperature reduction factor."
    )
    parser.add_argument(
        "--malicious_rate", default=0, type=float, help="Malicious participants rate. 0 for no malicious participants."
    )
    parser.add_argument(
        "--malicious_proba", default=1, type=float, help="Probability of being malicious."
    )
    parser.add_argument(
        "--malicious_aggregation", default=0, type=int, help="Which aggregation round the participant starts the malicious behavior."
    )
    parser.add_argument(
        "--num_fake_data", default=1500000, type=int, help="Number of fake samples."
    )

    parser.add_argument(
        "--malicious_type",
        default="random",
        type=str,
        metavar="N",
        help="Name of types of malicious participants",
    )

    parser.add_argument(
        "--mal_epsilon",
        default=0.5,
        type=float,
        metavar="N",
        help="The epsilon for FGSM attack",
    )
    parser.add_argument(
        "--mal_theta",
        default=1.0,
        type=float,
        metavar="N",
        help="The tetha for JSMA attack",
    )

    parser.add_argument(
        "--mal_gamma",
        default=0.1,
        type=float,
        metavar="N",
        help="The gamma for JSMA attack",
    )

    parser.add_argument(
        "--mal_alpha",
        default=0.1,
        type=float,
        metavar="N",
        help="The alpha for PGD attack",
    )

    parser.add_argument(
        "--cuda_device_id",
        default=0,
        type=float,
        metavar="N",
        help="which GPU to use",
    )
    parser.add_argument(
        "--train_on_gpu",
        default=True,
        type=bool,
        help="Name of cuda device if available (default: cuda:0)",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )

    parser.add_argument(
        "--create_client_dataset",
        action="store_true",
        help="Whether or not to make new client dataset",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Chose if you will sync with Weights and Bias",
    )
    parser.add_argument(
        "--preserve_checkpoint",
        default=False,
        action="store_true",
        help="Preserve the checkpoint file in the end of the training.",
    )
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Active if you want to use a checkpoint",
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )

    ## Training Config
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--global_epochs",
        default=1000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--warmup_epoch",
        default=1,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--centralized_epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--local_epochs",
        default=5,
        type=int,
        metavar="N",
        help="number of local epochs to run on each global epoch",
    )
    parser.add_argument(
        "--image_epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of epochs for image gradient ascent",
    )

    parser.add_argument(
        "--num_of_clients",
        default=100,
        type=int,
        metavar="N",
        help="number of total clients",
    )

    parser.add_argument(
        "--epsilon_greedy",
        default=0.5,
        type=float,
        metavar="N",
        help="epsilon greedy value",
    )

    parser.add_argument(
        "--last_epsilon_greedy",
        default=0.01,
        type=float,
        metavar="N",
        help="The last epsilon greedy valuie",
    )

    parser.add_argument(
        "--participation_rate",
        default=0.1,
        type=float,
        metavar="N",
        help="number of participation rate at each global epoch",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=50,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument(
        "--momentum", default=0.0, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.0)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--ld",
        "--learning_rate_decay",
        default=1.0,
        type=float,
        metavar="LD",
        help="learning rate decay (default: 1.0)",
        dest="learning_rate_decay",
    )
    parser.add_argument(
        "--gr_clipping_max_norm",
        default=10,
        type=float,
        metavar="N",
        help="gradient clippling max norm",
    )

    parser.add_argument(
        "--eval_path",
        default=".",
        type=str,
        metavar="N",
        help="gradient clippling max norm",
    )

    ## Dataset
    parser.add_argument("--set", help="name of dataset", type=str, default="CIFAR10")
    parser.add_argument(
        "--dirichlet_alpha",
        default=-0.6,
        type=float,
        metavar="N",
        help="federated dirichlet alpha",
    )

    parser.add_argument(
        "--alpha_mul_epoch",
        action="store_true",
        help="Whether or not to multiplicate alpha with epoch",
    )
    parser.add_argument(
        "--alpha_divide_epoch",
        action="store_true",
        help="Whether or not to divide alpha by epoch",
    )
    parser.add_argument(
        "--data", help="path to dataset base directory", default="./data"
    )
    parser.add_argument(
        "--client_data", help="path to client dataset base directory", default="./client_data"
    )
    parser.add_argument("--num-classes", default=10, type=int)



    ## Eval

    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,
 
        help="print frequency (default: 1)",
    )
    parser.add_argument(
        
        "--t_sne_freq",
        default=1,
        type=int,
 
        help="print t_sne graph frequency (default: 1)",
    )
    parser.add_argument(
        
        "--umap_freq",
        default=1,
        type=int,
 
        help="print umap graph frequency (default: 1)",
    )

    ### Method Hyperparameters

    ## l2-act-reg
    parser.add_argument(
        "--alpha",
        default=0.0,
        type=float,

        help="federated activation regularization hyperparameter alpha",
    )
    ## pod
    parser.add_argument(
        "--pod_normalize",
        action="store_false",
        help="Whether or not to normalize activation in pod",
    )
    parser.add_argument(
        "--collapse_channels",
        help="Whether or not to normalize activation in pod",
        default='spatial'
        ####"channels" "width" "height" "gap" "spatial","pixel"
    )

    parser.add_argument(
        "--regularization_unit",
        help="regularization layer unit",
	type=str,
        default='Conv'
        ####"Conv",'Block',"BasicBlock',"BottleBlock'
    )    
    parser.add_argument(
        "--knowledge_temperature",
        default=1,
        type=float,
        metavar="N",
        help="knowledge_temperature",
    )        
    parser.add_argument(
        "--lambda1",
        default=1,
        type=float,
        metavar="N",
        help="intermediate activation loss weight",
    )  
    parser.add_argument(
        "--lambda2",
        default=1,
        type=float,
        metavar="N",
        help="last activation loss weight",
    )    
    parser.add_argument(
        "--lambda3",
        default=1,
        type=float,
        metavar="N",
        help="logit loss weight",
    )    
    ## l2-weight
    parser.add_argument(
        "--mu",
        default=0.01,
        type=float,
        metavar="N",
        help="federated weight regularization hyperparameter mu",
    )
    
    ## FedAdam
    parser.add_argument(
        "--beta_1",
        
        default=0.9,
        type=float,
        metavar="N",
        help="FedAdam beta_1",
        
    )
    parser.add_argument(
        "--beta_2",
        
        default=0.99,
        type=float,
        metavar="N",
        help="FedAdam beta_2",
        
    )
    parser.add_argument(
        "--eta",
        
        default=1e-2,
        type=float,
        metavar="N",
        help="FedAdam server learning rate",
        
    )
    parser.add_argument(
        "--server_learning_rate_decay",
        
        default=1,
        type=float,
        metavar="N",
        help="FedAdam server learning rate decay",
        
    )    
    parser.add_argument(
        "--tau",
        
        default=0.2,
        type=float,
        metavar="N",
        help="FedAdam beta_2",
        
    )

    ## FedCM
    parser.add_argument(
        "--g_lr",

        default=1.0,
        type=float,
        metavar="N",
        help="global learning rate",

    )

    parser.add_argument(
        "--lamb",

        default=0.1,
        type=float,
        metavar="N",
        help="weight of local gradient for local update",

    )

    ## Proxy Regularization

    parser.add_argument(
        "--temp",
        default=0.2,
        type=float,
        metavar="N",
        help="temperature",

    )
    parser.add_argument(
        "--temp2",
        default=1,
        type=float,
        metavar="N",
        help="temperature",

    )
    parser.add_argument(
        "--l2_norm",
        action="store_true",
        help="l2_norm classifier",
    )


    ## ETC

    parser.add_argument(
        "--epsilon",
        default=0.0,
        type=float,
        metavar="N",
        help="Used in LSC Loss",
    )

    
    ##proto
    parser.add_argument(
    "--num_img",
    default=10,
    type=int,

    help="num of common generated images",
    )
    
    parser.add_argument(
    "--alpha_proto",
    default=0.0,
    type=float,

    help="federated generated image activation regularization hyperparameter alpha_proto",
    )
    parser.add_argument(
    "--cealpha_proto",
    default=1.0,
    type=float,

    help="federated ce loss ratio hyperparameter cealpha_proto",
    )
    parser.add_argument(
        "--alpha_proto_mul_epoch",
        action="store_true",
        help="Whether or not to multiplicate alpha_proto with epoch",
    )
    parser.add_argument(
        "--alpha_proto_divide_epoch",
        action="store_true",
        help="Whether or not to divide alpha_proto by epoch",
    )
    ## byol
    parser.add_argument('--rampup_length', default=200, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=5)

    ## FedCM
    parser.add_argument(
        "--p_lr_beta",

        default=1.0,
        type=float,
        metavar="N",
        help="global learning rate",

    )

    ##IL
    parser.add_argument(
    "--abs_thres",
    action="store_true",
    help="Whether or not to use absolute threshold",
)
    parser.add_argument(
    "--thres",
    default=0.5,
    type=float,

    help="Threshold of IL",
    )


    ## Train Generator
    parser.add_argument(
        "--g_temp",
        default=1e1,
        type=float,
        metavar="N",
        help="generator prior loss",
    )

    parser.add_argument(
        "--g1",
        default=1,
        type=float,
        metavar="N",
        help="generator ce",
    )

    parser.add_argument(
        "--train_size",
        default=0.9,
        type=float,
        help="The size of the train set",
    )
    parser.add_argument(
        "--test_size",
        default=0.05,
        type=float,
        help="The size of the test set",
    )
    parser.add_argument(
        "--val_size",
        default=0.05,
        type=float,
        help="The size of the val set",
    )

    parser.add_argument(
        "--g2",
        default=1,
        type=float,
        metavar="N",
        help="generator information loss",
    )

    parser.add_argument(
        "--g3",
        default=1e-3,
        type=float,
        metavar="N",
        help="generator prior loss",
    )
    
    ###

    parser.add_argument(
        "--gamma",
        default=0.8,
        type=float,
        metavar="N",
        help="weight for nestrov",
    )

    parser.add_argument(
        "--beta",
        default=0.0,
        type=float,
        metavar="N",
        help="weight for server momentum",
    )

    parser.add_argument(
        "--warmup", default=0, type=int, help="Number of warmup epoch"
    )

    ###DYN

    parser.add_argument(
        "--only_ce",
        default=False,
        action="store_true",
        help="use only ce_loss for FedDyn",
    )
    parser.add_argument(
        "--only_linear",
        default=False,
        action="store_true",
        help="use only Linear term with ce_loss",
    )
    parser.add_argument(
        "--no_sm",
        default=False,
        action="store_true",
        help="no server momentum for FedDyn",
    )
    parser.add_argument(
        "--no_os",
        default=False,
        action="store_true",
        help="no overshooting for FedDyn",
    )

    parser.add_argument(
        "--analysis",
        default=False,
        action="store_true",
        help="Turn on analysis mode",
    )
    parser.add_argument(
        
        "--compare_with_center",
        default=0,
        type=int,
 
        help="compare with center update vs FL update (0:Don't compare,1:center update can access to whole dataset,2:center update can access to union of participated clients)",
    )
    parser.add_argument(

        "--entity",
        default="",
        type=str,

        help="The user or team of wandb",
    )

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from: {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    #global args
    #if args is None:
    #    args = parse_arguments()
    return parse_arguments()


#run_args()
