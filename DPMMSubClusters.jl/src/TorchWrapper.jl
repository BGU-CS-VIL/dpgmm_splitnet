
module TorchWrapper
using PyCall
import YAML

export torch_model

global const torch_model_2d = PyNULL()
global const torch_model_10d = PyNULL()
global const torch_model_128d = PyNULL()

function __init__()
    scriptdir = "../splitnet"
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    torch_wrapper = pyimport("JMWrapper")


    cfg = YAML.load_file("./configs/basic.yaml")
    ckpt_path_2d = cfg["models"]["2d_model_path"]
    ckpt_path_10d = cfg["models"]["10d_model_path"]
    ckpt_path_128d = cfg["models"]["20d_model_path"]


    torch_model_1 = torch_wrapper.JuliaModelWrapper(ckpt_path_2d, gpu=true)
    copy!(torch_model_2d, torch_model_1)

    torch_model_2 = torch_wrapper.JuliaModelWrapper(ckpt_path_10d, gpu=true)
    copy!(torch_model_10d, torch_model_2)
    
    torch_model_3 = torch_wrapper.JuliaModelWrapper(ckpt_path_128d, gpu=true)
    copy!(torch_model_128d, torch_model_3)
end
    

end