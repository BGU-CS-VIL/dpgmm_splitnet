#Data Loading specifics
data_path = "/path/to/data/"
data_prefix = "data_prefix"  #If the data file name is bob.npy, this should be 'bob'


#Model Parameters
iterations = 100
hard_clustering = false  #Soft or hard assignments
initial_clusters = 1
argmax_sample_stop = 5 #Change to hard assignment from soft at iterations - argmax_sample_stop
split_stop = 5#Stop split/merge moves at  iterations - split_stop

random_seed = nothing #When nothing, a random seed will be used.

max_split_iter = 20
burnout_period = 20
max_clusters = Inf

#Model hyperparams
α = 10.0 #Concetration Parameter
hyper_params = niw_hyperparams(1.0, zeros(Float32, 2), 5, Matrix{Float32}(I, 2, 2) * 1.0)


outlier_mod = 0.05 #Concetration Parameter
outlier_hyper_params =
    niw_hyperparams(1.0, zeros(Float32, 2), 5, Matrix{Float32}(I, 2, 2) * 1.0)


 # "Learning To Propose" Params:
# smart_init_type = "splitnet"


#Saving specifics:
enable_saving = true
model_save_interval = 1000
save_path = "/path/to/save/dir/"
overwrite_prec = false
save_file_prefix = "checkpoint_"
