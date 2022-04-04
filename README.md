referenced:https://github.com/ctallec/world-models

original files(basically won't used): 

	data/carracing.py
	
	data/gene_data.py
	
	tests/*
	
	utils/*
	
	envs/*

usage: 

* collect data with `data_collect/player.py`(optional)
* train VAE with `trainvae.py`
* train MDN-RNN with `trainmdrnn.py`
* train policy with simulate env(sim_env)
    * to test in real env when training, use`sim_env/a2c_common.py`  to replace respective files in rl_games

