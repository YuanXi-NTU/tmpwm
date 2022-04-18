referenced:https://github.com/ctallec/world-models

usage: 

* collect data with `data_collect/player.py` with a PPO policy, and replace respective files in rl_games (optional,called `res_buffer.pickle` here)
* train VAE with `trainvae.py`
* train MDN-RNN with `trainmdrnn.py`
* train policy with simulated env(files in sim_env, temporarily)
    * to train in simulated env, move `TestPPO.yaml` to `isaacgymenvs/cfg/train/`, `Test.yaml` to `isaacgymenvs/cfg/task/`, `test.py` to `isaacgymenvs/tasks/`(`config-test.yaml` is optional)
    * to test in real env when training, use`sim_env/a2c_common.py`  to replace respective files in rl_games

