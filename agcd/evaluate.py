import numpy as np
import torch
from . import helper
from .envs import make_vec_envs

def evaluate(
    num_evals, 
    uni_agent, 
    ob_rms,
    env_name, 
    init_robots,
    seed=1, 
    device=None,
    return_attention=False):

    eval_envs = make_vec_envs(env_name, init_robots, seed, None, device, ret=False, ob=True)
    vec_norm = helper.get_vec_normalize(eval_envs)
    
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    # recorders
    eval_episode_rewards = []
    attention_list=[]
    obs = eval_envs.reset()
    while True:
        with torch.no_grad():
            if return_attention:
                val, action, logp,attention = uni_agent.uni_act(obs, mean_action=True,return_attention=return_attention)
                attention_list.append(attention)
            else:
                val, action, logp = uni_agent.uni_act(obs, mean_action=True,return_attention=return_attention)
        obs, rewards, done, infos = eval_envs.step(action)
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
        # check
        if num_evals*len(init_robots) == len(eval_episode_rewards):
            break
    eval_envs.close()
    if return_attention:
        attention_tensor = torch.cat(attention_list, dim=0)
        attention_avg = attention_tensor.mean(dim=0)
        attention_flat=attention_avg.sum(dim=0)
        return np.average(eval_episode_rewards),attention_flat.cpu().numpy()
    else:
        return np.average(eval_episode_rewards),None

   