#! /usr/bin/env python
import os
import pickle
import gym_csgo
import gymnasium as gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
import time
import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import PixelBCLearner, PixelIQLLearner
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_pixels
from jaxrl2.data import MemoryEfficientReplayBuffer
from datetime import datetime
from flax.training.checkpoints import save_checkpoint, latest_checkpoint,restore_checkpoint
import cv2
import h5py
import numpy as np
from skimage.transform import resize
from flax import jax_utils
FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", np.random.randint(10990), "Random seed.")
flags.DEFINE_integer("eval_episodes", 1, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 10000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 99999, "Eval interval.")
flags.DEFINE_integer("batch_size", 32, "Mini batch size.")
flags.DEFINE_integer("max_steps", 100_000, "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e1), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 200, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
config_flags.DEFINE_config_file(
    "config",
    "configs/offline_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spi-n-0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}
mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
mouse_x_lim = (mouse_x_possibles[0],mouse_x_possibles[-1])
mouse_y_lim = (mouse_y_possibles[0],mouse_y_possibles[-1])
data_folder = "../../../../testML/datasets2/csgo_RL/"
metadata_folder = data_folder + "metadata/"
dataset_folder = data_folder +"dataset/" #"dataset"
expert_dataset_folder = data_folder +"dataset_dm_expert_dust2/"



import itertools

def split_dict(d, n):
    # Calculate number of items per partition
    k = len(d) // n
    
    # Split the dictionary into partitions
    partitions = [dict(itertools.islice(d.items(), i*k, (i+1)*k)) for i in range(n-1)]
    partitions.append(dict(itertools.islice(d.items(), (n-1)*k, None)))
    
    return partitions




def mouse_preprocess(mouse_x, mouse_y):
    # clip and distcretise mouse
    mouse_x = np.clip(mouse_x, mouse_x_lim[0],mouse_x_lim[1])
    mouse_y = np.clip(mouse_y, mouse_y_lim[0],mouse_y_lim[1])

    # find closest in list
    mouse_x = min(mouse_x_possibles, key=lambda x_:abs(x_-mouse_x))
    mouse_y = min(mouse_y_possibles, key=lambda x_:abs(x_-mouse_y))

    return mouse_x, mouse_y
def convert_act(act_list):
    actions = []
  
    keyboard = act_list[0]
    x,y = mouse_preprocess(act_list[1],act_list[2])
    left_click = act_list[3]
    right_click = act_list[4]


    # Check for each key and append 1 or 0 to actions accordingly
    if "w" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "a" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "s" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "d" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "shift" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "space" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "ctrl" in keyboard:
        actions.append(1)
    else:
        actions.append(0)


    actions.append(left_click)
    actions.append(right_click)

    if "r" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "g" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "e" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "q" in keyboard:
        actions.append(1)
    else:
        actions.append(0)

    if "x" in keyboard:
        actions.append(1)
    else:
        actions.append(0)
    for i in range(10):
        if str(i) in keyboard:
            actions.append(1)
        else:
            actions.append(0) 
    pos_x = mouse_x_possibles.index(x)
    pos_y = mouse_y_possibles.index(y)
    for i in range(len(mouse_x_possibles)):
        if i ==pos_x:
            actions.append(1)
        else:
            actions.append(0)
   
    for i in range(len(mouse_y_possibles)):
        if i ==pos_y:
            actions.append(1)
        else:
            actions.append(0)

    # Set equip state
    return np.array(actions)

def convert_def(raw_dict):
# Define the relevant keys to extract from the second dict


    conv_v2=[]
    if "gsi_health" in raw_dict.keys():
        conv_v2.append(raw_dict["gsi_health"])  # health
    else:
        conv_v2.append(0)
    conv_v2.append(0)  # helmet    
    conv_v2.append(0)                       # armor
    conv_v2.append(0)                        # flashed
    conv_v2.append(0)                        # smoked
    conv_v2.append(0)                        # bruning
    conv_v2.append(0) # money
    conv_v2.append(0) # value
    if "gsi_kills" in raw_dict.keys():
        conv_v2.append(raw_dict["gsi_kills"]) # kills
    else:
        conv_v2.append(0)
    if "gsi_assists" in raw_dict.keys():
        conv_v2.append(raw_dict["gsi_assists"]) # assits
    else:
        conv_v2.append(0)
    if "gsi_deaths" in raw_dict.keys():
        conv_v2.append(raw_dict["gsi_deaths"]) # deaths
    else:
        conv_v2.append(0)
    conv_v2.append(0) # mvps
    conv_v2.append(0) # score

    if raw_dict["gsi_team"] is not None:
        if raw_dict["gsi_team"] == "CT":
            conv_v2.append(1)
        else:
            conv_v2.append(0)
    else:
        conv_v2.append(2)
     # team
   #"gsi_weapons"
    weapons = raw_dict["gsi_weapons"]
    active_wep=[]
    weps=[]
    for x in range(10):
        wep = []
        if f"weapon_{x}" in weapons.keys():
            is_active =False
            if weapons[f"weapon_{x}"]["state"] is not None:
                if weapons[f"weapon_{x}"]["state"] == "active":
                    is_active=True
                    wep.append(1)
                elif weapons[f"weapon_{x}"]["state"] == "holstered":
                    wep.append(2)
                else:
                    wep.append(3)
            else:
                wep.append(0)
            if "ammo_clip" in weapons[f"weapon_{x}"].keys():
                wep.append(weapons[f"weapon_{x}"]["ammo_clip"])
            else:
                wep.append(0)
            if "ammo_clip_max" in weapons[f"weapon_{x}"].keys():
                wep.append(weapons[f"weapon_{x}"]["ammo_clip_max"])
            else:
                wep.append(0)
            if "ammo_reserve" in weapons[f"weapon_{x}"].keys():
                wep.append(weapons[f"weapon_{x}"]["ammo_reserve"])
            else:
                wep.append(0)
            if is_active:
                is_active=False
                active_wep = wep.copy()
        else:
            wep = [0,0,0,0]
        
        weps.extend(wep)
    if len(active_wep) != 4:
        active_wep.extend([0,0,0,0])
    conv_v2.extend(active_wep)
    conv_v2.extend(weps)
    return np.array(conv_v2)

def get_rew(sum_shots,current_obs,last_obs):
        # Difference of scores

        sum_shots=sum_shots
        try:
            sum_shots+= current_obs['states'][f'weapon_active']['ammo_clip_max'] -  current_obs['states'][f'weapon_active']['ammo_clip']
            sum_shots+= last_obs['states'][f'weapon_active']['ammo_reserve']-  current_obs['states'][f'weapon_active']['ammo_reserve']
        
            for index in range(10):
                sum_shots+= current_obs['states'][f'weapon_{index}']['ammo_clip_max'] -  current_obs['states'][f'weapon_{index}']['ammo_clip']
                sum_shots+= last_obs['states'][f'weapon_{index}']['ammo_reserve']-  current_obs['states'][f'weapon_{index}']['ammo_reserve']
        except:
            pass
        sum_shots=sum_shots
        

        try:
            return  current_obs['states'][8] - 0.5* current_obs['states'][10] - 0.02* sum_shots
        except:
            return  last_obs['states'][8] - 0.5* last_obs['states'][10] - 0.02* sum_shots


def read_conv_dataset(env,dict_chunks,type):
    
    replay_buffer_size = len(dict_chunks) 
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
    )
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    observation = {"pixels": np.stack([black_img]*3, axis=-1), "states": env.observation_space.sample()['states']}

    #observation =  env.observation_space.sample()
    # convert img

    try:

        for index, dict_chunk in enumerate(dict_chunks):


            #dict_chunk = 'file_num101_frame_3'
            splits = dict_chunk.split("_")
            hdf5num = splits[1].removeprefix("num")
            i = splits[3]
            # if not os.path.exists(dataset_folder+f'hdf5_dm_july2021_{hdf5num}.hdf5'):
            #     return None
            # with h5py.File(dataset_folder+f'hdf5_dm_july2021_{hdf5num}.hdf5', 'r') as f:
            
            if "expert" in type:
                filename = expert_dataset_folder+f'hdf5_dm_july2021_expert_{hdf5num}.hdf5'
            else:
                filename = dataset_folder+f'hdf5_dm_july2021_{hdf5num}.hdf5'
            print(filename)
            if not os.path.exists(filename):
                return None
            with h5py.File(filename, 'r') as f:
                print(f"hdf5_dm_july2021_expert_{hdf5num}.hdf5")
                sum_shots=0
                
                image = np.array(f[f"frame_{i}_x"])
                states = convert_def(dict_chunks[dict_chunk][0])
                
                
                action = convert_act(dict_chunks[dict_chunk][1])
                
                print(index)
                #print(image.shape)
                # re_pov =resize(image, (150, 150,3))
                re_pov =resize(image, (200, 200,3))
                

            # print(observation["pixels"].shape)
                split_array_list = [x.squeeze() for x in np.split(observation["pixels"], observation["pixels"].shape[-1], axis=-1)]
                #print(split_array_list[0].shape,split_array_list[1].shape,split_array_list[2].shape)

                obs_pixels = np.stack([re_pov, split_array_list[0], split_array_list[1] ], axis=-1)
            
                next_observation = {"pixels":obs_pixels,"states":states} # 0
            #
            # print(obs_pixels.shape)
                reward = get_rew(sum_shots,next_observation,observation)

                # for now
                mask = 0
                done = True if index == len(dict_chunk)-1  else False
                replay_buffer.insert(
                    dict(
                        observations=observation,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,
                        next_observations=next_observation,
                    )
                )
                observation = next_observation

        return replay_buffer
    except:
        return None


def main(_):

    wandb.init(project="jaxrl2_offline_pixels")
    wandb.config.update(FLAGS)

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    def wrap(env):
        if "quadruped" in FLAGS.env_name:
            camera_id = 2
        else:
            camera_id = 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )


    env_config = {'width':1120,'height':600,'display':":89","in_thread":False}
    eval_env_config= {'width':1120,'height':600,'display':":99","in_thread":True}    
    eval_env = gym.make(FLAGS.env_name,**eval_env_config)
    eval_env = wrap(eval_env)
    env = gym.make(FLAGS.env_name, **env_config)
    print(env.observation_space)
    env = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    #env.seed(FLAGS.seed)

   
    #eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = FLAGS.max_steps
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    )
    try:
        #agent = agent.load(chek_path+"models/testing.pkl",**kwargs)
        agent._actor = restore_checkpoint(data_folder+"models/testing",agent._actor,parallel=False)
        print("got latest")
    except Exception as e:
        #raise e
        print(e,"failed getting latest")
    time.sleep(3)
    
    

    # dataset_folder = os.path.join("datasets")
    # dataset_file = os.path.join(dataset_folder, f"{FLAGS.env_name}")
    # with open(dataset_file, "rb") as f:
    #     replay_buffer = pickle.load(f)
    all_meta = os.listdir(metadata_folder)
    expert_meta = [filename for filename in all_meta if "expert" in filename]
    other_meta = [filename for filename in all_meta if "expert" not in filename]

    for _ in range(100):
        for folder, type in zip([other_meta,expert_meta],["other","expert"]):#,expert_meta
            for meta_file in folder:
                dict_chunkx = np.load(metadata_folder+meta_file,allow_pickle=True)
    
                dict_chunks = dict_chunkx.item()
                
                partitions = split_dict(dict(dict_chunks), 5)
                # items= list(dict_chunks.items())
                # dict_chunks = list(dict_chunks.keys())
                
                # segment_length = len(dict_chunks) // 10

                # segmented_list = [dict_chunks[i:i+segment_length] for i in range(0, len(dict_chunks), segment_length)]
                
            # print(segmented_list)

                for seg in partitions:
                    replay_buffer = read_conv_dataset(env,seg, type)
                
                    if replay_buffer is not None:
                    
                        
                        replay_buffer.seed(FLAGS.seed)
                        if FLAGS.config.model_constructor == "PixelBCLearner":
                            replay_buffer_iterator = replay_buffer.get_iterator(
                                sample_args={
                                    "batch_size": FLAGS.batch_size,
                                    "include_pixels": True,
                                    "keys": ["observations", "actions"],
                                }
                            )
                        else:
                            replay_buffer_iterator = replay_buffer.get_iterator(
                                sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False}
                            )

                        for i in tqdm.tqdm(range(1, FLAGS.max_steps+1),smoothing=0.1,disable=not FLAGS.tqdm,):
                            
                            batch = next(replay_buffer_iterator)
                            update_info = agent.update(batch)

                            if i % FLAGS.log_interval == 0:
                                for k, v in update_info.items():
                                    wandb.log({f"training/{k}": v}, step=i * action_repeat)

                            if i % FLAGS.eval_interval == 0:

                                save_checkpoint(
                                    ckpt_dir=data_folder+"models/testing",
                                    target=agent._actor,
                                    step=agent.step,
                                    overwrite=True,keep=550
                                )
                                agent.step+=1
                                            
                                print("going to eval")
                                eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
                                for k, v in eval_info.items():
                                    wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)
                                

if __name__ == "__main__":
    app.run(main)
