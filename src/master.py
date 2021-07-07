import gym
import tensorflow as tf
import controller_img_dnn
# import controller_masstree

from stable_baselines3 import DQN

from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


if __name__ == "__main__":
    env1 = controller_img_dnn.CustomEnv()
    # env2 = controller_masstree.CustomEnv()

    class CustomPolicy(MlpPolicy):
        __module__ = None
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                            layers=[256, 128, 64],act_fun=tf.nn.relu,)

    model1 = DQN.load('./models/07_06_09_img_dnn/model.zip', kwargs=dict(policy_kwargs=dict(act_fun=tf.nn.relu, layers=[256, 128, 64])) , env=env1)
    # model2 = DQN.load('./models/07_03_09_masstree/model.zip')

    # state1 = env1.reset()
    # state2 = env2.reset()

    act1 , _ = model1.predict([0] * (11 + 2), deterministic=True)

    print(act1)