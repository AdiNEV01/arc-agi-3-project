"""Regression checks for history-only ablation; run from repository root."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from arcengine import GameAction
from DQN.train import get_args
from DQN.dqn_improvement_1 import StateNoveltyReward
from DQN.dqn_improvement_2 import FrameHistory
from DQN.q_network import QNetwork, encode_frames, observation_to_tensor
from DQN.dqn_agent import DQNAgent

def obs(color):
    return SimpleNamespace(frame=[np.full((64,64),color,dtype=np.uint8)])

class Task1Tests(unittest.TestCase):
    def test_variant_contract_and_defaults(self):
        for variant,frames,beta in [('v0',1,.01),('v1',1,.01),('v2',4,.01),('v2_history_only',4,0.)]:
            with patch.object(sys,'argv',['train','--variant',variant]):
                args=get_args()
                self.assertEqual((args.num_frames,args.novelty_beta),(frames,beta))
                self.assertEqual((args.buffer_capacity,args.target_update_interval),(3000,250))

    def test_zero_beta_still_counts_but_never_rewards(self):
        novelty=StateNoveltyReward(beta=0.)
        novelty.register_reset(obs(0))
        for a,b in [(0,1),(1,1),(1,2),(2,0)]:
            self.assertEqual(novelty.bonus(obs(a),obs(b)),0.)
        self.assertEqual(novelty.unique_states,3)

    def test_history_channel_order_and_reset(self):
        h=FrameHistory(4)
        first=h.reset(obs(1))
        following=h.advance(obs(2))
        self.assertEqual(following[:,0,0].tolist(),[1,1,1,2])
        encoded=encode_frames(following)
        self.assertEqual(tuple(encoded.shape),(1,64,64,64))
        self.assertEqual(encoded[0,:,0,0].nonzero().flatten().tolist(),[1,17,33,50])
        self.assertEqual(tuple(QNetwork(4)(encoded).shape),(1,4102))
        h.reset(obs(3))
        self.assertEqual(first[:,0,0].tolist(),[1]*4)
        self.assertEqual(h.state()[:,0,0].tolist(),[3]*4)

    def test_legacy_encoding_and_checkpoint_shapes(self):
        torch.testing.assert_close(observation_to_tensor(obs(3)),encode_frames(np.full((1,64,64),3,dtype=np.uint8)))
        for variant in ('v0','v1','v2'):
            for seed in (42,123,456):
                cp=torch.load(Path(f'DQN/checkpoints/dqn_{variant}_ls20_seed{seed}.pt'),weights_only=True)
                frames=cp.get('config',{}).get('num_frames',1)
                model=QNetwork(frames)
                model.load_state_dict(cp['online_state_dict'])
                self.assertEqual(model.encoder[0].in_channels,16*frames)

    def test_terminal_replay_is_independent_and_learns(self):
        h=FrameHistory(4)
        a=DQNAgent(num_frames=4,batch_size=1)
        state=h.reset(obs(0)); following=h.advance(obs(1))
        a.remember(state,0,0,following,True,[])
        following[:]=4
        t=a.replay.storage[0]
        self.assertEqual(t.next_state[:,0,0].tolist(),[0,0,0,1])
        self.assertFalse(t.next_valid_mask.any())
        self.assertTrue(np.isfinite(a.learn()))
        self.assertEqual(a.gradient_steps,1)

    def test_greedy_masks_illegal_actions(self):
        a=DQNAgent(num_frames=4)
        state=FrameHistory(4).reset(obs(0))
        # Only ACTION1 is permitted, regardless of network preferences.
        self.assertEqual(a.select_action(state,[GameAction.ACTION1],0),0)
        self.assertEqual(a.gradient_steps,0)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
