from multiprocessing import Process, Queue
from gym import Env
from env.qwop_env import QWOP_Env
from env.score_detector import ScoreDetector
import numpy as np
import time
import torch

def qwop_env(command_queue, result_queue):
    env = QWOP_Env(headless=True)
    # send something through result_queue to signal intiailization completion
    result_queue.put(0)

    while True:
        command = command_queue.get()
        if command == "quit":
            return
        elif command == "reset":
            env.reset()
        elif isinstance(command, int):
            env.action(command)
        
        state = env.state()
        result_queue.put(state)

class Multi_QWOP_Env(Env):
    def __init__(self, num=1, headless=True):
        super().__init__()
        self.num = num

        self.score_detector = ScoreDetector()

        self.command_queues: list[Queue] = [Queue(2) for _ in range(num)]
        self.result_queues: list[Queue] = [Queue(2) for _ in range(num)]
        self.processes: list[Process] = []
        for i in range(num):
            process = Process(target=qwop_env, args=(self.command_queues[i], self.result_queues[i]))
            process.start()
            self.processes.append(process)
        print("created processes")
        
        # wait for initialization completion
        [queue.get() for queue in self.result_queues]
        self.time = None
    
    def free(self):
        for i in range(self.num):
            self.command_queues[i].put("quit")
        
        for p in self.processes:
            p.join()

    def reset(self):
        for i in range(self.num):
            self.command_queues[i].put("reset")
        
        self.time = time.time()

        result = [queue.get() for queue in self.result_queues]
        return np.stack(result)

    def step(self, actions: torch.Tensor):
        '''
        action: (thighs, calves)
        - thighs [0: no thigh, 1: q, 2]
        '''

        self.time += 0.2
        sleep_time = self.time - time.time()
        print(sleep_time, self.time)
        time.sleep(max(self.time - time.time(), 0))

        start = time.time()

        for i in range(self.num):
            self.command_queues[i].put(actions[i].item())

        observs = []
        rewards = []
        dones = []
        for queue in self.result_queues:
            observ, reward, done = queue.get()
            
            observs.append(observ)
            rewards.append(reward)
            dones.append(done)
        
        # self.score_detector.score()
        print("time: ", time.time() - start)

        return np.stack(observs), rewards, dones

    def reward(self, screenshot, i):
        score = self.score_detector.score(screenshot)
        if score == None:
            score = self.sum_dists[i] / 4
        # head_height = self.score_detector.head_height(screenshot)
        # reward += math.tanh((130 - head_height) / 10)
        reward = max(score - self.sum_dists[i] / 4, 0)

        self.sum_dists[i] += score
        self.sum_dists[i] -= self.dists[i].popleft()
        self.dists[i].append(score)

        # reward += clip(score - (self.sum_dist / 4), 1, -1)
        return reward

if __name__ == '__main__':
    env = Multi_QWOP_Env(num=3, headless=False)
    print(env.reset().shape)
    for i in range(1):
        actions = torch.randint(0, 8, (3,))
        print(env.step(actions).shape)
    
    env.free()