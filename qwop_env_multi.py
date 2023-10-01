from multiprocessing import Process, Queue
from gym import Env
from qwop_env import QWOP_Env
import numpy as np
import time
import torch
import game_host

def qwop_env(command_queue, result_queue, headless):
    env = QWOP_Env(headless=headless)
    # send something through result_queue to signal intiailization completion
    result_queue.put(0)

    while True:
        command = command_queue.get()
        if command == "quit":
            return
        elif command == "reset":
            state = env.reset()
        elif isinstance(command, int):
            state = env.step(command)
        
        result_queue.put(state)

class QWOP_Env_Multi(Env):
    def __init__(self, num=1, headless=True):
        super().__init__()
        self.num = num

        self.command_queues: list[Queue] = [Queue(2) for _ in range(num)]
        self.result_queues: list[Queue] = [Queue(2) for _ in range(num)]
        self.processes: list[Process] = []
        for i in range(num):
            process = Process(target=qwop_env, args=(self.command_queues[i], self.result_queues[i], headless))
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

        return np.stack(observs), rewards, dones

if __name__ == '__main__':
    game_host.start()
    env_count = 2
    env = QWOP_Env_Multi(num=env_count, headless=True)
    time.sleep(5)
    start = time.time()
    print(env.reset().shape)
    for i in range(100):
        actions = torch.randint(0, 8, (env_count,))
        state, reward, done = env.step(actions)
        # print(reward)
        end = time.time()
        print(end - start)
        start = end
    
    env.free()
    game_host.end()