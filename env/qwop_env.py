from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from gym import Env
from PIL import Image
import numpy as np
import io
import cv2
import time
from score_detector import ScoreDetector

class QWOP_Env(Env):
    def __init__(self, headless=True):
        super().__init__()
        # load the game
        URL = f"http://www.foddy.net/Athletics.html"
        options = ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument("--mute-audio")
        options.add_argument("--window-size=675,560")
        
        self.browser = Chrome(options=options)
        self.browser.get(URL)
        
        time.sleep(1)

        self.game = self.browser.find_element(By.TAG_NAME, 'canvas')
        self.game.click()

        self.game_image = self.screenshot()
        self.done = False
        self.time = None

        self.last_score = 0

        self.score_detector = ScoreDetector()

    def screenshot(self):
        image = Image.open(io.BytesIO(self.game.screenshot_as_png))
        image = np.array(image)[:, :, :3]
        return image

    def reset(self):
        self.done = False

        key_actions = ActionChains(self.browser)
    
        key_actions.key_up('q')
        key_actions.key_up('w')
        key_actions.key_up('o')
        key_actions.key_up('p')
        
        key_actions.send_keys('r')
        key_actions.perform()

        self.time = None # time.time()
    
    def state(self):
        if self.done:
            return (np.zeros((50, 80, 3)), 0, True)

        image = self.screenshot()
        image_batch = np.stack([image])
        
        self.done = self.score_detector.done(image_batch)
        reward = self.score_detector.score(image_batch)[0]

        state = cv2.resize(image, (80, 50))

        return (state, reward, self.done)

    def action(self, action):
        if self.done:
            return
        
        thighs = action // 3
        calves = action % 3

        action_chain = ActionChains(self.browser)
        # thighs actions
        match thighs:
            case 0:
                action_chain.key_up('q')
                action_chain.key_up('w')
            case 1:
                action_chain.key_down('q')
                action_chain.key_up('w')
            case 2:
                action_chain.key_up('q')
                action_chain.key_down('w')
        # calves actions
        match calves:
            case 0:
                action_chain.key_up('o')
                action_chain.key_up('p')
            case 1:
                action_chain.key_down('o')
                action_chain.key_up('p')
            case 2:
                action_chain.key_up('o')
                action_chain.key_down('p')
    
        action_chain.perform()
    
    def step(self, action):
        if self.time == None:
            self.time = time.time()
        
        self.time += 0.2
        sleep_time = self.time - time.time()
        time.sleep(max(sleep_time, 0))

        self.action(action)
        return self.state()

if __name__ == "__main__":
    import random
    
    env = QWOP_Env(headless=False)
    print("env started")
    env.reset()

    for i in range(100):
        action = random.randint(0, 9)
        obs, reward, done = env.step(action)

        print(reward)
        if done:
            env.reset()

