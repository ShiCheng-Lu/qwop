from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from gym import Env
from PIL import Image
import numpy as np
import io
import cv2
import time
from env.score_detector import ScoreDetector

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
    
    def state(self):
        image = self.screenshot()

        reward = self.score_detector.score(image)
        done = self.score_detector.done(image)

        state = cv2.resize(image, (160, 100))

        return (state, reward, done)

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

if __name__ == "__main__":
    pass

