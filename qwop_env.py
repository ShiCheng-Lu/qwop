from selenium.webdriver import Chrome, ChromeOptions, Edge, EdgeOptions
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from gym import Env
from PIL import Image
import numpy as np
import io
import cv2
import time


class QWOP_Env(Env):
    OBSERVATION_SPACE = 71

    def __init__(self, headless=True):
        super().__init__()
        # load the game
        URL = f"http://localhost:8000/Athletics.html"
        options = ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument("--mute-audio")
        options.add_argument("--window-size=700,600")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-client-side-phishing-detection")
        options.add_argument("--disable-crash-reporter")
        options.add_argument("--disable-oopr-debug-crash-dump")
        options.add_argument("--no-crash-upload")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-low-res-tiling")
        options.add_argument("--log-level=3")
        options.add_argument("--silent")

        self.browser = Chrome(options=options)
        self.browser.get(URL)

        time.sleep(1)

        self.game = self.browser.find_element(By.TAG_NAME, 'canvas')
        self.game.click()

        self.game_image = self.screenshot()
        self.done = False
        self.time = 0

        self.last_torso_x = 0
        self.feet_calf_limit = 1

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

        self.time = 0  # time.time()

        return self.state()[0]

    def snapshot(self):
        print(self.browser.execute_script("return globalbodystate;"))

    def ss(self): # using canvas.context("webgl").readPixels(), this is a lot faster than self.screenshot
        image = self.browser.execute_script('return getImage();')
        array = np.array(list(map(ord, image)))
        return np.rollaxis(array.reshape((100, 160, 3)), 2, 0)

    def state(self):
        body_state = self.browser.execute_script("return globalbodystate;")
        game_state = self.browser.execute_script("return globalgamestate;")
        
        self.done = self.done or game_state["gameEnded"] or game_state["gameOver"] # or bad_pos 

        torso_x = body_state['torso']['position_x']
        reward = 0

        # bad_pos = (body_state['leftFoot']['position_y'] - body_state['leftCalf']['position_y'] < self.feet_calf_limit or
        #            body_state['rightFoot']['position_y'] - body_state['rightCalf']['position_y'] < self.feet_calf_limit)

        if not self.done:
            reward += max(torso_x - self.last_torso_x, 0)
        self.last_torso_x = torso_x

        state = self.ss()

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
        current = time.time()
        self.time += (1 / 6)
        sleep = self.time - current
        if sleep > 0:
            time.sleep(sleep)
        else:
            self.time = current
        
        self.action(action)
        return self.state()

import game_host
if __name__ == "__main__":
    game_host.start()
    env = QWOP_Env(headless=False)
    env.reset()
    start = time.time()
    print(env.reset().shape)
    for i in range(100):
        actions = 0
        state, reward, done = env.step(actions)
        # print(reward)
        end = time.time()
        print(end - start)
        start = end
    # game_host.end()