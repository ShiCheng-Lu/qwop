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

        self.browser = Chrome(options=options)
        self.browser.get(URL)

        time.sleep(1)

        self.game = self.browser.find_element(By.TAG_NAME, 'canvas')
        self.game.click()

        self.game_image = self.screenshot()
        self.done = False
        self.time = None

        self.last_torso_x = 0

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

        self.time = None  # time.time()

        return self.state()[0]

    def snapshot(self):
        print(self.browser.execute_script("return globalbodystate;"))

    def ss(self):
        start = time.time()
        
        
        # image = Image.open(io.BytesIO(self.game.screenshot_as_png))

        image = self.browser.execute_script('return getImage();')
        # image = np.fromiter(map(ord, image), dtype=np.uint8, count=96000)
        image = np.array(list(map(ord, image)))
        # iamge = np.array(image)

        print(time.time() - start)

        # from matplotlib import pyplot as plt
        # plt.imshow(image.reshape((100, 160, 3)))
        # plt.gca().invert_yaxis()
        # plt.show()

        print(len(image), image[:10])

    def state(self):
        body_state = self.browser.execute_script("return globalbodystate;")

        state = np.array([value for part in body_state.values()
                         for value in part.values()], dtype=float)

        torso_x = body_state['torso']['position_x']
        reward = max(torso_x - self.last_torso_x, 0)
        
        # reward for 

        self.last_torso_x = torso_x

        bad_pos = (body_state['leftFoot']['position_y'] < body_state['leftCalf']['position_y'] or
                   body_state['rightFoot']['position_y'] < body_state['rightCalf']['position_y'])

        self.done = self.done or bad_pos

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
        self.action(action)
        time.sleep(0.1)
        return self.state()

import game_host
if __name__ == "__main__":
    game_host.start()
    env = QWOP_Env(headless=False)
    env.reset()
    for i in range(10): env.ss()
    # game_host.end()