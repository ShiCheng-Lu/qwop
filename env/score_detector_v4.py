import torch
from torch import nn
import cv2
import numpy as np

import matplotlib.pyplot as plt

class ScoreDetector:

    def __init__(self):
        # nn for character recognition
        # input size: 1x7x7
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            # nn.Linear(2048, 512),
            # nn.LeakyReLU(),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 62),
        )
    
    def to(self, device):
        self.model.to(device)
    
    def __call__(self, x):
        # input size: 256x32x4

        # segmentation with histogram
        x = torch.sum(x, dim=2)
        x = x > x.float().mean()

        hist = torch.sum(x, dim=0)
        hist_max = torch.max(hist) 
        hist_min = torch.min(hist)
    
        res = hist > (hist_max * 0.1 + hist_min * 0.9)

        lows = []
        highs = []
        for i, p in enumerate(res):
            if not p: continue
            if len(lows) == 0 or highs[-1] != i - 1:
                lows.append(i)
                highs.append(i)
            else:
                highs[-1] = i

        images = torch.zeros((len(lows), 1, 32, 32))
        for i, low, high in zip(range(len(lows)), lows, highs):
            mid = (low + high) // 2
            if high - low > 32:
                low, high = mid - 14, mid + 14
            
            images[i, :, :, low - mid + 14:high - mid + 14] = x[:, low : high]
        
        plt.figure(dpi=90)
        for i, img in enumerate(images):
            plt.subplot(5, 5, i+1)
            plt.imshow(img[0])
            plt.axis('off')
        # thinning
        # weights = torch.ones((1, 1, 3, 3)) / 9
        # images = (torch.conv2d(images, weights, padding=1) > 0.9).float()
        # images = (torch.conv2d(images, weights, padding=1)).float()

        for i, img in enumerate(images):
            plt.subplot(5, 5, i+11)
            plt.imshow(img[0])
            plt.axis('off')
        plt.show()

        result = self.model(images)[:, :]
        result = torch.softmax(result, dim=1)
        scores, indices = torch.max(result, dim=1)
        print(result[:, :10])
        print([to_char(i) for i in indices])
        return scores, indices

    def load(self, model_path="qwop.pth"):
        self.model.load_state_dict(torch.load(model_path))
        return self

    def save(self, model_path="qwop.pth"):
        torch.save(self.model.state_dict(), model_path)
        return self

if __name__ == "__main__":
    detector = ScoreDetector().load("score_detector.pth")
    image = cv2.imread("test.png")
    index = 19
    # x = torch.tensor(np.array(image))
    # print(x.shape)
    print(detector(torch.tensor(image[index:index+32, 200:-200])))
    