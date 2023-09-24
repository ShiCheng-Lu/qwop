import torch
import numpy as np
import cv2

def to_black_white_tensor(image: torch.Tensor):
    return torch.tensor(image.mean(-1) / 255 > 0.5).float()

class Kernel:
    def __init__(self, filter):
        filter = torch.tensor(filter).float()
        self.kernel = torch.stack([filter, 1 - filter]).unsqueeze(1)
        self.sum = torch.sum(filter).item()
    
    def match(self, images,threshold=0.85):
        result = torch.conv2d(images.unsqueeze(1), self.kernel)
        matches = (result[:, 0] - result[:, 1] > self.sum * threshold)
        return matches.squeeze(1)

    def to(self, device):
        self.kernel = self.kernel.to(device)

class ScoreDetector:
    def __init__(self):
        score_text = cv2.imread("res/score_text.png", cv2.IMREAD_GRAYSCALE) / 255
        
        hist = score_text.sum(axis=0) > 0
        starts = np.nonzero(np.convolve(hist, np.array([1, -1])) > 0)[0]
        ends = np.nonzero(np.convolve(hist, np.array([-1, 1])) > 0)[0]

        digits = [Kernel(score_text[:, starts[i]:ends[i]]) for i in range(12)]

        self.period = digits[10]
        self.negate = digits[11]
        self.digits = digits[:10]

        self.done_img = cv2.imread("res/done.png", cv2.IMREAD_GRAYSCALE) / 255
        self.done_img = Kernel(self.done_img[150:250, 300:340])

        self.metres_text = Kernel(score_text[:, starts[12]:ends[-1]])

        self.device = "cpu"

    def to(self, device, type='torch'):
        # for digit in self.digits:
        #     digit.to(device)
        self.digits = [d.to(device) for d in self.digits]
        self.period = self.period.to(device)
        self.negate = self.negate.to(device)
        self.metres_text =  self.metres_text.to(device)
        self.done_img = self.done_img.to(device)

        self.device = device
        return self
    
    def score(self, images):
        images = to_black_white_tensor(images[:, 19:19+32, 200:-200])

        metre_matches = self.metres_text.match(images, 0.7).float()
        metre_index = torch.argmax(metre_matches, dim=-1).view(-1)

        distances = []
        for i, e in enumerate(metre_index):
            s = e - 80
            distances.append(images[i, :, s:e])
        images = torch.stack(distances)

        # reconstruct number
        result = torch.zeros((images.shape[0], images.shape[2]), dtype=torch.int, device = self.device)
        # find period sign
        matches = self.period.match(images)
        matches = torch.conv1d(matches.float(), torch.tensor([[[1., -1.]]])) > 0 # dedup
        result[:, :-self.period.kernel.shape[-1]] += matches * 46
        # find negate sign
        matches = self.negate.match(images)
        matches = torch.conv1d(matches.float(), torch.tensor([[[1., -1.]]])) > 0 # dedup
        result[:, :-self.negate.kernel.shape[-1]] += matches * 45
        # find where digits are
        for i, digit in enumerate(self.digits):
            matches = digit.match(images)
            matches = torch.conv1d(matches.float(), torch.tensor([[[1., -1.]]])) > 0 # dedup
            result[:, :-digit.kernel.shape[-1]] += matches * (i + 48)

        numbers = []
        # reconstruct number:
        for data in result:
            final = ""
            for digit in data:
                if digit:
                    final += chr(digit.item())
            numbers.append(float(final))
        return numbers
    
    def done(self, image):
        image = to_black_white_tensor(image[:, 150:250, 300:340])

        done_match = self.done_img.match(image, 0.7)
        return done_match.squeeze()
    
    def head_height(self, image):
        image = torch.tensor(image)
        head_colour_filtered = image == (13 / (255 * 3))
        head_colour_hist = head_colour_filtered.sum(dim=1)
        
        result = torch.nonzero(head_colour_hist)
        if len(result) == 0:
            return 400
        return result.min().item()

if __name__ == "__main__":
    detector = ScoreDetector()
    images = cv2.imread("res/0.png")

    print(images.shape)
    
    res = detector.score(np.stack([images]))
    print(res)

