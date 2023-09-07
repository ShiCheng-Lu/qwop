import cv2
import numpy as np
import torch

class ScoreDetector:
    def __init__(self):
        score_text = cv2.imread("res/score_text.png", cv2.IMREAD_GRAYSCALE) / 255
        
        hist = score_text.sum(axis=0) > 0
        starts = np.nonzero(np.convolve(hist, np.array([1, -1])) > 0)[0]
        ends = np.nonzero(np.convolve(hist, np.array([-1, 1])) > 0)[0]

        self.digits = [score_text[:, s:e] for s, e in zip(starts, ends)]
        self.period = self.digits[10]
        self.negate = self.digits[11]
        self.digits = self.digits[:10]

        self.done_img = cv2.imread("res/done.png", cv2.IMREAD_GRAYSCALE) / 255
        self.done_img = (self.done_img[150:250, 300:340] > 0.5).astype(np.float32)

        self.metres_text = score_text[:, starts[12]:ends[-1]]
    
    def to(self, device, type='torch'):
        self.digits = [torch.tensor(d).to(device) for d in self.digits]
        self.period = torch.tensor(self.period).to(device)
        self.negate = torch.tensor(self.negate).to(device)
        self.metres_text = torch.tensor(self.metres_text).to(device)
        self.done_img = torch.tensor(self.done_img).to(device)
        return self

    def match(self, image, subsection, threshold=0.85):
        kernel = torch.stack([subsection, 1 - subsection]).unsqueeze(1)
        # meters text match:
        result = torch.conv2d(image.unsqueeze(0), kernel.float())
        matches = (result[0] - result[1] > (torch.sum(subsection) * threshold))
        if matches.shape[1] >= 2:
            matches = torch.conv1d(matches.float(), torch.tensor([[[-1, 1]]]).float().cuda()) > 0
        return torch.nonzero(matches[0])
    
    def score(self, image):
        image = image[19:19+32, 200:-200] > 0.5

        image = torch.tensor(image).cuda().float()
        metre_index = self.match(image, self.metres_text)
        if len(metre_index) == 0:
            return None
        
        image = image[:, :metre_index]

        period_index = self.match(image, self.period)
        negate_index = self.match(image, self.negate)
        digits_index = []
        for i, digit in enumerate(self.digits):
            for match in self.match(image, digit):
                digits_index.append((match.item(), i))
        
        # reconstruct number:
        digits_index = sorted(digits_index)

        result = ""
        if len(negate_index) != 0:
            result += "-"
        for index, digit in digits_index:
            if len(period_index) != 0 and index > period_index:
                result += "."
                period_index = []
            result += f"{digit}"
        try:
            return float(result)
        except:
            return None
    
    def done(self, image):
        image = torch.tensor(image[150:250, 300:340] > 0.5).cuda().float()
        done_match = self.match(image, self.done_img)
        return len(done_match) > 0
    
    def head_height(self, image):
        image = torch.tensor(image).cuda()
        head_colour_filtered = image == (13 / (255 * 3))
        head_colour_hist = head_colour_filtered.sum(dim=1)
        
        result = torch.nonzero(head_colour_hist)
        if len(result) == 0:
            return 400
        return result.min().item()
    
if __name__ == "__main__":
    detector = ScoreDetector().to("cuda")
    image = cv2.imread(f"0.png").sum(axis=2) / (255 * 3)

    # print(image.shape)
    # detector.to("cuda")
    result = detector.score(image)
    result = detector.done(image)

    result = (image == (13 / (255 * 3))).sum(axis=1) # y < 130 ?
    print(detector.head_height(image))
    cv2.imshow("ts", result.astype(np.float32))
    # cv2.imwrite("test.png", result * 255)
    cv2.waitKey()
