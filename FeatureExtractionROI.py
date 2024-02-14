import cv2
import numpy as np
import torch


class FeatureExtractionROI:
    # Con questa classe vogliamo estrarre le regioni d'interesse sinistra e destra per ogni coppia di frame

    def __init__(self, intended_img_size=227):
        self.intended_img_size = intended_img_size
        # La dimensione della regione d'interesse è 1/5 della dimensione dell'immagine
        self.roi_block = round(self.intended_img_size / 5)

    def extract_roi_sequence(self, batch):
        """Extract the region of interests features for a batch of sequences"""
        left_batch = []
        right_batch = []
        for sequence in batch:
            left_sequence = []
            right_sequence = []
            sequence = sequence.permute(0, 2, 3, 1).numpy()  # Convert tensors to numpy arrays
            gray_images = [
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in sequence
            ]

            for i, j in zip(range(len(gray_images) - 1), range(1, len(gray_images))):
                flow = self.calculate_flow(gray_images[i], gray_images[j])

                warped_image = self.warp_image(sequence[i], flow)
                difference_image = self.get_difference_image(warped_image, sequence[j])

                left_roi, right_roi = self.get_region_of_interests(difference_image)
                # applichiamo un filtro di sharpening
                left_roi = self.sharpen_image(left_roi)
                right_roi = self.sharpen_image(right_roi)

                right_roi = cv2.flip(right_roi, flipCode=1)

                # facciamo il resize delle immagini a 227x227
                left_roi = cv2.resize(left_roi, (self.intended_img_size, self.intended_img_size))
                right_roi = cv2.resize(right_roi, (self.intended_img_size, self.intended_img_size))

                left_roi = torch.from_numpy(left_roi).permute(2, 0, 1)
                right_roi = torch.from_numpy(right_roi).permute(2, 0, 1)
                left_sequence.append(left_roi)
                right_sequence.append(right_roi)
            left_sequence = torch.stack(left_sequence)
            right_sequence = torch.stack(right_sequence)
            left_batch.append(left_sequence)
            right_batch.append(right_sequence)
        return left_batch, right_batch

    @staticmethod
    def calculate_flow(prev, next):
        return cv2.calcOpticalFlowFarneback(prev=prev, next=next, flow=None, pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    @staticmethod
    def warp_image(image, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        return cv2.remap(src=image, map1=flow, map2=None, interpolation=cv2.INTER_LINEAR)

    def get_region_of_interests(self, image):
        x_start = 0
        x_end = 2 * self.roi_block
        y_start = self.roi_block
        y_end = 3 * self.roi_block
        left_roi = image[y_start:y_end, x_start:x_end]
        right_roi = image[y_start:y_end, -x_end:]
        return left_roi, right_roi

    @staticmethod
    def get_difference_image(image_a, image_b):
        # Convert to int32 to allow subtraction
        return cv2.absdiff(image_b, image_a)

    @staticmethod
    def sharpen_image(image):
        """Sharpen the image using a simple sharpening kernel"""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
