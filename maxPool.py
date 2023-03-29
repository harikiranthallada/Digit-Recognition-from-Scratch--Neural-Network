import numpy as np


class MaxPool:
    def iterate_image(self, image):
        h, w, _ = image.shape
        for i in range(h // 2):
            for j in range(w // 2):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):

        self.last_input = input
        h, w, k = input.shape
        output = np.zeros((h // 2, w // 2, k))

        for im_region, i, j in self.iterate_image(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backprop(self, d_L_d_out):

        d_L_d_inputs = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_image(self.last_input):
            h, w, f = im_region.shape
            # print(im_region.shape)
            amax = np.amax(im_region, axis=(0,1))

            for i2 in range(h):
                for j2 in range(w):
                    for k in range(f):
                        if self.last_input[i2, j2, k] == amax[k]:
                            d_L_d_inputs[i * 2 + i2, j * 2 + j2, k] = d_L_d_out[i, j, k]
        return d_L_d_inputs
