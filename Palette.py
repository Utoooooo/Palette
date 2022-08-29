import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class EMD:
    def __init__(self, tau=30, cell_array=[3, 3], start_pos=[10, 10], pos_mat=[3, 4], Ag=[0, 20], mode='trans'):
        """
        Elementary Motion Detector
        :param tau: Time constant in Hz (Tau/dT in some literature)
        :param cell_array: num of cells in x, y
        :param start_pos: location of first cell
        :param pos_mat: distance between cells
        :param Ag: distance between ChE and ChI
        :param mode: translation(trans) or looming(loom)
        """
        self.tau = tau
        self.Ch1E = np.zeros(cell_array)
        self.Ch1I = np.zeros(cell_array)
        self.Ch2E = np.zeros(cell_array)
        self.Ch2I = np.zeros(cell_array)
        self.LP1 = np.zeros(cell_array)
        self.LP2 = np.zeros(cell_array)
        self.Rexcite = []
        self.Rinhibit = []
        self.neuron_map = []
        self.Rout = []
        self.log = []
        if mode == 'loom':
            self.sample_matrix_ChE = self.get_sampling_matrix(cell_array, start_pos, pos_mat)
            self.sample_matrix_ChI = self.get_sampling_matrix_loom(self.sample_matrix_ChE,
                                                                   cell_array[0]*cell_array[1], Ag)
            print(self.sample_matrix_ChE)
            print(self.sample_matrix_ChI)
        else:
            self.sample_matrix_ChE = self.get_sampling_matrix(cell_array, start_pos, pos_mat)
            self.sample_matrix_ChI = self.get_sampling_matrix(cell_array, np.add(start_pos, Ag), pos_mat)
            print(self.sample_matrix_ChE)

    @staticmethod
    def get_sampling_matrix(cell_array, start_pos, pos_mat):
        """
        Generates coordinate for pixel extraction
        Convert to Tuple list
        """
        sampling_matrix = np.mgrid[start_pos[0]:(start_pos[0] + cell_array[0] * pos_mat[0]):pos_mat[0],
                            start_pos[1]: (start_pos[1] + cell_array[1] * pos_mat[1]): pos_mat[1]].reshape(2, -1)
        sampling_matrix = tuple(sampling_matrix.tolist())
        return sampling_matrix

    @staticmethod
    def get_sampling_matrix_loom(ChE_array, num_cell, Ag):
        """
        For Inhibitory channel in looming mode only
        Generates coordinate for pixel extraction
        Convert to Tuple list
        """
        ChE = np.asarray(ChE_array)
        print(type(ChE))
        mid_point = np.mean(ChE, axis=1)
        ChI = np.zeros(ChE.shape)
        for i in range(num_cell):
            if ChE[0, i] > mid_point[0]:
                ChI[0, i] = ChE[0, i] + Ag[0]
            elif ChE[0, i] < mid_point[0]:
                ChI[0, i] = ChE[0, i] - Ag[0]
            else:
                ChI[0, i] = ChE[0, i]
        for j in range(num_cell):
            if ChE[1, j] > mid_point[1]:
                ChI[1, j] = ChE[1, j] + Ag[1]
            elif ChE[1, j] < mid_point[1]:
                ChI[1, j] = ChE[1, j] - Ag[1]
            else:
                ChI[1, j] = ChE[1, j]
        ChI = ChI.astype(int)
        sample_matrix = tuple(ChI.tolist())
        return sample_matrix

    def update(self, frame):
        """
        Pass new frame to cell array and compute EMD result.
        """
        # print(frame[self.sample_matrix_ChE])
        self.Ch1E = frame[self.sample_matrix_ChE].reshape(self.Ch1E.shape)
        self.Ch1I = frame[self.sample_matrix_ChI].reshape(self.Ch1I.shape)
        self.Ch2I = self.Ch1E
        self.Ch2E = self.Ch1I
        self.lp_filter()
        self.Rexcite = np.multiply(self.LP1, self.Ch2E)
        self.Rinhibit = np.multiply(self.LP2, self.Ch2I)
        self.neuron_map = np.subtract(self.Rinhibit, self.Rexcite)
        # print(self.neuron_map)
        self.Rout = np.mean(self.neuron_map)
        self.log.append(self.Rout)

    def lp_filter(self):
        """
        Forward Euler: x(t+1) = (z(t)-x(t))/tau +x(t)
        """
        # print(self.LP1)
        self.LP1 = np.add(np.divide(np.subtract(self.Ch1E, self.LP1), self.tau), self.LP1)
        self.LP2 = np.add(np.divide(np.subtract(self.Ch1I, self.LP2), self.tau), self.LP2)


class LGMD:
    def __init__(self):
        self.P = []


class LMC:
    def __init__(self, x_dim, y_dim, kernel=None):
        """
        Laminar Monopolar Cell, includes subsampling and low-pass filtering in photoreceptor
        :param x_dim: size x
        :param y_dim: size y
        :param kernel: convolution kernel
        self.b: denominator coefficient
        self.a: numerator coefficient
        """
        self.filtered_data = None
        self.lmc_output = None
        if kernel is None:
            kernel = np.array([[-1 / 9, -1 / 9, -1 / 9], [-1 / 9, 8 / 9, -1 / 9], [-1 / 9, -1 / 9, -1 / 9]])
        self.dst = None
        self.kernel = kernel
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.b = np.array([0, 0.0001, -0.0011, 0.0052, -0.0170, 0.0439, -0.0574, 0.1789, -0.1524])
        self.a = np.array([1.0000, -4.3331, 8.6847, -10.7116, 9.0004, -5.3058, 2.1448, -0.5418, 0.0651])
        self.buffer = np.zeros([y_dim, x_dim, len(self.b)])

    def iir_filter(self, src):
        """
        Infinite impulse response (IIR) filter for temporal band-pass filtering
        :param src: raw video frame
        :return: reconstructed photoreceptor response
        """
        src = src/255
        for k in range(len(self.b) - 1):
            self.buffer[:, :, k] = self.buffer[:, :, k + 1]

        self.buffer[:, :, -1] = 0
        for k in range(len(self.b)):
            self.buffer[:, :, k] = self.buffer[:, :, k] + src * self.b[k]

        for k in range(len(self.b) - 1):
            self.buffer[:, :, k + 1] = self.buffer[:, :, k + 1] - self.buffer[:, :, 0] * self.a[k + 1]
        self.filtered_data = self.buffer[:, :, 0]

    def centre_surrounding_antagonism(self):
        """
        Zero padding and convolution.
        """
        G1 = np.ones([self.y_dim + 2, self.x_dim + 2])
        G1[1:-1, 1:-1] = self.buffer[:, :, 0]
        G1[0, 1:-1] = G1[1, 1:-1]
        G1[-1, 1:-1] = G1[-2, 1:-1]
        G1[:, 0] = G1[:, 1]
        G1[:, -1] = G1[:, -2]

        G2 = signal.convolve2d(G1, self.kernel, 'same')
        self.lmc_output = G2[1:-1, 1:-1]

    def update(self, src):
        """
        Subsample frame, filter and enhance
        """
        sig = cv2.resize(src, (self.x_dim, self.y_dim))
        sub = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
        self.iir_filter(sub)
        self.centre_surrounding_antagonism()


class RTC:
    def __init__(self, ts, tau, kernel, x_dim, y_dim):
        """
        :param ts: Sampling time constant
        :param tau: Gradient
        :param kernel: spatial filtering kernel
        """
        self.ts = ts
        self.tau = tau
        self.kernel = kernel
        self.on_channel = np.zeros([y_dim, x_dim])
        self.off_channel = np.zeros([y_dim, x_dim])
        self.on_delay = np.zeros_like(self.on_channel)
        self.off_delay = np.zeros_like(self.off_channel)
        self.on_delay_filtered = np.zeros_like(self.on_channel)
        self.off_delay_filtered = np.zeros_like(self.off_channel)
        self.b_array = np.array([1 / (1 + 2 * 1.25 / ts), 1 / (1 + 2 * 1.25 / ts)])
        self.a_array = np.array([1, (1 - 2 * 1.25 / ts) / (1 + 2 * 1.25 / ts)])
        self.on_buffer = np.zeros([y_dim, x_dim, len(self.b_array)])
        self.off_buffer = np.zeros([y_dim, x_dim, len(self.b_array)])
        self.output = None

    @staticmethod
    def gradient_check(diff):
        m, n = diff.shape
        tau = np.zeros_like(diff)
        for x_coord in range(m):
            for y_coord in range(n):
                if diff[x_coord, y_coord] > 0:
                    tau[x_coord, y_coord] = 3
                else:
                    tau[x_coord, y_coord] = 70
        return tau

    @staticmethod
    def filter1(u, tau, ts):
        # ts = 0.05
        para = 1 - np.exp(-(np.divide(ts, tau)))
        filtered_on = np.multiply(para, u)
        return filtered_on

    @staticmethod
    def filter2(u, tau, ts):
        # ts = 0.05
        para = np.exp(-(np.divide(ts, tau)))
        filtered_u = np.multiply(para, u)
        return filtered_u

    @staticmethod
    def spatial_filter(channel, kernel):
        n, m = channel.shape
        G1 = np.ones([n + 4, m + 4])
        G1[2:-2, 2:-2] = channel
        G1[1, 2:-2] = G1[2, 2:-2]
        G1[0, 2:-2] = G1[3, 2:-2]
        G1[-2, 2:-2] = G1[-3, 2:-2]
        G1[-1, 2:-2] = G1[-4, 2:-2]
        G1[:, 1] = G1[:, 2]
        G1[:, 0] = G1[:, 3]
        G1[:, -2] = G1[:, -3]
        G1[:, -1] = G1[:, -4]

        G3 = signal.convolve2d(G1, kernel, 'same')
        filtered_channel_small = G3[2:-2, 2:-2]
        return filtered_channel_small

    @staticmethod
    def iir_filter(b, a, sig, buffer):
        for k in range(len(b) - 1):
            buffer[:, :, k] = buffer[:, :, k + 1]

        buffer[:, :, -1] = 0
        for k in range(len(b)):
            buffer[:, :, k] = buffer[:, :, k] + sig * b[k]

        for k in range(len(b) - 1):
            buffer[:, :, k + 1] = buffer[:, :, k + 1] - buffer[:, :, 0] * a[k + 1]

        filtered_data = buffer[:, :, 0]
        return filtered_data, buffer

    def update(self, src):
        self.on_channel = src > 0
        self.on_channel = np.multiply(self.on_channel, src)
        self.off_channel = src <= 0
        self.off_channel = -(np.multiply(self.off_channel, src))
        on_difference = np.subtract(self.on_channel, self.on_delay)
        off_difference = np.subtract(self.off_channel, self.off_delay)
        self.on_delay = self.on_channel
        self.off_delay = self.off_channel
        tau_on = self.gradient_check(on_difference)
        tau_off = self.gradient_check(off_difference)
        on_filter1 = self.filter1(self.on_channel, tau_on, self.ts)
        off_filter1 = self.filter1(self.off_channel, tau_off, self.ts)
        on_filter2 = self.filter2(self.on_delay_filtered, tau_on, self.ts)
        off_filter2 = self.filter2(self.off_delay_filtered, tau_off, self.ts)
        on_filtered = np.add(on_filter1, on_filter2)
        off_filtered = np.add(off_filter1, off_filter2)
        subtracted_on_channel = np.subtract(self.on_channel, self.on_delay_filtered)
        subtracted_off_channel = np.subtract(self.off_channel, self.off_delay_filtered)
        self.on_delay_filtered = on_filtered
        self.off_delay_filtered = off_filtered
        on_channel_dead = np.multiply((subtracted_on_channel > 0)*1, subtracted_on_channel)
        off_channel_dead = np.multiply((subtracted_off_channel > 0)*1, subtracted_off_channel)
        on_spatial_filtered = self.spatial_filter(on_channel_dead, self.kernel)
        off_spatial_filtered = self.spatial_filter(off_channel_dead, self.kernel)
        on_spatial_dead = np.multiply((on_spatial_filtered > 0)*1, on_spatial_filtered)
        off_spatial_dead = np.multiply((off_spatial_filtered > 0)*1, off_spatial_filtered)
        on_delay_output, self.on_buffer = self.iir_filter(self.b_array, self.a_array, on_spatial_dead, self.on_buffer)
        off_delay_output, self.off_buffer = self.iir_filter(self.b_array, self.a_array, off_spatial_dead, self.off_buffer)
        correlated_on_off =np.multiply(on_spatial_dead, off_delay_output)
        correlated_off_on =np.multiply(off_spatial_dead, on_delay_output)
        self.output = np.add(correlated_on_off, correlated_off_on)


def optic_blur(src, pixelPerDegree):
    sigma_deg = 1.4/2.35
    sigma_pixel = sigma_deg*pixelPerDegree
    kernel_size = 2*(int(sigma_pixel))
    result = cv2.GaussianBlur(src, (kernel_size+1, kernel_size+1), sigma_pixel)
    return result


def sub_sampling(src, XDim, YDim):
    sig = cv2.resize(src, (XDim, YDim))
    dst = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
    return dst


def remapping(ndarray, new_max=255, new_min=0):
    remapped_array = np.add(np.multiply(np.divide(np.subtract(ndarray, np.min(ndarray)),
                                                  (np.max(ndarray)-np.min(ndarray))), (new_max-new_min)), new_min)
    return remapped_array


# cap = cv2.VideoCapture('corridor.avi')
#
# test_emd = EMD(tau=10, Ag=[4, 4], mode='loom')
# print(test_emd.sample_matrix_ChI)
# photoreceptor = LMC(100, 100)
# detector = RTC(0.1, )
#
# while cap.isOpened():
#     ret, frames = cap.read()
#     # frames = cv2.resize(frames, (256, 256))
#     # test_emd.update(frames)
#     # # print(frames[a])
#     # # print(test_emd.Rexcite)
#     # fig = remapping(test_emd.neuron_map)
#     fig = optic_blur(frames, 15)
#     # fig = sub_sampling(fig, 100, 100)
#     photoreceptor.update(fig)
#     cv2.imshow('neuron map', cv2.resize(photoreceptor.lmc_output, (400, 400)))
#     cv2.imshow('filtered', cv2.resize(photoreceptor.filtered_data, (400, 400)))
#     cv2.imshow('frame', fig)
#
#     plt.plot(test_emd.log)
#     if cv2.waitKey(1) == ord('q'):
#         print(test_emd.Rout)
#         print(test_emd.log)
#         plt.plot(test_emd.log)
#         plt.show()
#         break
#
# cap.release()
# cv2.destroyAllWindows()
