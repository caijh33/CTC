import csv
import numpy as np
import torch
import time

class Timer(object):
	"""
	docstring for Timer
	"""
	def __init__(self):
		super(Timer, self).__init__()
		self.total_time = 0.0
		self.calls = 0
		self.start_time = 0.0
		self.diff = 0.0
		self.average_time = 0.0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average = False):
		self.diff = time.time() - self.start_time
		self.calls += 1
		self.total_time += self.diff
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff

	def format(self, time):
		m,s = divmod(time, 60)
		h,m = divmod(m, 60)
		d,h = divmod(h, 24)
		return ("{}d:{}h:{}m:{}s".format(int(d), int(h), int(m), int(s)))

	def end_time(self, extra_time):
		"""
		calculate the end time for training, show local time
		"""
		localtime= time.asctime(time.localtime(time.time() + extra_time))
		return localtime

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size

class MixUp(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def mixup_data(self, x, y, use_cuda=True):
        """
        return mixed inputs. pairs of targets
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class TrainingHelper(object):
    def __init__(self, image):
        self.image = image
    def congratulation(self):
        """
        if finish training success, print congratulation information
        """
        for i in range(40):
            print('*')*i
            print('finish training')

def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))