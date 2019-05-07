import torch


class MultiItemAverageMeter:

    def __init__(self):
        self.content = {}

    def update(self, val):
        '''
        :param val: dict, keys are strs, values are torch.Tensor or np.array
        '''
        for key in val.keys():
            value = val[key]
            if key not in self.content.keys():
                self.content[key] = {'avg': value, 'sum': value, 'count': 1.0}
            else:
                self.content[key]['sum'] += value
                self.content[key]['count'] += 1.0
                self.content[key]['avg'] = self.content[key]['sum'] / self.content[key]['count']

    def get_val(self):
        keys = self.content.keys()
        values = []
        for key in keys:
            try:
                values.append(self.content[key]['avg'].data.cpu().numpy())
            except:
                values.append(self.content[key]['avg'])
        return keys, values

    def get_str(self):

        result = ''
        keys, values = self.get_val()

        for key, value in zip(keys, values):
            result += key
            result += ': '
            result += str(value)
            result += ';  '

        return result

