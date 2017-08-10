class ModifiedIterator(object):
    def __init__(self, iterator, batch_modifier):
        self.iterator = iterator
        self.batch_modifier = batch_modifier

    def reset_rng(self):
        self.iterator.reset_rng()

    def get_batches(self, dataset, shuffle=False):
        for inputs, targets in self.iterator.get_batches(dataset, shuffle=shuffle):
            inputs, targets = self.batch_modifier.process(inputs, targets)
            yield inputs, targets

