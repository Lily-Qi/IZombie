import warnings


class Epsilons:
    def __init__(
        self,
        seq_length,
        start_epsilon,
        end_epsilon,
        interpolation="exponential",
    ):
        self.seq_length = seq_length
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.interpolation = interpolation

        assert seq_length > 1, "seq_length must be positive"
        assert interpolation in ["exponential"], "not implemented"

        self.index = 0
        self.decay_rate = (self.end_epsilon / self.start_epsilon) ** (
            1 / (self.seq_length - 1)
        )
        self.has_warned = False

    def get(self):
        return self.start_epsilon * self.decay_rate**self.index

    def next(self):
        self.index += 1
        if self.index >= self.seq_length:
            self.index = self.seq_length - 1
            if not self.has_warned:
                self.has_warned = True
                warnings.warn(
                    f"index ({self.index}) ≥ seq_length ({self.seq_length}), using index = {self.seq_length - 1}."
                )