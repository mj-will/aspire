from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class History:
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)

    def plot_loss(self):
        fig = plt.figure()
        plt.plot(self.training_loss, label="Training loss")
        plt.plot(self.validation_loss, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        return fig
