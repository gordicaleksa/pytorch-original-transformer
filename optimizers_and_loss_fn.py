class CustomLearningRateAdamOptimizer:
    """
        Linear ramp for the warm-up number of steps and then start decaying
        according to the inverse square root law of the current training step number
    """

    def __init__(self, optimizer, model_dimension, num_of_warmup_steps):
        self.optimizer = optimizer
        self.model_size = model_dimension
        self.num_of_warmup_steps = num_of_warmup_steps

        self.current_step_number = 0
        self.current_learning_rate = 0

    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.get_current_learning_rate()

        for p in self.optimizer.param_groups:
            p['lr'] = current_learning_rate

        self.current_learning_rate = current_learning_rate
        self.optimizer.step()  # apply gradients

    # Check out the formula at Page 7, Chapter 5.3 "Optimizer" and playground.py for visualization
    def get_current_learning_rate(self):
        # For readability purpose
        step = self.current_step_number
        warmup = self.num_of_warmup_steps

        return self.model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
