import unittest
import torch
from torch import Tensor
from cr_boosters.adam import Adam, adam
import math

class TestAdam(unittest.TestCase):
    def test_adam_initialization(self):
        params = [torch.tensor([1.0, 2.0], requires_grad=True)]
        optimizer = Adam(params)
        self.assertEqual(optimizer.defaults['lr'], 1e-3)

    def test_adam_function(self):
        params = [torch.tensor([1.0, 2.0], requires_grad=True)]
        grads = [torch.tensor([0.1, 0.1])]
        exp_avgs = [torch.zeros_like(params[0])]
        exp_avg_sqs = [torch.zeros_like(params[0])]
        max_exp_avg_sqs = [torch.zeros_like(params[0])]
        state_steps = [torch.tensor(0)]
        
        adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad=False, beta1=0.9, beta2=0.999, lr=0.001, weight_decay=0, eps=1e-8, maximize=False)
           
        # Calculate expected values based on the Adam update rule
        beta1, beta2 = 0.9, 0.999
        lr, eps = 0.001, 1e-8
        exp_avg = 0.1 * (1 - beta1)
        exp_avg_sq = 0.1 * 0.1 * (1 - beta2)
        bias_correction1 = 1 - beta1
        bias_correction2 = 1 - beta2
        step_size = lr / bias_correction1
        denom = (math.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps
        expected_value = 1.0 - step_size * (exp_avg / denom)

        self.assertAlmostEqual(params[0][0].item(), expected_value, places=4)
        self.assertAlmostEqual(params[0][1].item(), expected_value, places=4)

if __name__ == '__main__':
    unittest.main()