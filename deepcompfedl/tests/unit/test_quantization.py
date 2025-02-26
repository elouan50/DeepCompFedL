"""
Unit tests for the quantization module in the deepcompfedl.compression.quantization package.
This module contains tests for the following functionalities:
- Layer-wise quantization
- Global quantization
- Quantization with no change when the number of bits is large enough
Classes:
    TestQuantization: A unittest.TestCase subclass that contains methods to test the quantization functionality.
Methods:
    setUp(self):
        Set up sample parameters for testing.
    test_quantize_layer_scale(self):
        Test layer-wise quantization by checking if the quantized parameters have the same shape as the original
        parameters and if the quantized values are within the range of the original values.
    test_quantize_global_scale(self):
        Test global quantization by checking if the quantized parameters have the same shape as the original
        parameters and if the quantized values are within the range of the original values.
    test_quantize_no_change(self):
        Test quantization with a large number of bits (nbits=8) to ensure that the quantized parameters are the same
        as the original parameters.
"""

import unittest
import numpy as np
from deepcompfedl.compression.quantization import quantize

class TestQuantization(unittest.TestCase):
    
    def setUp(self):
        # Set up some sample parameters for testing
        self.params = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]])
        ]
    
    def test_quantize_layer_scale(self):
        # Test layer-wise quantization
        quantized_params = quantize(self.params, nbits=2, layer_scale=True)
        
        # Check if the output has the same shape as the input
        for original, quantized in zip(self.params, quantized_params):
            self.assertEqual(original.shape, quantized.shape)
        
        # Check if the quantized values are within the range of original values
        for original, quantized in zip(self.params, quantized_params):
            self.assertTrue(np.all(quantized >= np.min(original)))
            self.assertTrue(np.all(quantized <= np.max(original)))
    
    def test_quantize_global_scale(self):
        # Test global quantization
        quantized_params = quantize(self.params, nbits=2, layer_scale=False)
        
        # Check if the output has the same shape as the input
        for original, quantized in zip(self.params, quantized_params):
            self.assertEqual(original.shape, quantized.shape)
        
        # Check if the quantized values are within the range of original values
        for original, quantized in zip(self.params, quantized_params):
            self.assertTrue(np.all(quantized >= np.min(original)))
            self.assertTrue(np.all(quantized <= np.max(original)))
    
    def test_quantize_no_change(self):
        # Test quantization with nbits large enough to cause no change
        quantized_params = quantize(self.params, nbits=8, layer_scale=True)
        
        # Check if the output is the same as the input
        for original, quantized in zip(self.params, quantized_params):
            np.testing.assert_array_equal(original, quantized)

if __name__ == '__main__':
    unittest.main()