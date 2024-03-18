# -*- coding: utf-8 -*-
# architecture of network
# below the CNN is the architecture for a skip-CNN

import torch
from torch import nn


class CNN(torch.nn.Module):
    def __init__(self, n_input_channels: int, n_hidden_layers: int, n_kernels: int, kernel_size: int):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_input_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding='same',
                padding_mode='replicate'
            ))
            cnn.append(torch.nn.ReLU())
            n_input_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_input_channels,
            out_channels=3,
            kernel_size=kernel_size,
            padding='same',
            padding_mode='replicate'
        )
    
    def forward(self,
                input_tensor: torch.Tensor,
                known_tensor: torch.tensor,
                offsets: torch.tensor,
                spacings: torch.tensor):
        process_tensor = torch.concat((input_tensor, known_tensor[:, 0][:, None, :, :]), 1).float()
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(process_tensor)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 3, X, Y)
        output = torch.relu_(pred)
        return output


class SkipCNN(nn.Module):
    def __init__(self, n_input_channels: int, n_hidden_layers: int, n_kernels: int, kernel_size: int):
        """CNN, consisting of "n_hidden_layers" linear layers, using relu
        activation function in the hidden CNN layers.

        Parameters
        ----------
        n_input_channels: int
            Number of features channels in input tensor
        n_hidden_layers: int
            Number of conv. layers
        n_kernels: int
            Number of kernels in each layer
        kernel_size: int
            Number of features in output tensor
        """
        super().__init__()

        layers = []
        n_concat_channels = n_input_channels
        for i in range(n_hidden_layers):
            # Add a CNN layer
            layer = nn.Conv2d(
                in_channels=n_concat_channels,
                out_channels=3,
                kernel_size=kernel_size,
                padding='same',
                padding_mode='replicate'
            )
            layers.append(layer)
            self.add_module(f"conv_{i:0{len(str(n_hidden_layers))}d}", layer)
            # Prepare for concatenated input
            n_concat_channels = n_kernels + n_input_channels
            n_input_channels = n_kernels

        self.layers = layers

    def forward(self,
                input_tensor: torch.Tensor,
                known_tensor: torch.tensor,
                offsets: torch.tensor,
                spacings: torch.tensor):
        """Apply CNN to "x"

        Parameters
        ----------
        process_tensor: torch.Tensor
            Input tensor of shape (n_samples, n_input_channels, x, y)

        Returns
        ----------
        torch.Tensor
            Output tensor of shape (n_samples, n_output_channels, u, v)
        """

        process_tensor = torch.concat((input_tensor, known_tensor[:, 0][:, None, :, :]), 1).float()

        skip_connection = None
        output = None

        # Apply layers module
        for layer in self.layers:
            # If previous output and skip_connection exist, concatenate
            # them and store previous output as new skip_connection. Otherwise,
            # use process_tensor as input and store it as skip_connection.
            if skip_connection is not None:
                assert output is not None
                inp = torch.cat([output, skip_connection], dim=1)
                skip_connection = output
            else:
                inp = process_tensor
                skip_connection = process_tensor
            # Apply CNN layer
            output = torch.relu_(layer(inp))

        return output
