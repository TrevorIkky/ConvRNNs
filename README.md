# ConvRNNs
Implementation of ConvLSTM, ConvGRU. Similar to canonical LSTMs and GRUs. However, their fully connected parts have been replaced with convolution operations.
`ConvLSTM` and `ConvGRU` inherit from `BaseConvRNN` which is a `nn.Module`. Therefore, the two classes provided can be used in tandem with other pytorch modules for
tasks such as video frame prediction.
