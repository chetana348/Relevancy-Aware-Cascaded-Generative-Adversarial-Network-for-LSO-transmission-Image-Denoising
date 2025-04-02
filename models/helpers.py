# Pseudocode for helpers - to be imported to reGAN

class FeatureBlock:
    def __init__(self, input_size, output_size):
        # Initialize a block that transforms input features
        self.layer_stack = BuildBasicUnit(input_size, output_size)

    def forward(self, data):
        # Apply basic transformations
        return self.layer_stack(data)

class DownStage:
    def __init__(self, input_size, output_size):
        self.pool = DownsampleUnit()
        self.encode = FeatureBlock(input_size, output_size)

    def forward(self, input_tensor):
        reduced = self.pool(input_tensor)
        return self.encode(reduced)

class UpStage:
    def __init__(self, input_size, output_size):
        self.upsample = UpsampleUnit()
        self.decode = FeatureBlock(input_size, output_size)

    def forward(self, low_res, skip_connection):
        upscaled = self.upsample(low_res)
        merged = MergeTensors(upscaled, skip_connection)
        return self.decode(merged)

class RelevancyNetwork:
    def __init__(self, input_channels, output_channels):
        # Define encoder and decoder hierarchy
        self.encoder = [
            FeatureBlock(input_channels, 64),
            DownStage(64, 128),
            DownStage(128, 256)
        ]
        self.decoder = [
            UpStage(256, 128),
            UpStage(128, 64)
        ]
        # Prediction and relevancy heads
        self.main_output = FinalMapper(64, output_channels)
        self.relevancy_map = FinalMapper(64, 1)

    def forward(self, input_data):
        # Encoder pass
        skip_feats = []
        data = input_data
        for enc in self.encoder:
            data = enc.forward(data)
            skip_feats.append(data)

        # Decoder pass
        for i, dec in enumerate(self.decoder):
            data = dec.forward(data, skip_feats[-(i+2)])

        # Outputs
        prediction = self.main_output(data)
        relevancy = NormalizeMap(self.relevancy_map(data))
        return prediction, relevancy

class CascadedRefinement:
    def __init__(self, stages, channels):
        self.modules = [RelevancyNetwork(channels, channels) for _ in range(stages)]

    def forward(self, initial_input):
        current = initial_input
        for net in self.modules:
            output, attention = net.forward(current)
            current = Blend(output, current, attention)
        return output
