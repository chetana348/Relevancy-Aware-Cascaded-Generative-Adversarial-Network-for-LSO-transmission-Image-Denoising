# PSEUDOCODE: reGAN
# This pseudocode illustrates a deep learning architecture for PET to CT (denoising) tasks.
# It includes residual convolutional blocks, U-Net variants with multiple outputs, a generator with ResNet backbone, and a patch-based discriminator.


CLASS BasicResidualBlock:
    INIT(input_features):
        DEFINE convolution_sequence:
            - Pad input with 1 pixel (reflection padding)
            - Apply Conv2D with same input/output features
            - Normalize (instance norm)
            - ReLU
            - Repeat padding, conv, norm

        STORE the sequence as block

    FORWARD(input_tensor):
        RETURN input_tensor + block(input_tensor)

# Encoder with residual conv block and downsampling
CLASS EncoderStage:
    INIT(in_ch, out_ch, mid_ch OPTIONAL):
        IF mid_ch NOT provided: mid_ch = out_ch

        DEFINE primary_block AS SEQUENCE:
            - Conv(in_ch, mid_ch) → BN → ReLU
            - Conv(mid_ch, mid_ch) → BN → ReLU
            - Conv(mid_ch, out_ch) → BN → ReLU

        DEFINE skip_projection:
            - Conv(in_ch, out_ch) → BN → ReLU

    FORWARD(x):
        y_main = primary_block(x)
        y_skip = skip_projection(x)
        RETURN y_main + y_skip

# Downsampling + Encoder
CLASS DownBlock:
    INIT(input_channels, output_channels):
        SEQUENCE:
            - MaxPool2D
            - EncoderStage(input_channels, output_channels)

    FORWARD(x):
        RETURN sequence(x)

# Upsampling with optional bilinear resize + concat + conv
CLASS UpBlock:
    INIT(input_channels, output_channels, use_bilinear = TRUE):
        IF use_bilinear:
            USE Upsample
            USE EncoderStage
        ELSE:
            USE ConvTranspose2d + EncoderStage

    FORWARD(x1, x2):
        UPSAMPLE x1 to match spatial size of x2
        PAD if needed
        CONCATENATE x1 and x2 along channel axis
        RETURN convolution block

# 1x1 convolution for output prediction
CLASS OutputProjection:
    INIT(input_channels, output_channels):
        DEFINE Conv2D(kernel_size=1)

    FORWARD(x):
        RETURN conv(x)

# Full Encoder-Decoder network (U-Net style)
CLASS BaseUPlus:
    INIT(input_channels, output_channels, bilinear=True):
        Construct 4 down blocks + 4 up blocks
        Use OutputProjection at the end

    FORWARD(x):
        Collect encoder outputs
        Apply upsampling blocks with skip connections
        RETURN final prediction

# Cascaded U-Net where outputs are reused
CLASS CascadedUPlus:
    INIT(num_stages, channels, bilinear):
        CREATE ModuleList of BaseUPlus networks

    FORWARD(input_image):
        y = input_image
        FOR each stage i:
            IF i == 0:
                y = UPlus_i(y)
            ELSE:
                y = UPlus_i(y + input_image)
        RETURN y

# Variants with additional uncertainty estimation heads
CLASS CascadedUPlus2Head:
    LAST stage is UPlus2Head:
        Output: mean + variance

CLASS CascadedUPlus3Head:
    LAST stage is UPlus3Head:
        Output: mean + alpha + beta

# UPlus with two output heads (mean + uncertainty)
CLASS UPlus2Head:
    SAME structure as BaseUPlus
    OUTPUT:
        - Mean: Conv2D
        - Variance: 2x Conv2D (e.g., 64→128→1)

    FORWARD:
        Process through encoder-decoder
        RETURN mean_output, variance_output

# UPlus with three output heads (mean + two parameters for GGD)
CLASS UPlus3Head:
    SAME structure
    OUTPUT:
        - Mean: Conv2D
        - Alpha: Conv2D → ReLU
        - Beta: Conv2D → ReLU

    FORWARD:
        RETURN mean_output, alpha_output, beta_output

# Generator with ResNet blocks for super-resolution or translation
CLASS FlexibleGenerator:
    INIT(input_channels, output_channels, residual_blocks=9):
        SEQUENCE:
            - Initial reflection padding + conv → norm → ReLU
            - 2 downsampling stages (Conv2D stride 2)
            - N ResNet blocks (BasicResidualBlock)
            - 2 upsampling stages (TransposeConv2D stride 2)
            - Output layer with Tanh

    FORWARD(x):
        RETURN model(x)

# ResNet-style generator with configurable depth
CLASS ResnetBasedGenerator:
    INIT(input_ch, output_ch, filters=64, norm, dropout, num_blocks, padding):
        Start with Conv → Norm → ReLU
        2 downsample convs
        ADD num_blocks ResnetBlocks
        2 upsample transposed convs
        END with final Conv2D + Tanh

    FORWARD(x):
        RETURN sequential model output

# Single ResNet block used in generators
CLASS FlexibleResBlock:
    INIT(dim, padding, norm_layer, use_dropout, use_bias):
        ConvBlock = [Pad, Conv, Norm, ReLU] x 2
        Optional: Dropout

    FORWARD(x):
        RETURN x + conv_block(x)

# Patch-level discriminator (PatchGAN-style)
CLASS PatchDiscriminator:
    INIT(input_channels, base_filters=64, layers=3, norm_layer):
        INITIAL Conv + LeakyReLU
        FOR i IN range(1, layers):
            INCREASE channel size exponentially
            ADD: Conv → Norm → LeakyReLU

        FINAL Conv: maps to 1-channel patch output

    FORWARD(x):
        RETURN model(x)
