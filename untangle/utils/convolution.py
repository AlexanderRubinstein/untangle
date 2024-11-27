"""Same padding and output padding utilities."""


def calculate_output_padding(
    input_shape: list[int],
    output_shape: list[int],
    stride: list[int],
    padding: list[int],
    kernel_size: list[int],
    dilation: list[int] | None,
) -> tuple[int, int]:
    """Calculates output padding for transposed convolution operations.

    This function is an adaptation of PyTorch's output padding calculation, see
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L629.

    Args:
        input_shape: Shape of the input tensor.
        output_shape: Desired shape of the output tensor.
        stride: Stride of the convolution.
        padding: Padding applied to the input.
        kernel_size: Size of the convolving kernel.
        dilation: Dilation rate of the convolution.

    Returns:
        A tuple of integers representing the output padding.

    Raises:
        ValueError: If the requested output size is not within the valid range.
    """
    num_spatial_dims = 2  # [H, W]
    num_non_spatial_dims = 2  # [B, C]

    # Assume channels_last layout
    output_shape = output_shape[num_non_spatial_dims:]  # [H, W]

    min_sizes = []
    max_sizes = []
    for spatial_dim in range(num_spatial_dims):
        dim_size = (
            (input_shape[num_non_spatial_dims + spatial_dim] - 1) * stride[spatial_dim]
            - 2 * padding[spatial_dim]
            + (dilation[spatial_dim] if dilation is not None else 1)
            * (kernel_size[spatial_dim] - 1)
            + 1
        )
        min_sizes.append(dim_size)
        max_sizes.append(min_sizes[spatial_dim] + stride[spatial_dim] - 1)

    for i in range(len(output_shape)):
        size = output_shape[i]
        min_size = min_sizes[i]
        max_size = max_sizes[i]
        if size < min_size or size > max_size:
            msg = (
                f"requested an output size of {output_shape}, but valid sizes range "
                f"from {min_sizes} to {max_sizes} "
                f"(for an input of {input_shape[2:]})"
            )
            raise ValueError(msg)

    res = tuple(
        output_shape[spatial_dim] - min_sizes[spatial_dim]
        for spatial_dim in range(num_spatial_dims)
    )

    return res


def calculate_same_padding(
    input_shape: tuple[int, int, int, int],
    filter_shape: tuple[int, int, int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> tuple[int, int, int, int]:
    """Calculates padding values for 'SAME' padding in conv2d operations.

    Args:
        input_shape: Shape of the input data [batch, channels, height, width].
        filter_shape: Shape of the filter/kernel [out_channels, in_channels,
            kernel_height, kernel_width].
        stride: Stride of the convolution operation.
        dilation: Dilation rate of the convolution operation.

    Returns:
        A tuple representing padding
        (padding_left, padding_right, padding_top, padding_bottom).
    """
    if isinstance(stride, int):
        stride_height = stride_width = stride
    else:
        stride_height, stride_width = stride

    if isinstance(dilation, int):
        dilation_height, dilation_width = dilation, dilation
    else:
        dilation_height, dilation_width = dilation

    in_height, in_width = input_shape[2], input_shape[3]
    filter_height, filter_width = filter_shape[2], filter_shape[3]

    effective_filter_height = filter_height + (filter_height - 1) * (
        dilation_height - 1
    )
    effective_filter_width = filter_width + (filter_width - 1) * (dilation_width - 1)

    if in_height % stride_height == 0:
        pad_along_height = max(effective_filter_height - stride_height, 0)
    else:
        pad_along_height = max(effective_filter_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(effective_filter_width - stride_width, 0)
    else:
        pad_along_width = max(effective_filter_width - (in_width % stride_width), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom
