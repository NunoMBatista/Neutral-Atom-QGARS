import torch.nn as nn
from typing import Optional, Any
import re

# import all ANSI escape codes list


USE_COLORS = True

try:
    import colorama
    from colorama import Fore, Style
except ImportError:
    USE_COLORS = False


def print_sequential_model(model: nn.Sequential, model_name: Optional[str] = None, max_width: int = 80, use_colors: bool = USE_COLORS) -> str:
    """
    Pretty-print the layers of a PyTorch Sequential model and return the output as a string.
    
    Parameters
    ----------
    model : nn.Sequential
        The model to print
    model_name : Optional[str], optional
        Name of the model for the header, by default None
    max_width : int, optional
        Maximum width of the output, by default 80
    
    Returns
    -------
    str
        The formatted model summary as a string
    """
    # Set up colors if available
    if use_colors:
        colorama.init()
        c_header = Fore.CYAN
        c_highlight = Fore.YELLOW
        c_layer_header = Fore.GREEN
        c_layer_num = Fore.BLUE
        c_reset = Style.RESET_ALL
    else:
        c_header = c_highlight = c_layer_header = c_layer_num = c_reset = ""
    
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Prepare output string
    output = []
    
    # Print header
    header = f" Model Summary: {model_name or 'Sequential'} "
    output.append(f"\n{c_header}{header.center(max_width, '=')}{c_reset}")
    
    # Print model overview
    output.append(f"{c_highlight}Total parameters: {total_params:,}{c_reset}")
    output.append(f"{c_highlight}Trainable parameters: {trainable_params:,}{c_reset}")
    
    # Print layer header
    output.append(f"\n{c_layer_header}{'#':<5} {'Layer Type':<20} {'Param #':<15} {'Details'}{c_reset}")
    output.append("-" * max_width)
    
    # Iterate through layers
    for i, layer in enumerate(model):
        layer_type = layer.__class__.__name__
        layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        # Extract key attributes based on layer type
        details = ""
        if isinstance(layer, nn.Linear):
            details = f"in={layer.in_features}, out={layer.out_features}, bias={layer.bias is not None}"
        elif isinstance(layer, nn.Conv2d):
            details = f"in={layer.in_channels}, out={layer.out_channels}, k={layer.kernel_size}, s={layer.stride}"
        elif isinstance(layer, nn.BatchNorm1d):
            details = f"features={layer.num_features}, eps={layer.eps:.0e}, momentum={layer.momentum:.1f}"
        elif isinstance(layer, nn.BatchNorm2d):
            details = f"features={layer.num_features}, eps={layer.eps:.0e}, momentum={layer.momentum:.1f}"
        elif isinstance(layer, nn.Dropout):
            details = f"p={layer.p}"
        elif isinstance(layer, nn.LeakyReLU):
            details = f"negative_slope={layer.negative_slope}"
        elif hasattr(layer, "inplace") and isinstance(layer.inplace, bool):
            details = f"inplace={layer.inplace}"
        
        # Print the layer info with parameter count (or -- if no parameters)
        param_count = f"{layer_params:,}" if layer_params > 0 else "--"
        output.append(f"{c_layer_num}{i:<5}{c_reset} {layer_type:<20} {param_count:<15} {details}")
    
    # Print footer
    output.append("-" * max_width)
    output.append(f"{c_header}{'=' * max_width}{c_reset}\n")
    
    # Reset colorama if needed
    if use_colors:
        colorama.deinit()
    
    return "\n".join(output)


def print_quantum_layer(layer: Any, max_width: int = 80, use_colors: bool = USE_COLORS) -> str:
    """
    Pretty-print the parameters of a DetuningLayer and return the output as a string.

    Parameters
    ----------
    layer : DetuningLayer
        The quantum layer to print
    max_width : int, optional
        Maximum width of the output, by default 80

    Returns
    -------
    str
        The formatted quantum layer summary as a string
    """
    # Set up colors if available
    if use_colors:
        colorama.init()
        c_header = Fore.CYAN
        c_highlight = Fore.YELLOW
        c_reset = Style.RESET_ALL
    else:
        c_header = c_highlight = c_reset = ""

    # Prepare output string
    output = []

    # Print header
    header = " Quantum Layer Summary "
    output.append(f"\n{c_header}{header.center(max_width, '=')}{c_reset}")


    # Extract parameters from the DetuningLayer
    params = layer.qrc_params
    geometry = str(params.get("geometry_spec", None))

    # Remove escape color codes if use_colors is False
    if not use_colors:
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        geometry = '\n'.join(ansi_escape.sub('', line) for line in geometry.splitlines())


    output.append(f"{c_highlight}Geometry: {geometry}{c_reset}")

    output.append(f"{c_highlight}Number of Atoms: {params['atom_number']}{c_reset}")
    output.append(f"{c_highlight}Lattice Spacing: {params['geometry_spec'].lattice_spacing} μm{c_reset}")
    output.append(f"{c_highlight}Rabi Frequency: {params['rabi_frequency']} Hz{c_reset}")
    output.append(f"{c_highlight}Total Evolution Time: {params['total_time']} s{c_reset}")
    output.append(f"{c_highlight}Number of Time Steps: {params['time_steps']}{c_reset}")
    output.append(f"{c_highlight}Readout Type: {params['readouts']}{c_reset}")
    output.append(f"{c_highlight}Encoding Scale: {params['encoding_scale']}{c_reset}")

    # Print footer
    output.append(f"{c_header}{'=' * max_width}{c_reset}\n")

    # Reset colorama if needed
    if use_colors:
        colorama.deinit()

    return "\n".join(output)