import html
import numpy as np
import torch
import re
import ipywidgets as widgets
from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from matplotlib import pyplot as plt
from matplotlib import colors


@dataclass
class TokenStyle:
    seq_index: Optional[Union[int, List[int]]] = None
    color: Optional[str] = None
    background: Optional[str] = None
    font_weight: Optional[str] = None
    font_style: Optional[str] = None
    font_size: Optional[str] = None


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color."""
    return "#{:02x}{:02x}{:02x}".format(*tuple(map(int, rgb)))


def color_to_rgb(color: str) -> tuple:
    """Convert a color name or hex color to an RGB tuple."""
    try:
        r, g, b = colors.to_rgb(color)
        return r * 255, g * 255, b * 255
    except ValueError:
        if color.startswith("#"):
            return hex_to_rgb(color)
        else:
            raise ValueError(f"Invalid color value: {color}")


def average_colors(colors: List[Optional[str]]) -> Optional[str]:
    """Average a list of color names or hex codes."""
    colors = [color for color in colors if color is not None]
    if not colors:
        return None
    rgb_values = [color_to_rgb(color) for color in colors]  # type: ignore[arg-type]
    avg_rgb = tuple(sum(c[i] for c in rgb_values) // len(rgb_values) for i in range(3))
    return rgb_to_hex(avg_rgb)


def average_font_sizes(font_sizes: List[Optional[str]]) -> Optional[str]:
    """Average a list of font sizes with 'em' units."""
    sizes = []
    for size in font_sizes:
        if size is not None:
            match = re.match(r"([\d.]+)em", size)
            if match:
                sizes.append(float(match.group(1)))
            else:
                raise ValueError(f"Unsupported font size format: {size}")
    if not sizes:
        return None
    avg_size = sum(sizes) / len(sizes)
    return f"{avg_size:.2f}em"


def render_seq_to_html(
    tokenizer,
    seq_ids: Union[np.ndarray, torch.Tensor],
    *token_styles: TokenStyle,
    special_token_style: Optional[TokenStyle] = None,
    override_strategy: Literal["mix", "new", "old"] = "new",
):
    """
    Render text with HTML styles applied to specified tokens.

    Args:
        tokenizer: The tokenizer used to decode and convert tokens.
        seq_ids (np.ndarray): Array of token IDs.
        *token_styles (TokenStyle): Variable length TokenStyle arguments.
        special_token_styles (TokenStyle, optional): Style for special tokens. Defaults to gray and italic.
        override_strategy (str): Strategy to handle overlapping styles: "mix", "new", "old".

    Returns:
        str: HTML formatted string with styles applied.
    """
    tokens = tokenizer.convert_ids_to_tokens(seq_ids.tolist())

    special_token_style = special_token_style or TokenStyle(
        color="gray", font_style="italic"
    )

    # Initialize an empty list to hold styles corresponding to each token
    style_list: List[Optional[TokenStyle]] = [None] * len(tokens)

    # Function to add or mix styles into style_list based on the strategy
    def add_style(token_index: int, new_style: TokenStyle):
        existing_style = style_list[token_index]
        if existing_style is None:
            style_list[token_index] = new_style
        else:
            if override_strategy == "mix":
                mixed_color = average_colors([existing_style.color, new_style.color])
                mixed_background = average_colors(
                    [existing_style.background, new_style.background]
                )
                mixed_font_size = average_font_sizes(
                    [existing_style.font_size, new_style.font_size]
                )

                # Create mixed style
                style_list[token_index] = TokenStyle(
                    seq_index=token_index,
                    color=mixed_color,
                    background=mixed_background,
                    font_weight=existing_style.font_weight,
                    font_style=existing_style.font_style,
                    font_size=mixed_font_size,
                )
            elif override_strategy == "new":
                style_list[token_index] = TokenStyle(
                    seq_index=token_index,
                    color=new_style.color or existing_style.color,
                    background=new_style.background or existing_style.background,
                    font_weight=new_style.font_weight or existing_style.font_weight,
                    font_style=new_style.font_style or existing_style.font_style,
                    font_size=new_style.font_size or existing_style.font_size,
                )
            elif override_strategy == "old":
                style_list[token_index] = TokenStyle(
                    seq_index=token_index,
                    color=existing_style.color or new_style.color,
                    background=existing_style.background or new_style.background,
                    font_weight=existing_style.font_weight or new_style.font_weight,
                    font_style=existing_style.font_style or new_style.font_style,
                    font_size=existing_style.font_size or new_style.font_size,
                )
            else:
                raise ValueError(f"Unsupported override_strategy: {override_strategy}")

    # Add normal token styles
    for token_style in token_styles:
        if token_style.seq_index is not None and (
            isinstance(token_style.seq_index, int) or len(token_style.seq_index) > 0
        ):
            indices = (
                token_style.seq_index
                if isinstance(token_style.seq_index, list)
                else [token_style.seq_index]
            )
            for index in indices:
                add_style(index, token_style)

    # Add special token styles if provided
    if hasattr(tokenizer, "all_special_ids"):
        for idx, token_id in enumerate(seq_ids):
            if token_id in tokenizer.all_special_ids:
                add_style(idx, special_token_style)

    rendered_tokens = []
    for i, token in enumerate(tokens):
        token_style = style_list[i]  # type: ignore[assignment]
        if token_style:
            style_dict = {
                "color": token_style.color,
                "background": token_style.background,
                "font-weight": token_style.font_weight,
                "font-style": token_style.font_style,
                "font-size": token_style.font_size,
            }

            style_str = "".join(
                f"{key}: {value};" for key, value in style_dict.items() if value
            )
        else:
            style_str = ""

        rendered_tokens.append(
            (
                f'<span style="{style_str}">{html.escape(token)}</span>'
                if style_str
                else html.escape(token)
            )
        )

    return re.sub(r"▁", " ", "".join(rendered_tokens))


class ControlPanel(widgets.VBox):
    def __init__(self, **kwargs):
        self.head_axis_enabled = widgets.Checkbox(
            value=False,
            description="Enable head axis",
            tooltip="Enable show attention weights of each head",
        )

        self.layer_axis_enabled = widgets.Checkbox(
            value=True,
            description="Enable layer axis",
            tooltip="Enable show attention weights of each layer",
        )

        self.figsize = widgets.Text(
            value="8,6",
            placeholder="w,h",
            description="Figure size",
            continuous_update=False,
        )

        self.tick_fontsize = widgets.IntText(
            value=7,
            description="Tick font size",
            continuous_update=False,
        )

        self.color_map = widgets.Dropdown(
            options=plt.colormaps(),
            value="viridis",
            description="Color map of heatmap",
            style={"description_width": "initial"},
        )

        self.highlighted_token_color = widgets.ColorPicker(
            value="#f0b8ff",
            description="Highlighted text color",
            style={"description_width": "initial"},
        )

        self.blend_alpha = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.1,
            continuous_update=False,
            readout=True,
            readout_format=".1f",
            description="Blend alpha",
            tooltip="Blend ratio of image and attention. The larger the alpha, the fainter the heatmap.",
        )

        grid_layout = widgets.GridspecLayout(n_rows=3, n_columns=3)
        grid_layout[0, 0] = self.layer_axis_enabled
        grid_layout[0, 1] = self.head_axis_enabled
        grid_layout[1, 0] = self.figsize
        grid_layout[1, 1] = self.tick_fontsize
        grid_layout[2, 0] = self.color_map
        grid_layout[2, 1] = self.highlighted_token_color
        grid_layout[2, 2] = self.blend_alpha

        super().__init__([grid_layout], **kwargs)

    @property
    def seq_seq_widgets(self):
        return [
            self.head_axis_enabled,
            self.layer_axis_enabled,
            self.figsize,
            self.tick_fontsize,
            self.color_map,
        ]

    @property
    def seq_img_widgets(self):
        return [
            self.head_axis_enabled,
            self.layer_axis_enabled,
            self.figsize,
            self.tick_fontsize,
            self.color_map,
            self.blend_alpha,
        ]
