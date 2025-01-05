import PIL
import PIL.Image
import ipywidgets as widgets
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import html
import numpy as np
import torch
from typing import List, Optional, Tuple

from utils.control_panel import ControlPanel, render_seq_to_html, TokenStyle


def extract_image_attn(
    seq_ids: torch.Tensor,
    attn_weights: torch.Tensor,
    image_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Extract image tokens from the sequence. This method collapses consecutive image tokens into one in seq_ids,
    and returns the indices of original image tokens for each image group. For the collapsed attention weights,
    image tokens are averaged instead of summed.

    Args:
        seq_ids (torch.Tensor):
            The sequence ids. Shape: (seq_len,)
        attn_weights (torch.Tensor):
            The attention weights. Shape: (num_layer, num_head, seq_len, seq_len)
        image_token_id (int):
            The image token id.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
            - collapsed_seq_ids: The sequence ids with collapsed image tokens. Shape: (new_seq_len,)
            - collapsed_attn_weights: The attention weights adjusted for the collapsed sequence. Shape: (num_layer, num_head, new_seq_len, new_seq_len)
            - image_indices: A list containing tensors of the original indices in seq_ids for each image group.
                             Each element in the list is a tensor of shape (group_size,)
    """
    # Step 1: Identify image token groups
    is_image = seq_ids == image_token_id
    if not is_image.any():
        return seq_ids, attn_weights, []

    # Identify the start of each image token group
    shifted = torch.roll(is_image, shifts=1)
    shifted[0] = False
    image_starts = is_image & ~shifted
    # Identify the end of each image token group
    shifted_back = torch.roll(is_image, shifts=-1)
    shifted_back[-1] = False
    image_ends = is_image & ~shifted_back

    # Get start and end indices
    start_indices = torch.nonzero(image_starts, as_tuple=False).squeeze(-1)
    end_indices = torch.nonzero(image_ends, as_tuple=False).squeeze(-1)

    if torch.all(start_indices == end_indices):
        return seq_ids, attn_weights, [start_indices]

    # Collect image indices
    image_indices = [
        torch.arange(start, end + 1, device=seq_ids.device)
        for start, end in zip(start_indices, end_indices)
    ]

    # Step 2: Create a mask to keep non-image tokens and the first token of each image group
    keep = ~is_image | image_starts  # bool tensor
    collapsed_seq_ids = seq_ids[keep]  # shape: (new_seq_len,)
    new_seq_len = collapsed_seq_ids.size(0)

    # Step 3: Create a mapping from old indices to new indices
    mapping = torch.cumsum(keep.int(), dim=0) - 1
    group_start = torch.where(
        image_starts,
        torch.arange(len(seq_ids), device=seq_ids.device),
        torch.tensor(0, device=seq_ids.device),
    )
    group_start, _ = torch.cummax(group_start, dim=0)
    non_kept_image_index = is_image & ~image_starts
    mapping[non_kept_image_index] = mapping[group_start[non_kept_image_index]]

    group_sizes = torch.tensor(
        [len(indices) - 1 for indices in image_indices],
        dtype=torch.float32,
        device=seq_ids.device,
    )
    scaling = torch.ones(new_seq_len, device=seq_ids.device)
    print
    scaling[mapping[torch.nonzero(keep & image_starts, as_tuple=False).squeeze(-1)]] = (
        1.0 / group_sizes
    )

    # Step 4: Collapse attention weights using einsum
    new_seq_range = torch.arange(new_seq_len, device=seq_ids.device).unsqueeze(0)
    key_map = (mapping.unsqueeze(1) == new_seq_range).float() * scaling
    query_map = (mapping.unsqueeze(1) == new_seq_range).float() * scaling
    collapsed_attn_weights = torch.einsum(
        "l h i k, i m -> l h m k",
        torch.einsum("l h i j, j k -> l h i k", attn_weights, key_map),
        query_map,
    )

    return collapsed_seq_ids, collapsed_attn_weights, image_indices


class DisplayPanel(widgets.VBox):
    def __init__(
        self,
        control_panel: ControlPanel,
        attn_weights: torch.Tensor,
        seq_ids: torch.Tensor,
        tokenizer,
        images: Optional[List[PIL.Image.Image]] = None,
        image_token: Optional[str] = "<image>",
    ):
        self._control_panel = control_panel
        self._tokenizer = tokenizer
        self._seq_ids = seq_ids
        self._attn_weights = attn_weights
        self._image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        # try to collapse image tokens into one token of seq_ids and attn_weights
        self._collpased_seq_ids, self._collpased_attn_weights, self._image_indices = (
            extract_image_attn(
                seq_ids,
                attn_weights,
                self._image_token_id,
            )
        )

        if self._seq_ids.shape == self._collpased_seq_ids.shape:
            # there is no consecutive image tokens, which means the sequence is not from an autoregressive model,
            # or the image token is not in the sequence
            if images is not None:
                raise ValueError(
                    "No consecutive image tokens are collapsed, this sequence may come from a cross attention architecture model."
                    "Only auto-regressive models are supported to visualize image attentions."
                )
            self._images = []
        else:
            if images is None or len(images) != len(self._image_indices):
                raise ValueError(
                    "The number of images provided and used in model inputs doesn't match."
                )
            self._images = [img.convert("RGBA") for img in images]

        common_controls = self._init_common_controls()

        self.display_frame = widgets.Stack(
            children=[self._init_seq_seq_frame()]
            + ([self._init_seq_img_frame()] if self.architecture == "auto-reg" else []),
            selected_index=0,
        )

        super().__init__(
            children=[
                common_controls,
                self.display_frame,
            ]
        )

    def _get_plot_metadata(self, attn_weights):
        start_idx, end_idx = self.seq_range_slider.value
        try:
            figsize = tuple(map(int, self._control_panel.figsize.value.split(",")))
        except Exception:
            figsize = (8, 6)

        layer_idx = (
            (
                self.layer_slider.value
                if self._control_panel.layer_axis_enabled.value
                else None
            ),
        )
        head_idx = (
            self.head_slider.value
            if self._control_panel.head_axis_enabled.value
            else None
        )

        # slice by layer and head
        attn_data = (
            attn_weights[:, head_idx]
            if head_idx is not None
            else torch.mean(attn_weights, dim=1)
        )

        attn_data = (
            attn_data[layer_idx]
            if layer_idx is not None
            else torch.mean(attn_data, dim=0)
        )

        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "tick_fontsize": self._control_panel.tick_fontsize.value,
            "cmap": self._control_panel.color_map.value,
            "hover_token_color": self._control_panel.highlighted_token_color.value,
            "figsize": figsize,
            "attn_data": attn_data,
            "blend_alpha": self._control_panel.blend_alpha.value,
        }

    def _init_seq_seq_frame(self):
        heatmap_fig = go.FigureWidget(
            go.Heatmap(colorbar=dict(title="Attention Weight"))
        )
        heatmap_fig.update_layout(xaxis_title="Key", yaxis_title="Query")

        def hover_event(trace, points, state):
            if points.point_inds:
                start_idx, end_idx = self.seq_range_slider.value
                token_idx = points.point_inds[0]
                self.selected_text.value = render_seq_to_html(
                    self._tokenizer,
                    self._collpased_seq_ids[start_idx : end_idx + 1],
                    TokenStyle(
                        seq_index=token_idx,
                        background=self._control_panel.highlighted_token_color.value,
                    ),
                )

        heatmap_fig.data[0].on_hover(hover_event)

        def update_plot(change=None):
            meta = self._get_plot_metadata(self._collpased_attn_weights)
            attn_data = (
                meta["attn_data"][
                    meta["start_idx"] : meta["end_idx"] + 1,
                    meta["start_idx"] : meta["end_idx"] + 1,
                ]
                .cpu()
                .numpy()
            )
            subseq = self._collpased_seq_ids[meta["start_idx"] : meta["end_idx"] + 1]
            ticklabels = self._tokenizer.convert_ids_to_tokens(subseq)
            ticklabels = [html.escape(t) for t in ticklabels]

            with heatmap_fig.batch_update():
                heatmap_fig.data[0].update(
                    z=attn_data,
                    colorscale=meta["cmap"],
                )

                heatmap_fig.update_layout(
                    xaxis=dict(
                        tickvals=list(range(len(ticklabels))),
                        ticktext=ticklabels,
                        tickfont=dict(size=int(meta["tick_fontsize"])),
                    ),
                    yaxis=dict(
                        tickvals=list(range(len(ticklabels))),
                        ticktext=ticklabels,
                        tickfont=dict(size=int(meta["tick_fontsize"])),
                    ),
                    height=meta["figsize"][1] * 100,
                    width=meta["figsize"][0] * 100,
                )

        controls = self._control_panel.seq_seq_widgets + [
            self.seq_range_slider,
            self.head_slider,
            self.layer_slider,
        ]

        for control in controls:
            control.observe(update_plot, names="value")

        update_plot()

        return heatmap_fig

    def _init_seq_img_frame(self):
        heatmap_fig = go.FigureWidget(go.Image())
        collapse_tokens = self._tokenizer.convert_ids_to_tokens(self._collpased_seq_ids)

        self.selected_token = widgets.SelectionSlider(
            options=[("dummy", 0)],  # will be updated later
            value=0,
            description="Selected Token:",
            continuous_update=False,
            style={"description_width": "initial"},
        )
        self.selected_token.observe(self._update_selected_text, "value")

        def seq_range_change(change):
            start_idx, end_idx = change["new"]
            self.selected_token.options = [
                (token, idx)
                for idx, token in enumerate(collapse_tokens[start_idx : end_idx + 1])
                if self._tokenizer.convert_tokens_to_ids(token) != self._image_token_id
            ]

            self.selected_token.value = min(
                max(start_idx, self.selected_token.value), end_idx
            )

        seq_range_change({"new": self.seq_range_slider.value})
        self.seq_range_slider.observe(seq_range_change, "value")

        self.selected_image = widgets.IntSlider(
            min=0,
            max=len(self._images) - 1,
            step=1,
            value=0,
            description="Selected Image:",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        def update_plot(change=None):
            seq_idx_in_collapse = self.selected_token.value
            if seq_idx_in_collapse == 0:
                seq_idx = 0
            else:
                n_img_before_selected_idx = torch.cumsum(
                    self._collpased_seq_ids[:seq_idx_in_collapse]
                    == self._image_token_id,
                    dim=0,
                )[-1]
                seq_idx = (
                    seq_idx_in_collapse
                    + sum(
                        len(indices)
                        for indices in self._image_indices[
                            : n_img_before_selected_idx + 1
                        ]
                    )
                    - n_img_before_selected_idx
                )

            meta = self._get_plot_metadata(self._attn_weights[:, :, seq_idx, :])
            # get the corresponding attention for the selected image
            attn_data = meta["attn_data"][
                self._image_indices[self.selected_image.value]
            ].view(int(len(self._image_indices[self.selected_image.value]) ** 0.5), -1)
            assert attn_data.dim() == 2 and attn_data.size(0) == attn_data.size(1)

            # resize attention matrix to match the image size
            attn_data = (
                torch.nn.functional.interpolate(
                    attn_data.unsqueeze(0).unsqueeze(0) / attn_data.max(),
                    size=self._images[self.selected_image.value].size,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .squeeze()
            )

            attn_heatmap = plt.get_cmap(meta["cmap"])(attn_data.cpu().numpy())
            # PIL.Image.fromarray will swap height and width dim, so we transpose inputs
            img = PIL.Image.blend(
                PIL.Image.fromarray(
                    np.astype(attn_heatmap * 255, np.uint8).transpose((1, 0, 2))
                ),
                self._images[self.selected_image.value],
                meta["blend_alpha"],
            )

            with heatmap_fig.batch_update():
                heatmap_fig.data[0].update(
                    z=img,
                )
                heatmap_fig.update_layout(
                    height=meta["figsize"][1] * 100,
                    width=meta["figsize"][0] * 100,
                )

        controls = self._control_panel.seq_img_widgets + [
            self.selected_image,
            self.selected_token,
            self.head_slider,
            self.layer_slider,
        ]

        for control in controls:
            control.observe(update_plot, names="value")

        update_plot()

        return widgets.VBox(
            [
                widgets.HBox([self.selected_token, self.selected_image]),
                heatmap_fig,
            ]
        )

    def _init_common_controls(self):
        num_layer, num_head, _, seq_len = self._collpased_attn_weights.shape
        widget_width = "33%" if self.architecture == "cross-attn" else "25%"
        self.layer_slider = widgets.IntSlider(
            min=0,
            max=num_layer - 1,
            step=1,
            value=0,
            description="Layer:",
            continuous_update=False,
            layout=widgets.Layout(
                width=widget_width,
                display=(
                    "auto" if self._control_panel.layer_axis_enabled.value else "none"
                ),
            ),
        )

        self.head_slider = widgets.IntSlider(
            min=0,
            max=num_head - 1,
            step=1,
            value=0,
            description="Head:",
            continuous_update=False,
            layout=widgets.Layout(
                width=widget_width,
                display=(
                    "auto" if self._control_panel.head_axis_enabled.value else "none"
                ),
            ),
        )

        self.seq_range_slider = widgets.IntRangeSlider(
            value=[0, seq_len - 1],
            min=0,
            max=seq_len - 1,
            step=1,
            description="Seq Range:",
            continuous_update=False,
        )

        self.selected_text = widgets.HTML(
            render_seq_to_html(
                tokenizer=self._tokenizer,
                seq_ids=self._collpased_seq_ids,
            ),
            description="Selected Text:",
        )

        def head_layer_enabled_change(change):
            slider = (
                self.head_slider
                if change["owner"] == self._control_panel.head_axis_enabled
                else self.layer_slider
            )
            if change["new"]:
                slider.layout.display = "block"
            else:
                slider.layout.display = "none"

        self._control_panel.head_axis_enabled.observe(
            head_layer_enabled_change, "value"
        )
        self._control_panel.layer_axis_enabled.observe(
            head_layer_enabled_change, "value"
        )

        self.seq_range_slider.observe(self._update_selected_text, names="value")

        children = [
            self.seq_range_slider,
            self.layer_slider,
            self.head_slider,
        ]

        if self.architecture == "auto-reg" and len(self._image_indices) > 0:
            self.display_type = widgets.ToggleButtons(
                options=[("Seq-Seq", 0), ("Seq-Image", 1)],
                tooltips=[
                    "Display attention weights among sequence tokens",
                    "Display attention weights between sequence tokens and image tokens",
                ],
                value=0,
                style={"button_width": "7em"},
            )

            def display_frame_change(change):
                if change["new"] == 1:  # 0, 1 denotes seq-seq, seq-img
                    self.display_frame.selected_index = 1
                else:
                    self.display_frame.selected_index = 0
                self._update_selected_text(change)

            self.display_type.observe(display_frame_change, "value")

            children.append(
                widgets.HBox(
                    [widgets.Label("Display on:"), self.display_type],
                    layout=widgets.Layout(width=widget_width),
                )
            )

        return widgets.VBox([widgets.HBox(children), self.selected_text])

    @property
    def architecture(self):
        return "auto-reg" if self._images else "cross-attn"

    def _update_selected_text(self, change):
        start_idx, end_idx = self.seq_range_slider.value
        if self.display_frame.selected_index == 1:
            styles = [
                TokenStyle(
                    seq_index=self.selected_token.value,
                    background=self._control_panel.highlighted_token_color.value,
                )
            ]
        else:
            styles = []
        self.selected_text.value = render_seq_to_html(
            self._tokenizer,
            self._collpased_seq_ids[start_idx : end_idx + 1],
            *styles,
        )
