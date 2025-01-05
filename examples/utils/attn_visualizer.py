import ipywidgets as widgets
import PIL
from typing import List, Optional
from .control_panel import ControlPanel
from .display_panel import DisplayPanel


class AttnVisualizer(widgets.Tab):
    def __init__(
        self,
        attn_weights,
        seq_ids,
        tokenizer,
        images: Optional[List[PIL.Image.Image]] = None,
        image_token: Optional[str] = "<image>",
    ):
        control_panel = ControlPanel()

        super().__init__(
            children=[
                DisplayPanel(
                    control_panel=control_panel,
                    attn_weights=attn_weights,
                    seq_ids=seq_ids,
                    tokenizer=tokenizer,
                    images=images,
                    image_token=image_token,
                ),
                control_panel,
            ],
            titles=["Attention Heatmap", "Control Panel"],
        )
