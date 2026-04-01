from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """单步 ConvLSTM，输入/隐藏同空间分辨率。"""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=pad,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        x: (B, C_in, H, W)
        returns: h, (h, c)
        """
        if state is None:
            b, _, h, w = x.shape
            h = torch.zeros(
                b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype
            )
            c = torch.zeros_like(h)
        else:
            h, c = state
        combined = torch.cat([x, h], dim=1)
        cc = self.gates(combined)
        i, f, g, o = torch.chunk(cc, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class HydroBaseline(nn.Module):
    """
    Level 2 基线：多层 ConvLSTM 编码时序 + 逐帧解码到 T_out。
    输入 (B, T_in, C_in, H, W)，输出 (B, T_out, C_out, H, W)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_in: int,
        t_out: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_element_attention: bool = True,
        element_attention_hidden: int = 64,
    ):
        super().__init__()
        self.t_out = t_out
        self.out_channels = out_channels
        self.use_element_attention = use_element_attention

        self.in_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(hidden_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
                for i in range(num_layers)
            ]
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if use_element_attention:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, element_attention_hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(element_attention_hidden, hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.se = None

        self.decoders = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
                for _ in range(t_out)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B T C H W
        _, t, _, _, _ = x.shape
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.cells)
        inp = None
        for ti in range(t):
            cur = self.in_proj(x[:, ti])
            inp = cur
            for li, cell in enumerate(self.cells):
                inp, states[li] = cell(inp, states[li])

        feat = self.drop(inp)
        if self.se is not None:
            w_att = self.se(feat)
            feat = feat * w_att

        outs = []
        for dec in self.decoders:
            outs.append(dec(feat))
        return torch.stack(outs, dim=1)
