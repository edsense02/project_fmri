import torch
import torch.nn as nn


class MRITransformer(nn.Module):
    """
    Transformer encoder model for MRI VQ-VAE tokens.

    - Input:  input_ids of shape (batch_size, seq_len), integers in [0, vocab_size-1]
              (0..511 are real codebook tokens, 512 is the [MASK] token).
    - Output: logits of shape (batch_size, seq_len, vocab_size)

    Supports:
    - 1D positional encoding: standard sequence positions 0..L-1
    - 2D positional encoding: row + col embeddings for latent grid (H_lat x W_lat)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int | None = None,
        pos_encoding: str = "1d",  # "1d" or "2d"
        latent_height: int | None = None,
        latent_width: int | None = None,
    ):
        super().__init__()

        assert pos_encoding in ("1d", "2d"), "pos_encoding must be '1d' or '2d'"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pos_encoding = pos_encoding
        self.latent_height = latent_height
        self.latent_width = latent_width

        if self.pos_encoding == "2d":
            if latent_height is None or latent_width is None:
                raise ValueError(
                    "latent_height and latent_width must be provided for 2D positional encoding."
                )
            if max_seq_len is None:
                max_seq_len = latent_height * latent_width
        else:
            if max_seq_len is None:
                raise ValueError("max_seq_len must be provided for 1D positional encoding.")

        self.max_seq_len = max_seq_len

        # 1) Token embeddings: codebook IDs -> d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # 2) Positional embeddings
        #    1D positions 0..max_seq_len-1
        self.pos_embed_1d = nn.Embedding(max_seq_len, d_model)

        #    2D row / col embeddings for latent grid
        if self.pos_encoding == "2d":
            self.row_embed = nn.Embedding(latent_height, d_model)
            self.col_embed = nn.Embedding(latent_width, d_model)
        else:
            self.row_embed = None
            self.col_embed = None

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # input is (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Prediction head: hidden -> logits over vocab
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Tie weights between token embedding and output layer
        self.lm_head.weight = self.token_embed.weight

    def _build_positional_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build positional embeddings of shape (1, seq_len, d_model),
        matching self.pos_encoding.
        """
        if self.pos_encoding == "1d":
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len} for 1D encoding."
                )
            positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
            pos_emb = self.pos_embed_1d(positions)  # (1, seq_len, d_model)
            return pos_emb

        # 2D encoding
        H, W = self.latent_height, self.latent_width
        if H * W != seq_len:
            raise ValueError(
                f"2D encoding expects seq_len == H*W ({H}*{W}={H*W}), got {seq_len}."
            )

        # Row/col indices
        row_idx = torch.arange(H, device=device).unsqueeze(1).expand(H, W)  # (H, W)
        col_idx = torch.arange(W, device=device).unsqueeze(0).expand(H, W)  # (H, W)

        row_emb = self.row_embed(row_idx)  # (H, W, d_model)
        col_emb = self.col_embed(col_idx)  # (H, W, d_model)
        pos_2d = row_emb + col_emb         # (H, W, d_model)

        pos_emb = pos_2d.view(1, H * W, self.d_model)  # (1, seq_len, d_model)
        return pos_emb

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids:      (batch_size, seq_len) int64
        attention_mask: (batch_size, seq_len) float or bool
                        1 / True = keep, 0 / False = ignore (optional)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embed(input_ids)  # (B, L, d_model)

        # Positional embeddings (1D or 2D)
        pos_emb = self._build_positional_embeddings(seq_len, device)  # (1, L, d_model)

        hidden = token_emb + pos_emb  # (B, L, d_model)

        # Build key_padding_mask for TransformerEncoder:
        # expects True for positions that should be ignored.
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # (B, L) bool

        enc_out = self.encoder(hidden, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)
        logits = self.lm_head(enc_out)  # (B, L, vocab_size)

        return logits
