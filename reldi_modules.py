import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class WeightedHyperbolicLoss(nn.Module):
    """Horizon-weighted MSE with target-magnitude scaling.
    The implementation of the hyperbolic weight curve in the loss function is inspired by the findings of Aguado et al. (https://doi.org/10.1029/2009JA014658), who demonstrated that the recovery phase of the Dst index during geomagnetic storms follows a hyperbolic decay rather than an exponential one. Intense storms (large negative Dst values) exhibit faster initial recovery rates that gradually slow, a critical feature captured by hyperbolic decay.

    In machine learning contexts, extreme Dst values (e.g., Dst<-100nT) are rare but physically significant, as they represent severe geomagnetic disturbances. Standard loss functions may underweight these events due to their sparse occurrence. The hyperbolic weighting scheme addresses this by assigning higher weights to intense Dst values, ensuring the model prioritizes their accurate prediction. The weight for a Dst value yy is defined as:
    Weight(y)=1 / [α⋅(1-y~)+y~']


    Where y~ is the normalized Dst value (scaled to [0,1][0,1]) and increase of the weight. This inversely proportional relationship emphasizes rarer, extreme values (closer to y~=0) while smoothly leaving the weights for common, less critical values unchanged.

    Additionally, since later timesteps are harder to predict, we apply a linear ramp to the weights, where earlier horizons are weighted by *w₁* and the last horizon by *w_S*. The final weight for each timestep is given by:
    Weight(y)=1 / [α⋅(1-y~)+y~'] * w₁ + (w_S - w₁) * (timestep / S)
    where *timestep* is the current timestep (0 to S-1) and *S* is the total number of timesteps.

    Args
    ----
    scaled_range : tuple(float,float)
        Min / max of *scaled* Dst (after StandardScaler).
    hyperparam_hyperbolic   : float
        Controls how sharply weights grow toward extreme values.
    timestep_weights : (w₁,w_S)
        Linear ramp - earlier horizons weight *w₁*, last horizon *w_S*.
    output_timesteps : int  (S)
    device : torch.device
    """

    def __init__(
        self,
        scaled_range,
        hyperparam_hyperbolic,
    ):
        super().__init__()
        self.register_buffer(
            "scaled_range", torch.tensor(scaled_range, dtype=torch.float32)
        )
        self.hyperparam = hyperparam_hyperbolic
        self.div_term = self.scaled_range[1] - self.scaled_range[0]
        # (S,) tensor broadcast to (B,S,1)
        

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE.

        Shapes
        ------
        y_true, y_pred : (B, S, 1)
        """
        # Normalize y_true to be within [0, 1]
        y_true_normalized = (y_true - self.scaled_range[0]) / self.div_term
        y_true_normalized = torch.clamp(y_true_normalized, 0, 1)

        # Calculate weights for the loss
        hyper_weights = 1 / (
            self.hyperparam * (1 - y_true_normalized) + y_true_normalized
        )

        # Calculate the squared error
        base_loss = (y_true - y_pred) ** 2

        # Compute weighted loss
        weighted_loss = base_loss * hyper_weights

        # Return the mean loss across all samples
        return torch.mean(weighted_loss)
    

class WeightedHyperbolicLossMAE(nn.Module):
    """Horizon-weighted MAE with target-magnitude scaling.
    The implementation of the hyperbolic weight curve in the loss function is inspired by the findings of Aguado et al. (https://doi.org/10.1029/2009JA014658), who demonstrated that the recovery phase of the Dst index during geomagnetic storms follows a hyperbolic decay rather than an exponential one. Intense storms (large negative Dst values) exhibit faster initial recovery rates that gradually slow, a critical feature captured by hyperbolic decay.

    In machine learning contexts, extreme Dst values (e.g., Dst<-100nT) are rare but physically significant, as they represent severe geomagnetic disturbances. Standard loss functions may underweight these events due to their sparse occurrence. The hyperbolic weighting scheme addresses this by assigning higher weights to intense Dst values, ensuring the model prioritizes their accurate prediction. The weight for a Dst value yy is defined as:
    Weight(y)=1 / [α⋅(1-y~)+y~']


    Where y~ is the normalized Dst value (scaled to [0,1][0,1]) and increase of the weight. This inversely proportional relationship emphasizes rarer, extreme values (closer to y~=0) while smoothly leaving the weights for common, less critical values unchanged.

    Additionally, since later timesteps are harder to predict, we apply a linear ramp to the weights, where earlier horizons are weighted by *w₁* and the last horizon by *w_S*. The final weight for each timestep is given by:
    Weight(y)=1 / [α⋅(1-y~)+y~'] * w₁ + (w_S - w₁) * (timestep / S)
    where *timestep* is the current timestep (0 to S-1) and *S* is the total number of timesteps.

    Args
    ----
    scaled_range : tuple(float,float)
        Min / max of *scaled* Dst (after StandardScaler).
    hyperparam_hyperbolic   : float
        Controls how sharply weights grow toward extreme values.
    timestep_weights : (w₁,w_S)
        Linear ramp - earlier horizons weight *w₁*, last horizon *w_S*.
    output_timesteps : int  (S)
    device : torch.device
    """

    def __init__(
        self,
        scaled_range,
        hyperparam_hyperbolic,
    ):
        super().__init__()
        self.register_buffer(
            "scaled_range", torch.tensor(scaled_range, dtype=torch.float32)
        )
        self.hyperparam = hyperparam_hyperbolic
        self.div_term = self.scaled_range[1] - self.scaled_range[0]
        # (S,) tensor broadcast to (B,S,1)
        

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE.

        Shapes
        ------
        y_true, y_pred : (B, S, 1)
        """
        # Normalize y_true to be within [0, 1]
        y_true_normalized = (y_true - self.scaled_range[0]) / self.div_term
        y_true_normalized = torch.clamp(y_true_normalized, 0, 1)

        # Calculate weights for the loss
        hyper_weights = 1 / (
            self.hyperparam * (1 - y_true_normalized) + y_true_normalized
        )

        # Calculate the squared error
        base_loss = torch.abs(y_true - y_pred)    

        # Compute weighted loss
        weighted_loss = base_loss * hyper_weights

        # Return the mean loss across all samples
        return torch.mean(weighted_loss)



class WeightedQuantileLoss(nn.Module):
    """Adding Pinball (quantile) loss to the previous hyperbolic loss."""

    def __init__(
        self,
        q,
        scaled_range,
        hyperparam_hyperbolic,
        timestep_weights,
        output_timesteps,
        device,
    ):
        super(WeightedQuantileLoss, self).__init__()
        if not 0 < q < 1:
            raise ValueError("Quantile q must be between 0 and 1.")
        self.q = q
        self.scaled_range = torch.tensor(
            scaled_range, dtype=torch.float32, device=device
        )
        self.hyperparam = torch.tensor(
            hyperparam_hyperbolic, dtype=torch.float32, device=device
        )
        self.div_term = self.scaled_range[1] - self.scaled_range[0]
        self.timestep_weights = torch.linspace(
            timestep_weights[0],
            timestep_weights[1],
            output_timesteps,
            dtype=torch.float32,
            device=device,
        )

    def forward(self, y_true, y_pred):
        # Normalize y_true to [0, 1] for hyperbolic weighting
        y_true_normalized = (y_true - self.scaled_range[0]) / self.div_term
        y_true_normalized = torch.clamp(y_true_normalized, 0, 1)

        # Hyperbolic weights (value-based importance)
        hyper_weights = 1 / (
            self.hyperparam * (1 - y_true_normalized) + y_true_normalized
        )

        # Quantile Loss calculation
        error = y_true - y_pred
        loss = torch.max(self.q * error, (self.q - 1) * error)

        # Combine hyperbolic weights and timestep weights
        combined_weights = hyper_weights * self.timestep_weights[
            : y_true.shape[1]
        ].view(1, -1, 1)

        # Final weighted loss
        weighted_loss = loss * combined_weights

        return torch.mean(weighted_loss)


class BFE_Metric():
    """
    BFE_Metric is a custom metric class that calculates the Binned Forecasting Error (BFE) for a given set of predictions and true values.
    The BFE is computed by dividing the absolute differences between the true values and the predictions into specified bins, summing the differences within each bin, and then averaging these sums.
    Args:
        bins (list or tensor): The bin edges used to categorize the true values.
    Methods:
        update(y_true, y_pred):
            Updates the internal state with the absolute differences between the true values and the predictions.
        compute():
            Computes the BFE across all bins.
        reset_state():
            Resets the internal state of the metric.
    Example:
        metric = BFE_Metric(bins=[0, 1, 2, 3])
        metric.update(y_true, y_pred)
        result = metric.compute()
        metric.reset_state()
    """

    def __init__(self, bins, device, **kwargs):
        super().__init__(**kwargs)
        self.bins = torch.tensor(bins, dtype=torch.float32, device=device)
        self.bins_sum = torch.zeros_like(self.bins, dtype=torch.float32, device=device)
        self.bins_count = torch.zeros_like(
            self.bins, dtype=torch.float32, device=device
        )

    def update(self, y_true, y_pred) -> None:
        diffs = torch.abs(y_true - y_pred)
        y_true_flat = y_true.view(-1)
        diffs_flat = diffs.view(-1)
        labels_bins = torch.bucketize(y_true_flat, self.bins) - 1

        for i in range(len(labels_bins)):
            self.bins_sum[labels_bins[i]] += diffs_flat[i]
            self.bins_count[labels_bins[i]] += 1

    def compute(self):
        bfe_values = torch.div(self.bins_sum, self.bins_count)
        return torch.mean(bfe_values[~torch.isnan(bfe_values)])

    def reset_state(self):
        self.bins_sum = torch.zeros_like(self.bins_sum)
        self.bins_count = torch.zeros_like(self.bins_count)
        
# utils.py  ──────────────────────────────────────────────────────────────
def causal_mask(input_tensor : torch.Tensor) -> torch.Tensor:
    """
    Returns a (size, size) upper-triangular mask filled with -inf above
    the diagonal and 0 on / below it.  Compatible with nn.MultiheadAttention.
    """
    mask = torch.triu(torch.ones(input_tensor.size(1), input_tensor.size(1), device=input_tensor.device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )

    def forward(self, x):
        # Calculate the required padding on the left
        if x.dim() == 2:
            x = x.unsqueeze(0)
        pad = (self.kernel_size - 1) * self.dilation
        # Apply padding (only on the left)
        x = F.pad(x, (pad, 0))  # (left_pad, right_pad)
        return self.conv1d(x)
    

class Encoder(nn.Module):
    """
    based on https://doi.org/10.1029/2023SW003485

    -----------------
    1. **Causal Conv stack** - captures local temporal patterns.
    2. **Self-attention**    - long-range dependencies within look-back window.
    3. **Bi-LSTM**           - directional context.

    Output
    ------
    A single vector per batch of size **2·D** (forward & backward last hidden).
    Shape: *(B, 2·D)*
    """

    def __init__(
        self,
        input_dim_solar_wind: int,
        input_dim_ldi: int,
        d_model: int,
        conv_kernels,
        dilation=(1, 1, 1),
        p_dropout=0.1,
    ):
        super().__init__()

        # ─── Causal convolution stack ────────────────────────────────────────
        self.conv_block_solar_wind = nn.Sequential(
            CausalConv1d(
                input_dim_solar_wind,
                d_model,
                conv_kernels[0],
                dilation=dilation[0],
            ),
            nn.SiLU(),
            CausalConv1d(
                d_model,
                d_model,
                conv_kernels[1],
                dilation=dilation[1],
            ),
            nn.SiLU(),
            CausalConv1d(
                d_model,
                d_model,
                conv_kernels[2],
                dilation=dilation[2],
            ),
            nn.SiLU(),
        )
        self.conv_res_solar_wind = (
            nn.Identity()
            if input_dim_solar_wind == d_model
            else nn.Conv1d(input_dim_solar_wind, d_model, 1)
        )

        # ─── Sequence modelling layers ──────────────────────────────────────
        self.mha = nn.MultiheadAttention(
            d_model,
            4,
            batch_first=True,
            dropout=p_dropout,
            kdim=d_model + input_dim_ldi,
            vdim=d_model + input_dim_ldi,
        )
        self.bilstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input_solar_wind, input_ldi): 
        # → (B, F, T) for Conv1d

        input_solar_wind = input_solar_wind.permute(0, 2, 1)
        input_solar_wind_conv = self.conv_block_solar_wind(input_solar_wind)
        input_solar_wind = self.dropout(
            input_solar_wind_conv + self.conv_res_solar_wind(input_solar_wind)
        )
        input_solar_wind = input_solar_wind.permute(0, 2, 1)  # back to (B,T,D)

        kv = torch.cat([input_solar_wind, input_ldi], dim=2)  # (B,T,2·D)

        attn, _ = self.mha(input_solar_wind, kv, kv)

        x = self.dropout(attn + input_solar_wind)  # residual

        bi_out, (bi_last_hidden, bi_last_cell) = self.bilstm(x)  # (B,T,2·D)
        bi_last_hidden = bi_last_hidden.permute(1, 0, 2)

        bi_last_hidden = self.dropout(bi_last_hidden)

        return bi_last_hidden

class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_feat_ldi,
        p_dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        
        self.input_projection_ldi = nn.Sequential(nn.Linear(1, d_model // 2), nn.SiLU())
        self.input_projection_metadata = nn.Sequential(
            nn.Linear(num_feat_ldi - 1, d_model // 2), nn.SiLU()
        )
        self.self_attn = nn.MultiheadAttention(
            d_model // 2,
            4,
            batch_first=True,
            kdim=d_model,
            vdim=d_model,
            dropout=p_dropout,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, 4, batch_first=True, dropout=p_dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.output_proj = nn.Linear(d_model, 1)
        self.quantile_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, decoder_input, encoder_context):

        ldi_prev = decoder_input[..., :1]
        metadata = decoder_input[..., 1:]

        ldi = self.input_projection_ldi(ldi_prev)  # (B,s,1) -> (B,s,D/2)
        metadata = self.input_projection_metadata(metadata)  # (B,s,D/2) -> (B,s,D/2)
        kv = torch.cat([metadata, ldi], dim=-1)
        
        mask = causal_mask(decoder_input)
        attn, _ = self.self_attn(query=ldi, key=kv, value=kv, attn_mask=mask)

        x = torch.cat([ldi, attn], dim=-1)

        ctx, _ = self.cross_attn(
            x,
            encoder_context,
            encoder_context,  # self.enc_proj(encoder_context), self.enc_proj(encoder_context)
        )
        x = x + ctx
        x = x + self.ffn(x)
        return self.output_proj(x), self.quantile_proj(x)


    
class RELDi(nn.Module):
    """
    RELDi: A transformer-based model for predicting the Dst index from solar wind and interplanetary magnetic field data.
    """

    def __init__(
        self,
        input_dim_solar_wind,
        input_dim_ldi,
        d_model,
    ):
        super().__init__()        
        self.encoder = Encoder(
            input_dim_solar_wind,
            input_dim_ldi,
            d_model,
            conv_kernels=(7, 5, 3),
        )
        
        self.decoder = Decoder(
            d_model,
            input_dim_ldi,
        )
        
    def _encode(self, encoder_input_solar_wind, encoder_input_ldi):
        x = self.encoder(encoder_input_solar_wind, encoder_input_ldi)
        return x

    def forward(
        self, encoder_input_solar_wind, encoder_input_ldi, decoder_input_teacher_forcing
    ):
        encoder_context = self._encode(encoder_input_solar_wind, encoder_input_ldi)

        point_forecast, quantile_forecasts = self.decoder(
            decoder_input_teacher_forcing, encoder_context
        )

        return (
            point_forecast,
            quantile_forecasts[:, :, 0:1],
            quantile_forecasts[:, :, 1:2],
        )

    @torch.jit.export
    def inference(
        self,
        encoder_input_solar_wind,
        encoder_input_ldi,
        decoder_input_ldi_first,
        decoder_future_mlts,
    ):
        encoder_context = self._encode(encoder_input_solar_wind, encoder_input_ldi)

        point_outputs, quantile_lower, quantile_upper = [], [], []
        timestep_input = torch.cat(
            (decoder_input_ldi_first, decoder_future_mlts[:, :1, :]), dim=-1
        )

        for t in range(decoder_future_mlts.size(1)):
            point_forecast, quantile_forecast = self.decoder(
                timestep_input, encoder_context
            )
            if t < decoder_future_mlts.size(1) - 1:
                cur = torch.cat(
                    (
                        point_forecast[:, -1, :].unsqueeze(1),
                        decoder_future_mlts[:, t + 1, :].unsqueeze(1),
                    ),
                    dim=-1,
                )
                timestep_input = torch.cat([timestep_input, cur], 1)

            point_outputs.append(point_forecast[:, -1:, :])
            quantile_lower.append(quantile_forecast[:, -1:, 0:1])
            quantile_upper.append(quantile_forecast[:, -1:, 1:2])
        return (
            torch.cat(point_outputs, 1),
            torch.cat(quantile_lower, 1),
            torch.cat(quantile_upper, 1),
        )   

