import torch
from torch import nn
from torchvision.ops import roi_align


class RegionAwareGeometryHead(nn.Module):
    """Predict query-specific depth corrections from regions inside each predicted 2D box."""

    def __init__(
        self,
        hidden_dim,
        grid_size=(3, 3),
        output_dim=1,
        use_uncertainty=False,
        uncertainty_temperature=1.0,
        uncertainty_eps=1e-4,
    ):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.output_dim = output_dim
        self.use_uncertainty = use_uncertainty
        self.uncertainty_temperature = uncertainty_temperature
        self.uncertainty_eps = uncertainty_eps

        self.region_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.query_region_logits = nn.Linear(hidden_dim, self.grid_size[0] * self.grid_size[1])

        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.uncertainty_head = None

    def _boxes_to_rois(self, boxes, feature_h, feature_w):
        batch_size, num_queries, _ = boxes.shape
        boxes = boxes.detach().clamp(0.0, 1.0)

        x1 = torch.minimum(boxes[..., 0], boxes[..., 2]) * feature_w
        y1 = torch.minimum(boxes[..., 1], boxes[..., 3]) * feature_h
        x2 = torch.maximum(boxes[..., 0], boxes[..., 2]) * feature_w
        y2 = torch.maximum(boxes[..., 1], boxes[..., 3]) * feature_h

        x2 = torch.maximum(x2, x1 + 1.0)
        y2 = torch.maximum(y2, y1 + 1.0)

        batch_idx = torch.arange(batch_size, device=boxes.device, dtype=boxes.dtype)
        batch_idx = batch_idx[:, None].expand(batch_size, num_queries)
        rois = torch.stack([batch_idx, x1, y1, x2, y2], dim=-1)
        return rois.reshape(-1, 5)

    def forward(self, feature_map, query_features, query_boxes, region_mask=None):
        batch_size, num_queries, _ = query_features.shape
        feature_h, feature_w = feature_map.shape[-2:]
        rois = self._boxes_to_rois(query_boxes, feature_h, feature_w)

        roi_features = roi_align(
            feature_map,
            rois,
            output_size=self.grid_size,
            spatial_scale=1.0,
            aligned=True,
        )

        if region_mask is not None:
            roi_mask = roi_align(
                region_mask.float(),
                rois,
                output_size=self.grid_size,
                spatial_scale=1.0,
                aligned=True,
            ).clamp(0.0, 1.0)
            roi_features = roi_features * roi_mask

        roi_features = roi_features.flatten(2).transpose(1, 2)
        region_features = roi_features.reshape(
            batch_size,
            num_queries,
            self.grid_size[0] * self.grid_size[1],
            -1,
        )

        region_error = self.region_encoder(region_features)
        region_logits = self.query_region_logits(query_features)

        if self.uncertainty_head is not None:
            region_uncertainty = torch.nn.functional.softplus(
                self.uncertainty_head(region_features)
            ) + self.uncertainty_eps
            region_logits = region_logits - self.uncertainty_temperature * region_uncertainty.squeeze(-1)
        else:
            region_uncertainty = None

        region_weights = torch.softmax(region_logits, dim=-1)
        query_region_error = torch.sum(region_weights.unsqueeze(-1) * region_error, dim=2)

        output = {
            "region_geometry_error": region_error,
            "region_weights": region_weights,
            "query_region_geometry_error": query_region_error,
        }

        if region_uncertainty is not None:
            output["region_uncertainty"] = region_uncertainty

        return output
