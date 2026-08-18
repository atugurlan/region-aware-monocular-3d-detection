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
        fusion_type="logit",
        use_query_gate=False,
        gate_init=0.0,
        use_reliability=False,
        reliability_mode="auxiliary_only",
        reliability_logit_scale=1.0,
    ):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.output_dim = output_dim
        self.use_uncertainty = use_uncertainty
        self.uncertainty_temperature = uncertainty_temperature
        self.uncertainty_eps = uncertainty_eps
        self.fusion_type = fusion_type
        self.use_query_gate = use_query_gate
        self.use_reliability = use_reliability
        self.reliability_mode = reliability_mode
        self.reliability_logit_scale = reliability_logit_scale

        self.region_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.query_region_logits = nn.Linear(hidden_dim, self.grid_size[0] * self.grid_size[1])

        if fusion_type == "adaptive":
            self.adaptive_region_logits = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.adaptive_region_logits = None

        if use_query_gate:
            self.query_depth_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.query_depth_gate[-1].weight)
            nn.init.constant_(self.query_depth_gate[-1].bias, gate_init)
        else:
            self.query_depth_gate = None

        if use_reliability:
            self.reliability_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.reliability_head = None

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
        if self.adaptive_region_logits is not None:
            query_region_features = query_features.unsqueeze(2).expand_as(region_features)
            fusion_features = torch.cat(
                [
                    region_features,
                    query_region_features,
                    region_features * query_region_features,
                ],
                dim=-1,
            )
            region_logits = self.adaptive_region_logits(fusion_features).squeeze(-1)
        else:
            region_logits = self.query_region_logits(query_features)

        if self.reliability_head is not None:
            region_reliability_logits = self.reliability_head(region_features)
            region_reliability = torch.sigmoid(region_reliability_logits)
            if self.reliability_mode == "logit_bias":
                region_logits = region_logits + self.reliability_logit_scale * region_reliability.squeeze(-1)
        else:
            region_reliability_logits = None
            region_reliability = None

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

        if self.query_depth_gate is not None:
            output["query_region_depth_gate"] = torch.tanh(self.query_depth_gate(query_features))

        if region_reliability is not None:
            output["region_reliability"] = region_reliability
            output["region_reliability_logits"] = region_reliability_logits

        if region_uncertainty is not None:
            output["region_uncertainty"] = region_uncertainty

        return output
