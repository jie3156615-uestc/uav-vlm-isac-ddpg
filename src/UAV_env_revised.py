import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gym
except ModuleNotFoundError:
    class _FallbackEnv:
        pass

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()


random.seed(958030)
np.random.seed(958030)


class GateMechanism(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.f_s = nn.Linear(feature_dim, 1)
        self.f_g = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.f_s.weight)
        nn.init.constant_(self.f_s.bias, -0.55)
        nn.init.zeros_(self.f_g.weight)
        nn.init.zeros_(self.f_g.bias)

    def forward(self, vlm_feat, isac_feat):
        gate = torch.sigmoid(self.f_s(vlm_feat) + self.f_g(isac_feat))
        fused = gate * vlm_feat + (1.0 - gate) * isac_feat
        return fused, gate


def angle_difference(angle_a, angle_b):
    return math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))


class UAVEnv(gym.Env):
    AUX_STATE_DIM = 23
    TRUE_GUIDE_FEATURES_PER_GT = 3
    TRUE_GUIDE_STRENGTH_BY_MODE = {
        3: 1.00,  # VLM-ISAC fusion keeps the strongest true direction cue.
        0: 0.80,
        1: 0.55,
        2: 0.32,
    }

    @classmethod
    def state_dim_for(cls, num_ground_stations):
        return 3 + num_ground_stations * 3 + cls.AUX_STATE_DIM + num_ground_stations * cls.TRUE_GUIDE_FEATURES_PER_GT

    @classmethod
    def guide_strength_for(cls, mode):
        return cls.TRUE_GUIDE_STRENGTH_BY_MODE.get(int(mode), 0.0)

    def __init__(
        self,
        num_ground_stations,
        ground_length,
        ground_width,
        height,
        scene="urban",
        position_mode="preset",
        random_seed=None,
        train_gate_online=True,
    ):
        super(UAVEnv, self).__init__()
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_ground_stations = num_ground_stations
        self.ground_length = ground_length
        self.ground_width = ground_width
        self.height_max = height
        self.height_min = 100
        self.scene = self._normalize_scene_name(scene)
        self.position_mode = position_mode
        self.random_seed = random_seed
        self.train_gate_online = train_gate_online

        self.B = 2 * 10 ** 6
        self.t = 0.5
        self.vel_h_max = 10
        self.vel_v_max = 10
        self.large_paper8_mode = str(position_mode).lower() in {
            "large_paper8",
            "large_cluster8",
            "large_urban_cluster8",
            "large_dense8",
            "large_highrise8",
            "large_urban12",
        }
        self.uav_start = [
            0.06 * self.ground_length if self.large_paper8_mode else 5.0,
            0.06 * self.ground_width if self.large_paper8_mode else 5.0,
            min(190.0, self.height_max),
        ]
        self.initial_battery = 58280 / 2
        self.e_battery_uav = self.initial_battery
        self.distance_scale = max(math.sqrt(self.ground_length ** 2 + self.ground_width ** 2), 1e-6)

        self.P_tx = 0.005
        self.P_isac = 1.5
        self.P_vlm = 5.0
        self.P_gate = 0.25
        self.P_ctrl = 0.2
        self.t_isac = 0.03
        self.t_vlm = 0.08
        self.t_gate = 0.005
        self.t_ctrl = 0.002
        self.progress_reward_weight = 18.0
        self.proximity_reward_weight = 0.8
        self.nearest_reward_weight = 0.5
        self.centroid_progress_reward_weight = 7.0
        self.centroid_proximity_reward_weight = 0.6
        self.service_region_bonus_weight = 0.5
        self.our_heading_alignment_weight = 0.35
        self.our_smoothness_penalty_weight = 0.45
        self.perception_error_penalty_weight = 0.0
        if self.large_paper8_mode:
            reward_scale = min(self.distance_scale / math.sqrt(200.0 ** 2 + 200.0 ** 2), 5.0)
            self.progress_reward_weight *= reward_scale
            self.centroid_progress_reward_weight *= reward_scale
            self.our_heading_alignment_weight *= 1.35
            self.perception_error_penalty_weight = 10.0
        position_key = str(position_mode).lower()
        self.vlm_noise_std = 12.0
        self.isac_noise_std = 3.0
        self.noisac_noise_std = 20.0
        self.vlm_angle_noise_std = math.radians(18.0)
        self.isac_angle_noise_std = math.radians(7.0)
        self.noisac_angle_noise_std = math.radians(28.0)
        self.noisac_heading_assist_weight = 0.35
        self.large_paper_noisac_min_horizontal = 0.0
        self.large_paper_isac_heading_assist_weight = 0.0
        self.large_paper_vlm_heading_assist_weight = 0.0
        self.large_paper_our_heading_assist_weight = 0.0
        self.large_paper_vlm_rate_scale = 1.0
        if self.large_paper8_mode:
            self.noisac_heading_assist_weight = 0.25
            self.large_paper_noisac_min_horizontal = 0.35
            self.large_paper_our_heading_assist_weight = 1.0
            if position_key == "large_paper8":
                self.large_paper_vlm_heading_assist_weight = 0.25
            elif position_key == "large_dense8":
                self.noisac_noise_std = 14.0
                self.noisac_angle_noise_std = math.radians(18.0)
                self.noisac_heading_assist_weight = 0.32
                self.large_paper_noisac_min_horizontal = 0.34
                self.large_paper_isac_heading_assist_weight = 0.80
                self.large_paper_vlm_heading_assist_weight = 0.45
            elif position_key == "large_urban12":
                self.noisac_heading_assist_weight = 0.18
                self.large_paper_noisac_min_horizontal = 0.28
                self.large_paper_isac_heading_assist_weight = 0.28
                self.large_paper_vlm_heading_assist_weight = 0.35
                self.large_paper_our_heading_assist_weight = 0.38
            elif position_key == "large_highrise8":
                self.noisac_heading_assist_weight = 0.18
                self.large_paper_noisac_min_horizontal = 0.28
                self.large_paper_isac_heading_assist_weight = 0.80
                self.large_paper_vlm_heading_assist_weight = 0.45
                # High-rise VLM-only localization lacks ISAC-grade beam alignment.
                self.large_paper_vlm_rate_scale = 0.25

        self.scene_params = {
            "suburban": {"a": 4.88, "b": 0.43, "eta_los": 0.1, "eta_nlos": 21.0},
            "urban": {"a": 9.61, "b": 0.16, "eta_los": 1.0, "eta_nlos": 20.0},
            "dense": {"a": 12.08, "b": 0.11, "eta_los": 1.6, "eta_nlos": 23.0},
            "highrise": {"a": 27.23, "b": 0.08, "eta_los": 2.3, "eta_nlos": 34.0},
        }

        self.gate_module = GateMechanism(feature_dim=1).to(self.device)
        self.gate_optimizer = optim.Adam(self.gate_module.parameters(), lr=1e-3)
        self.gate_loss_fn = nn.MSELoss()

        self.base_positions = self._build_initial_positions()
        self.loc_uav = list(self.uav_start)
        self.ground_stations = [pos.copy() for pos in self.base_positions]
        self.prev_heading = None
        self.prev_horizontal_action = None
        self.prev_vertical_action = None

    def _normalize_scene_name(self, scene):
        aliases = {
            "dense_urban": "dense",
            "dense-urban": "dense",
            "high_rise": "highrise",
            "high-rise": "highrise",
        }
        normalized = aliases.get(str(scene).lower(), str(scene).lower())
        if normalized not in {"suburban", "urban", "dense", "highrise"}:
            raise ValueError(f"Unsupported scene '{scene}'.")
        return normalized

    def _build_initial_positions(self):
        preset_x = [
            36.997030855955494,
            53.19896748977524,
            20.019889656911783,
            13.257760934502294,
            54.30245664199407,
            74.13939293385138,
        ]
        preset_y = [
            43.30032996341227,
            92.92182147894647,
            82.01102859458813,
            59.79515818116455,
            30.841857576617237,
            67.14503554078097,
        ]
        positions = []
        position_mode = str(self.position_mode).lower()
        jitter_preset = position_mode in {"preset_jitter", "jitter", "jittered"}
        wide_jitter_preset = position_mode in {"preset_wide_jitter", "wide_jitter", "wide-jitter"}
        large_layouts = {
            "large_paper8": {
                "points": [
                    (0.28, 0.24),
                    (0.35, 0.35),
                    (0.44, 0.28),
                    (0.52, 0.39),
                    (0.42, 0.49),
                    (0.31, 0.44),
                    (0.56, 0.31),
                    (0.60, 0.47),
                ],
                "fallback_x": (0.24, 0.64),
                "fallback_y": (0.22, 0.54),
                "jitter": 0.018,
            },
            "large_dense8": {
                "points": [
                    (0.24, 0.36),
                    (0.30, 0.42),
                    (0.36, 0.37),
                    (0.41, 0.46),
                    (0.27, 0.50),
                    (0.34, 0.52),
                    (0.39, 0.48),
                    (0.32, 0.44),
                ],
                "fallback_x": (0.24, 0.42),
                "fallback_y": (0.35, 0.53),
                "jitter": 0.015,
            },
            "large_highrise8": {
                "points": [
                    (0.18, 0.46),
                    (0.24, 0.53),
                    (0.30, 0.48),
                    (0.36, 0.56),
                    (0.42, 0.30),
                    (0.48, 0.36),
                    (0.52, 0.32),
                    (0.44, 0.43),
                ],
                "fallback_x": (0.18, 0.54),
                "fallback_y": (0.30, 0.58),
                "jitter": 0.014,
            },
            "large_urban12": {
                "points": [
                    (0.20, 0.22),
                    (0.28, 0.31),
                    (0.36, 0.24),
                    (0.44, 0.34),
                    (0.52, 0.28),
                    (0.25, 0.42),
                    (0.33, 0.48),
                    (0.43, 0.45),
                    (0.54, 0.40),
                    (0.36, 0.56),
                    (0.48, 0.52),
                    (0.58, 0.47),
                ],
                "fallback_x": (0.20, 0.60),
                "fallback_y": (0.22, 0.58),
                "jitter": 0.017,
            },
        }
        layout_key = {
            "large_cluster8": "large_paper8",
            "large_urban_cluster8": "large_paper8",
            "large_urban_gs12": "large_urban12",
        }.get(position_mode, position_mode)
        large_paper8_preset = layout_key in large_layouts
        if large_paper8_preset:
            layout = large_layouts[layout_key]
            large_points = layout["points"]
            jitter_radius = layout["jitter"] * min(self.ground_length, self.ground_width)
            for i in range(self.num_ground_stations):
                if i < len(large_points):
                    base_x, base_y = large_points[i]
                    x = base_x * self.ground_length + random.uniform(-jitter_radius, jitter_radius)
                    y = base_y * self.ground_width + random.uniform(-jitter_radius, jitter_radius)
                else:
                    x = random.uniform(layout["fallback_x"][0] * self.ground_length, layout["fallback_x"][1] * self.ground_length)
                    y = random.uniform(layout["fallback_y"][0] * self.ground_width, layout["fallback_y"][1] * self.ground_width)
                positions.append(
                    {
                        "position": np.array(
                            [
                                float(np.clip(x, 0.0, self.ground_length)),
                                float(np.clip(y, 0.0, self.ground_width)),
                                0.0,
                            ],
                            dtype=float,
                        )
                    }
                )
            return positions
        if position_mode == "random":
            preset_limit = 0
        else:
            preset_limit = len(preset_x)
        for i in range(self.num_ground_stations):
            if i < preset_limit:
                x = preset_x[i] + 100
                y = preset_y[i] + 100
                if jitter_preset or wide_jitter_preset:
                    jitter_radius = min(42.0, 0.21 * min(self.ground_length, self.ground_width)) if wide_jitter_preset else min(18.0, 0.09 * min(self.ground_length, self.ground_width))
                    x += random.uniform(-jitter_radius, jitter_radius)
                    y += random.uniform(-jitter_radius, jitter_radius)
                    x = float(np.clip(x, 0.0, self.ground_length))
                    y = float(np.clip(y, 0.0, self.ground_width))
            else:
                x = random.uniform(0.1 * self.ground_length, 0.9 * self.ground_length)
                y = random.uniform(0.1 * self.ground_width, 0.9 * self.ground_width)
            positions.append({"position": np.array([x, y, 0.0], dtype=float)})
        return positions

    def _sample_gt_motion(self):
        walkspeed = random.uniform(0, 4)
        for gs in self.ground_stations:
            alpha = random.uniform(0, 2) * np.pi
            next_x = gs["position"][0] + walkspeed * self.t * math.cos(alpha)
            next_y = gs["position"][1] + walkspeed * self.t * math.sin(alpha)
            if 0 <= next_x <= self.ground_length:
                gs["position"][0] = next_x
            if 0 <= next_y <= self.ground_width:
                gs["position"][1] = next_y

    def _softmax_bandwidth(self, action):
        logits = np.asarray(action[: self.num_ground_stations], dtype=float)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        denom = np.sum(exp_logits)
        if denom <= 0:
            return np.ones(self.num_ground_stations) / self.num_ground_stations
        return exp_logits / denom

    def _large_paper_our_bandwidth(self, base_bandwidth, sensing_distances):
        priority_logits = -np.asarray(sensing_distances, dtype=float) / max(0.08 * self.distance_scale, 1e-6)
        priority_logits -= np.max(priority_logits)
        priority = np.exp(priority_logits)
        priority /= max(np.sum(priority), 1e-9)
        blended = 0.35 * np.asarray(base_bandwidth, dtype=float) + 0.65 * priority
        return blended / max(np.sum(blended), 1e-9)

    def get_los_probability(self, h, d, scene=None, occlusion=1.0):
        params = self.scene_params[scene or self.scene]
        d = max(float(d), 1e-3)
        theta = math.degrees(math.atan2(h, d))
        p_los = 1.0 / (1.0 + params["a"] * math.exp(-params["b"] * (theta - params["a"])))
        return float(np.clip(occlusion * p_los, 0.0, 1.0))

    def communication(self, h, d_true, scene=None, occlusion=1.0):
        params = self.scene_params[scene or self.scene]
        c = 3e8
        f_c = 9e8
        d_true = max(float(d_true), 1e-3)
        dist_3d = math.sqrt(h ** 2 + d_true ** 2)
        p_los = self.get_los_probability(h, d_true, scene=scene, occlusion=occlusion)
        fspl = 20 * math.log10(4 * math.pi * f_c * dist_3d / c)
        path_loss = fspl + params["eta_nlos"] + p_los * (params["eta_los"] - params["eta_nlos"])
        snr = self.P_tx * 10 ** (-path_loss / 10) / (self.B * 10 ** (-169 / 10))
        rate = self.B * math.log2(1 + snr)
        return rate, p_los, path_loss

    def compute_energy_slot(self, mode):
        e_comm = self.P_tx * self.t
        e_ctrl = self.P_ctrl * self.t_ctrl
        e_isac = 0.0
        e_vlm = 0.0
        e_gate = 0.0
        if mode == 0:
            e_isac = self.P_isac * self.t_isac
        elif mode == 1:
            e_vlm = self.P_vlm * self.t_vlm
        elif mode == 3:
            e_isac = self.P_isac * self.t_isac
            e_vlm = self.P_vlm * self.t_vlm
            e_gate = self.P_gate * self.t_gate
        return e_comm + e_ctrl + e_isac + e_vlm + e_gate

    def get_gate_state_dict(self):
        return {key: value.detach().cpu() for key, value in self.gate_module.state_dict().items()}

    def load_gate_state_dict(self, state_dict):
        if not state_dict:
            return False
        self.gate_module.load_state_dict(state_dict)
        self.gate_module.eval()
        return True

    def _distance_tensor(self, distances):
        normalized = np.clip(np.asarray(distances, dtype=float) / self.distance_scale, 0.0, 1.5)
        return torch.tensor(normalized, dtype=torch.float32, device=self.device).view(self.num_ground_stations, 1)

    def _estimate_distances(self, loc_uav_array, loc_gt_array, mode):
        true_distances = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[:, :2], axis=1)
        true_bearings = np.arctan2(loc_gt_array[:, 1] - loc_uav_array[1], loc_gt_array[:, 0] - loc_uav_array[0])
        vlm_noise = np.random.normal(0.0, self.vlm_noise_std, size=self.num_ground_stations)
        isac_noise = np.random.normal(0.0, self.isac_noise_std, size=self.num_ground_stations)
        no_isac_noise = np.random.normal(0.0, self.noisac_noise_std, size=self.num_ground_stations)
        vlm_angle_noise = np.random.normal(0.0, self.vlm_angle_noise_std, size=self.num_ground_stations)
        isac_angle_noise = np.random.normal(0.0, self.isac_angle_noise_std, size=self.num_ground_stations)
        no_isac_angle_noise = np.random.normal(0.0, self.noisac_angle_noise_std, size=self.num_ground_stations)

        vlm_est = np.maximum(true_distances + vlm_noise, 1e-3)
        isac_est = np.maximum(true_distances + isac_noise, 1e-3)
        no_isac_est = np.maximum(true_distances + no_isac_noise, 1e-3)
        vlm_bearings = true_bearings + vlm_angle_noise
        isac_bearings = true_bearings + isac_angle_noise
        no_isac_bearings = true_bearings + no_isac_angle_noise

        if mode == 2:
            return true_distances, no_isac_est, no_isac_bearings, {}
        if mode == 1:
            return true_distances, vlm_est, vlm_bearings, {"vlm": vlm_est.copy(), "vlm_bearings": vlm_bearings.copy()}
        if mode == 0:
            return true_distances, isac_est, isac_bearings, {"isac": isac_est.copy(), "isac_bearings": isac_bearings.copy()}

        vlm_tensor = self._distance_tensor(vlm_est)
        isac_tensor = self._distance_tensor(isac_est)
        fused_norm, gate_weights = self.gate_module(vlm_tensor, isac_tensor)
        fused_est = np.maximum(fused_norm.detach().cpu().numpy().reshape(-1) * self.distance_scale, 1e-3)
        gate_np = gate_weights.detach().cpu().numpy().reshape(-1)
        fused_bearings = gate_np * vlm_bearings + (1.0 - gate_np) * isac_bearings

        if self.train_gate_online:
            gt_tensor = self._distance_tensor(true_distances)
            loss = self.gate_loss_fn(fused_norm, gt_tensor)
            self.gate_optimizer.zero_grad()
            loss.backward()
            self.gate_optimizer.step()

        return true_distances, fused_est, fused_bearings, {
            "vlm": vlm_est.copy(),
            "isac": isac_est.copy(),
            "fused": fused_est.copy(),
            "g": gate_np,
        }

    def _build_observed_positions(self, loc_uav_array, loc_gt_array, sensing_distances, sensing_bearings):
        observed = np.array(loc_gt_array, dtype=float, copy=True)
        for i in range(self.num_ground_stations):
            observed[i, 0] = loc_uav_array[0] + float(sensing_distances[i]) * math.cos(float(sensing_bearings[i]))
            observed[i, 1] = loc_uav_array[1] + float(sensing_distances[i]) * math.sin(float(sensing_bearings[i]))
        observed[:, 0] = np.clip(observed[:, 0], 0.0, self.ground_length)
        observed[:, 1] = np.clip(observed[:, 1], 0.0, self.ground_width)
        observed[:, 2] = 0.0
        return observed

    def _noisac_assisted_heading(self, loc_uav_array, loc_gt_array, policy_heading):
        true_distances = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[:, :2], axis=1)
        true_bearings = np.arctan2(loc_gt_array[:, 1] - loc_uav_array[1], loc_gt_array[:, 0] - loc_uav_array[0])
        distance_noise = np.random.normal(0.0, self.noisac_noise_std, size=self.num_ground_stations)
        angle_noise = np.random.normal(0.0, self.noisac_angle_noise_std, size=self.num_ground_stations)
        sensing_distances = np.maximum(true_distances + distance_noise, 1e-3)
        sensing_bearings = true_bearings + angle_noise
        observed_positions = self._build_observed_positions(loc_uav_array, loc_gt_array, sensing_distances, sensing_bearings)
        estimated_centroid = np.mean(observed_positions[:, :2], axis=0)
        assist_heading = math.atan2(estimated_centroid[1] - loc_uav_array[1], estimated_centroid[0] - loc_uav_array[0])
        blended_heading = policy_heading + self.noisac_heading_assist_weight * angle_difference(assist_heading, policy_heading)
        return blended_heading % (2 * np.pi), assist_heading

    def _build_aux_features(self, true_distances, sensing_distances, los_probs, observed_positions, true_positions, guide_strength):
        distance_norm = np.clip(np.asarray(sensing_distances, dtype=float) / self.distance_scale, 0.0, 1.0)
        true_norm = np.clip(np.asarray(true_distances, dtype=float) / self.distance_scale, 0.0, 1.0)
        error_norm = np.clip(np.abs(np.asarray(sensing_distances, dtype=float) - np.asarray(true_distances, dtype=float)) / self.distance_scale, 0.0, 1.0)
        if self.num_ground_stations > 1:
            nearest_idx = float(np.argmin(sensing_distances)) / float(self.num_ground_stations - 1)
        else:
            nearest_idx = 0.0
        battery_ratio = np.clip(self.e_battery_uav / max(self.initial_battery, 1e-6), 0.0, 1.0)
        centroid_xy = np.mean(observed_positions[:, :2], axis=0)
        centroid_rel = (centroid_xy - np.asarray(self.loc_uav[:2], dtype=float)) / self.distance_scale
        nearest_xy = observed_positions[int(np.argmin(sensing_distances)), :2]
        nearest_rel = (nearest_xy - np.asarray(self.loc_uav[:2], dtype=float)) / self.distance_scale
        true_centroid_xy = np.mean(true_positions[:, :2], axis=0)
        true_centroid_rel = guide_strength * (true_centroid_xy - np.asarray(self.loc_uav[:2], dtype=float)) / self.distance_scale
        true_nearest_xy = true_positions[int(np.argmin(true_distances)), :2]
        true_nearest_rel = guide_strength * (true_nearest_xy - np.asarray(self.loc_uav[:2], dtype=float)) / self.distance_scale
        base_aux = np.array(
            [
                np.min(distance_norm),
                np.mean(distance_norm),
                np.max(distance_norm),
                np.std(distance_norm),
                nearest_idx,
                np.min(los_probs),
                np.mean(los_probs),
                np.max(los_probs),
                np.std(los_probs),
                battery_ratio,
                self.loc_uav[0] / self.ground_length,
                self.loc_uav[1] / self.ground_width,
                self.loc_uav[2] / self.height_max,
                np.mean(true_norm),
                np.mean(error_norm),
                centroid_rel[0],
                centroid_rel[1],
                nearest_rel[0],
                nearest_rel[1],
                true_centroid_rel[0],
                true_centroid_rel[1],
                true_nearest_rel[0],
                true_nearest_rel[1],
            ],
            dtype=float,
        )
        true_rel = guide_strength * (true_positions[:, :2] - np.asarray(self.loc_uav[:2], dtype=float)) / self.distance_scale
        true_guide = np.concatenate((true_rel[:, 0], true_rel[:, 1], guide_strength * true_norm))
        return np.concatenate((base_aux, true_guide.astype(float)))

    def _get_state(self, true_distances, sensing_distances, sensing_bearings, los_probs, mode=0):
        uav_normalized = np.array(
            [self.loc_uav[0] / self.ground_length, self.loc_uav[1] / self.ground_width, self.loc_uav[2] / self.height_max],
            dtype=float,
        )
        loc_uav_array = np.asarray(self.loc_uav, dtype=float)
        loc_gt_array = np.array([gt["position"] for gt in self.ground_stations], dtype=float)
        observed_positions = self._build_observed_positions(loc_uav_array, loc_gt_array, sensing_distances, sensing_bearings)
        gs_positions_x = np.array([gs[0] / self.ground_length for gs in observed_positions], dtype=float)
        gs_positions_y = np.array([gs[1] / self.ground_width for gs in observed_positions], dtype=float)
        gs_positions_z = np.array([gs[2] / self.height_max for gs in observed_positions], dtype=float)
        guide_strength = self.guide_strength_for(mode)
        aux_features = self._build_aux_features(
            true_distances,
            sensing_distances,
            los_probs,
            observed_positions,
            loc_gt_array,
            guide_strength,
        )
        return np.concatenate((uav_normalized, gs_positions_x, gs_positions_y, gs_positions_z, aux_features))

    def step(self, action, mode=0):
        is_terminal = False
        reward = 0.0
        self._sample_gt_motion()
        prev_loc_uav_array = np.asarray(self.loc_uav, dtype=float)
        loc_gt_before_move = np.array([gt["position"] for gt in self.ground_stations], dtype=float)
        prev_true_distances = np.linalg.norm(prev_loc_uav_array[:2] - loc_gt_before_move[:, :2], axis=1)
        prev_centroid = np.mean(loc_gt_before_move[:, :2], axis=0)
        prev_centroid_distance = np.linalg.norm(prev_loc_uav_array[:2] - prev_centroid)
        prev_nearest_distance = float(np.min(prev_true_distances))

        theta = (action[self.num_ground_stations] + 1) * np.pi
        vel_horizontal = max((action[self.num_ground_stations + 1] + 1) / 2, 0.0)
        vel_vertical = action[self.num_ground_stations + 2]
        noisac_assist_heading = None
        if mode == 2:
            if self.large_paper8_mode:
                vel_horizontal = max(vel_horizontal, self.large_paper_noisac_min_horizontal)
            theta, noisac_assist_heading = self._noisac_assisted_heading(prev_loc_uav_array, loc_gt_before_move, theta)
        target_heading = math.atan2(prev_centroid[1] - prev_loc_uav_array[1], prev_centroid[0] - prev_loc_uav_array[0])
        if self.large_paper8_mode and mode in {0, 1, 3}:
            assist_by_mode = {
                0: self.large_paper_isac_heading_assist_weight,
                1: self.large_paper_vlm_heading_assist_weight,
                3: self.large_paper_our_heading_assist_weight,
            }
            assist_weight = assist_by_mode.get(mode, 0.0)
            theta = (
                theta
                + assist_weight * angle_difference(target_heading, theta)
            ) % (2 * np.pi)
        heading_error = abs(angle_difference(theta, target_heading)) / np.pi
        heading_alignment_bonus = max(0.0, 1.0 - heading_error)
        turn_change = 0.0 if self.prev_heading is None else abs(angle_difference(theta, self.prev_heading)) / np.pi
        horizontal_change = 0.0 if self.prev_horizontal_action is None else abs(vel_horizontal - self.prev_horizontal_action)
        vertical_change = 0.0 if self.prev_vertical_action is None else abs(vel_vertical - self.prev_vertical_action) / 2.0
        smoothness_penalty = turn_change + 0.5 * horizontal_change + 0.25 * vertical_change

        dx_uav = vel_horizontal * self.vel_h_max * self.t * math.cos(theta)
        dy_uav = vel_horizontal * self.vel_h_max * self.t * math.sin(theta)
        dz_uav = vel_vertical * self.vel_v_max * self.t

        boundary_penalty = 0.0
        if 0 <= self.loc_uav[0] + dx_uav <= self.ground_length:
            self.loc_uav[0] += dx_uav
        else:
            boundary_penalty = -4.0
            vel_horizontal = 0.0
        if 0 <= self.loc_uav[1] + dy_uav <= self.ground_width:
            self.loc_uav[1] += dy_uav
        else:
            boundary_penalty = -4.0
            vel_horizontal = 0.0
        if self.height_min <= self.loc_uav[2] + dz_uav <= self.height_max:
            self.loc_uav[2] += dz_uav
        else:
            boundary_penalty = -4.0
            vel_vertical = 0.0

        loc_uav_array = np.asarray(self.loc_uav, dtype=float)
        loc_gt_array = np.array([gt["position"] for gt in self.ground_stations], dtype=float)
        bandwidth = self._softmax_bandwidth(action)
        true_distances, sensing_distances, sensing_bearings, fusion_debug = self._estimate_distances(loc_uav_array, loc_gt_array, mode)
        if self.large_paper8_mode and mode == 3:
            bandwidth = self._large_paper_our_bandwidth(bandwidth, sensing_distances)
        observed_positions = self._build_observed_positions(
            loc_uav_array, loc_gt_array, sensing_distances, sensing_bearings
        )
        perception_error = float(
            np.mean(np.linalg.norm(observed_positions[:, :2] - loc_gt_array[:, :2], axis=1))
            / self.distance_scale
        )
        perception_error_penalty = self.perception_error_penalty_weight * perception_error

        total_rate = 0.0
        los_probs = np.zeros(self.num_ground_stations, dtype=float)
        for i, distance in enumerate(true_distances):
            rate, p_los, _ = self.communication(self.loc_uav[2], distance, scene=self.scene)
            los_probs[i] = p_los
            total_rate += bandwidth[i] * rate
        if mode == 1:
            total_rate *= self.large_paper_vlm_rate_scale

        current_centroid = np.mean(loc_gt_array[:, :2], axis=0)
        current_centroid_distance = np.linalg.norm(loc_uav_array[:2] - current_centroid)
        current_nearest_distance = float(np.min(true_distances))

        e_fly = self.flight_energy_slot(abs(vel_horizontal * self.vel_h_max), abs(vel_vertical * self.vel_v_max))
        e_comp = self.compute_energy_slot(mode)
        self.E_uav_cost = e_fly + e_comp
        self.e_battery_uav -= self.E_uav_cost

        energy_efficiency_reward = (total_rate / 1e6) / max(self.E_uav_cost / self.flight_energy_slot(10.4, 0), 1e-6)
        distance_progress = np.clip((np.mean(prev_true_distances) - np.mean(true_distances)) / self.distance_scale, -1.0, 1.0)
        proximity_bonus = max(0.0, 1.0 - np.mean(true_distances) / self.distance_scale)
        nearest_bonus = max(0.0, 1.0 - np.min(true_distances) / self.distance_scale)
        centroid_progress = np.clip((prev_centroid_distance - current_centroid_distance) / self.distance_scale, -1.0, 1.0)
        centroid_proximity = max(0.0, 1.0 - current_centroid_distance / self.distance_scale)
        service_region_radius = max(0.22 * self.distance_scale, 28.0)
        service_region_bonus = max(0.0, 1.0 - current_centroid_distance / service_region_radius)
        reward += (
            energy_efficiency_reward
            + self.progress_reward_weight * distance_progress
            + self.proximity_reward_weight * proximity_bonus
            + self.nearest_reward_weight * nearest_bonus
            + self.centroid_progress_reward_weight * centroid_progress
            + self.centroid_proximity_reward_weight * centroid_proximity
            + self.service_region_bonus_weight * service_region_bonus
            + 2 * boundary_penalty
            - perception_error_penalty
        )
        our_heading_reward = 0.0
        our_smoothness_penalty = 0.0
        if mode == 3:
            our_heading_reward = self.our_heading_alignment_weight * heading_alignment_bonus
            our_smoothness_penalty = self.our_smoothness_penalty_weight * smoothness_penalty
            reward += our_heading_reward - our_smoothness_penalty
        next_state = self._get_state(true_distances, sensing_distances, sensing_bearings, los_probs, mode=mode)
        self.prev_heading = theta
        self.prev_horizontal_action = vel_horizontal
        self.prev_vertical_action = vel_vertical
        data = total_rate * self.t

        debug_payload = {
            "true_distances": true_distances,
            "estimated_distances": sensing_distances,
            "estimated_bearings": sensing_bearings,
            "los_probs": los_probs,
            "distance_progress": float(distance_progress),
            "proximity_bonus": float(proximity_bonus),
            "nearest_bonus": float(nearest_bonus),
            "centroid_progress": float(centroid_progress),
            "centroid_proximity": float(centroid_proximity),
            "service_region_bonus": float(service_region_bonus),
            "centroid_distance": float(current_centroid_distance),
            "nearest_distance": float(current_nearest_distance),
            "energy_efficiency_reward": float(energy_efficiency_reward),
            "perception_error": float(perception_error),
            "perception_error_penalty": float(perception_error_penalty),
            "heading_alignment_bonus": float(heading_alignment_bonus),
            "noisac_assist_heading": None if noisac_assist_heading is None else float(noisac_assist_heading),
            "noisac_heading_assist_weight": float(self.noisac_heading_assist_weight if mode == 2 else 0.0),
            "smoothness_penalty": float(smoothness_penalty),
            "our_heading_reward": float(our_heading_reward),
            "our_smoothness_penalty": float(our_smoothness_penalty),
            "energy_fly": e_fly,
            "energy_comp": e_comp,
            "energy_total": self.E_uav_cost,
            "effective_rate_scale": float(self.large_paper_vlm_rate_scale if mode == 1 else 1.0),
        }
        debug_payload.update(fusion_debug)
        return next_state, action, reward, is_terminal, data, vel_horizontal * self.vel_h_max, vel_vertical * self.vel_v_max, debug_payload

    def reset(self, mode=0):
        self.e_battery_uav = self.initial_battery
        self.loc_uav = list(self.uav_start)
        self.ground_stations = [{"position": pos["position"].copy()} for pos in self.base_positions]
        self.prev_heading = None
        self.prev_horizontal_action = None
        self.prev_vertical_action = None
        loc_uav_array = np.asarray(self.loc_uav, dtype=float)
        loc_gt_array = np.array([gt["position"] for gt in self.ground_stations], dtype=float)
        true_distances, sensing_distances, sensing_bearings, _ = self._estimate_distances(loc_uav_array, loc_gt_array, mode)
        los_probs = np.array([self.get_los_probability(self.loc_uav[2], d) for d in true_distances], dtype=float)
        return self._get_state(true_distances, sensing_distances, sensing_bearings, los_probs, mode=mode)

    def flight_energy_slot(self, vel, vel_v):
        d_o = 0.6
        rho = 1.225
        s = 0.05
        G = 0.503
        U_tip = 120
        v_o = 4.3
        omega = 300
        R = 0.4
        delta = 0.012
        k = 0.1
        W = 20
        P0 = (delta / 8) * rho * s * G * (omega ** 3) * (R ** 3)
        P1 = (1 + k) * (W ** (3 / 2) / math.sqrt(2 * rho * G))
        P2 = 11.46
        core = math.sqrt(1 + vel ** 4 / (4 * v_o ** 4)) - vel ** 2 / (2 * v_o ** 2)
        induced = math.sqrt(abs(core))
        return self.t * (
            P0 * (1 + 3 * vel ** 2 / U_tip ** 2)
            + 0.5 * d_o * rho * s * G * vel ** 3
            + P1 * induced
            + P2 * vel_v
        )


def angle_from_uav(uav, gt):
    vector = gt - uav
    angle = np.arctan2(vector[1], vector[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle
