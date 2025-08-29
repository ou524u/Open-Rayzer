# code for rayzer.py
# we reinplement RayZer: A Self-supervised Large View Synthesis Model from Hanwen Jiang et. al here.


import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from src.utils import data_utils
# from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer
from functools import partial
from .tokenizers import create_tokenizer, create_unpatchifier, create_transformer_blocks, c2w_to_plucker
from .tokenizers import ConcatMLPFuse # dinov3_tokenizer, dinov2_tokenizer
from .layers.camera_head import CameraHead
from .layers.pos_embed import RoPE2D, PositionGetter
import copy

class rayzer(nn.Module):
# Below are init functions
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformers()

        # Initialize latent scene tokens
        self._init_latent_tokens()
        
        # Initialize loss computer
        self.loss_computer = LossComputer(config)
        

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        image_tokenizer_mode = self.config.model.tokenizer.get("mode", "linear")
        if image_tokenizer_mode == "linear":
            self.image_tokenizer = create_tokenizer( # [b, v, c, h, w] -> [b*v, n_patches, d]
                in_channels = 3,
                patch_size = self.config.model.tokenizer.patch_size,
                d_model = self.config.model.transformer.d
            )
        elif image_tokenizer_mode == "dinov3":
            self.image_tokenizer = dinov3_tokenizer() # using pretrained dinov3 as tokenizer.
        

        self.target_pose_tokenizer = create_tokenizer( # [b, v, c, h, w] -> [b*v, n_patches, d]
            in_channels = 6,
            patch_size = self.config.model.tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )

        # Image unpatchfier
        self.image_unpatchifier = create_unpatchifier( # [b, v, n_patches, d] -> [b, v, c, h, w]
            image_size=self.config.model.tokenizer.image_size,
            patch_size = self.config.model.tokenizer.patch_size,
            d_model = self.config.model.transformer.d,
            out_channels = self.config.model.tokenizer.out_channels
        )

        # Encoder fusion strategy
        if self.config.get("unposed", {}).get("fusion_strategy", "pixel_fuse") == "pixel_fuse":
            self.encoder_tokenizer = create_tokenizer( # [b, v, c, h, w] -> [b*v, n_patches, d]
                in_channels = 9,
                patch_size = self.config.model.tokenizer.patch_size,
                d_model = self.config.model.transformer.d
            )
        elif self.config.get("unposed", {}).get("fusion_strategy", "pixel_fuse") == "feature_fuse":
            # Pose tokenizer
            if self.config.get("unposed", {}).get("decoder_as_canonical", False):
                self.pose_tokenizer = self.target_pose_tokenizer
            else:
                self.pose_tokenizer = create_tokenizer( # [b, v, c, h, w] -> [b*v, n_patches, d]
                    in_channels = 6,
                    patch_size = self.config.model.tokenizer.patch_size,
                    d_model = self.config.model.transformer.d
                )
            self.encoder_fuse = ConcatMLPFuse( # [b*v, n_patches, d] -> [b*v, n_patches, d]
                d = self.config.model.transformer.d,
                # hidden = self.config.model.transformer.d,
                out_dim = self.config.model.transformer.d,
            )

        # Pose wrapper
        self.pose_wrapper = CameraHead( # [b, v, n_patches, d] -> [b, v, 4, 4], [b, v, 1]
            dim = self.config.model.transformer.d,
            image_size = self.config.model.tokenizer.image_size,
            patch_size = self.config.model.tokenizer.patch_size,
            predict_fov = self.config.get("unposed", {}).get("predict_fov", True) # rayzer predicts fov by default
        )

        # Pose unwrapper
        self.pose_unwrapper = partial( # [b, v, 4, 4] & [b, v, 4] -> [b, v, 6, h, w]
            c2w_to_plucker, 
            H = self.config.model.tokenizer.image_size,
            W = self.config.model.tokenizer.image_size
        )


    def _init_transformers(self):
        # rope
        self.rope = None
        if self.config.model.transformer.get("rope", False):
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.config.model.transformer.get("rope_freq", 100))
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()

        # Estimator
        self.estimator_blocks, self.estimator_layernorm = create_transformer_blocks(
            depth=self.config.model.transformer.estimator_depth,
            d=self.config.model.transformer.d,
            d_head=self.config.model.transformer.d_head,
            use_special_init=self.config.model.transformer.special_init,
            use_depth_init=self.config.model.transformer.depth_init,
            use_qk_norm=self.config.model.transformer.use_qk_norm,
            rope=self.rope
        )

        # Decoder
        self.decoder_blocks, self.decoder_layernorm = create_transformer_blocks(
            depth=self.config.model.transformer.decoder_depth,
            d=self.config.model.transformer.d,
            d_head=self.config.model.transformer.d_head,
            use_special_init=self.config.model.transformer.special_init,
            use_depth_init=self.config.model.transformer.depth_init,
            use_qk_norm=self.config.model.transformer.use_qk_norm,
            rope=self.rope
        )

        # encoder
        self.encoder_blocks, self.encoder_layernorm = create_transformer_blocks(
            depth=self.config.model.transformer.encoder_depth,
            d=self.config.model.transformer.d,
            d_head=self.config.model.transformer.d_head,
            use_special_init=self.config.model.transformer.special_init,
            use_depth_init=self.config.model.transformer.depth_init,
            use_qk_norm=self.config.model.transformer.use_qk_norm,
            rope=self.rope
        )

    def pass_layers(self, input_tokens, transformer_type='estimator', pos=None, v_all=1):
        """
        Helper function to pass input tokens through transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            transformer_type: str, one of ['estimator', 'decoder', 'encoder']
                Which transformer to use for processing.
            pos: Tensor of shape [batch_size, num_views * num_patches, 2] 
                Used for RoPE.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through the transformer blocks.
        """
        gradient_checkpoint = self.config.training.get("grad_checkpoint", True)
        checkpoint_every = self.config.training.get("grad_checkpoint_every", 1)
        if checkpoint_every <= 1:
            gradient_checkpoint = False

        vggstyle = self.config.model.transformer.get("vggstyle", False)
        if transformer_type != 'estimator':
            vggstyle = False # NOTE: we only do vggt style attention in estimator.
        B_v_all, n_patches, d = input_tokens.shape
        B = B_v_all // v_all

        # Select the appropriate transformer blocks and layernorm
        if transformer_type == 'estimator':
            blocks = self.estimator_blocks
            layernorm = self.estimator_layernorm
        elif transformer_type == 'decoder':
            blocks = self.decoder_blocks
            layernorm = self.decoder_layernorm
        elif transformer_type == 'encoder':
            blocks = self.encoder_blocks
            layernorm = self.encoder_layernorm
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")

        # Apply input layernorm
        input_tokens = layernorm(input_tokens)
        
        num_layers = len(blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for idx, layer in enumerate(blocks):
                if self.rope is not None and pos is not None:
                    if vggstyle: # vggt style attention
                        if idx % 2 == 0:
                            input_tokens = input_tokens.reshape(B, v_all*n_patches, -1)
                            pos = pos.reshape(B, v_all*n_patches, -1)
                        else:
                            input_tokens = input_tokens.reshape(B_v_all, n_patches, -1)
                            pos = pos.reshape(B_v_all, n_patches, -1)
                    
                    input_tokens = layer(input_tokens, xpos=pos)
                else:
                    input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                layer = blocks[idx]
                if self.rope is not None and pos is not None:
                    if vggstyle: # vggt style attention
                        if idx % 2 == 0:
                            tokens = tokens.reshape(B, v_all*n_patches, -1)
                            pos = pos.reshape(B, v_all*n_patches, -1)
                        else:
                            tokens = tokens.reshape(B_v_all, n_patches, -1)
                            pos = pos.reshape(B_v_all, n_patches, -1)
                    
                    tokens = layer(tokens, xpos=pos)
                else:
                    tokens = layer(tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens
    
    def _init_latent_tokens(self):
        # latent vectors for LVSM encoder-decoder
        self.latent_scene_tokens = nn.Parameter(
            torch.randn(
                self.config.model.transformer.n_latent_vectors,
                self.config.model.transformer.d,
            )
        )
        nn.init.trunc_normal_(self.latent_scene_tokens, std=0.02)

        # if self.config.get("unposed", {}).get("learnable_fov", False):
        #     self.learnable_fov = nn.Parameter(
        #         torch.randn(1) # [1]
        #     )
        #     nn.init.trunc_normal_(self.learnable_fov, std=0.02)
        #     # self.learnable_fov = F.softplus(self.learnable_fov).clamp(min=1e-6)


    def fov_to_intrinsics(self, fov: torch.Tensor, image_size: int, v: int) -> torch.Tensor:
        """
        """
        b, _ = fov.shape
        f = 0.5 * image_size / torch.tan(0.5 * fov)  # [b, 1]
        cx = cy = image_size / 2.0

        intrinsics = torch.stack([
            f.squeeze(-1),  # fx
            f.squeeze(-1),  # fy
            torch.full((b,), cx, device=fov.device, dtype=fov.dtype),  # cx
            torch.full((b,), cy, device=fov.device, dtype=fov.dtype)   # cy
        ], dim=-1)  # [b, 4]

        # expand to [b, v, 4]
        intrinsics = intrinsics.unsqueeze(1).expand(b, v, 4)
        return intrinsics

    def forward(self, data_batch, 
                has_target_image=True, # If true, target views will have image. This enables supervision.
                target_has_input=False, 
            ):
        # NOTE: in rayzer model the estimator operates over all views.
        #
        # Input and Target data_batch (dict): Contains processed tensors with the following keys:
        #     - 'image' (torch.Tensor): Shape [b, v, c, h, w]
        #     - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
        #     - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            # - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
            # - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
        #     - 'image_h_w' (tuple): (height, width)
        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = target_has_input, compute_rays=False)
        
        B, v_input, c, h, w = input.image.shape
        _, v_target, _, _, _ = target.image.shape

        # preparations for the forward pass
        n_latents, _ = self.latent_scene_tokens.shape
        latent_scene_tokens = self.latent_scene_tokens.expand(B, -1, -1) # [B, latent_scene_tokens, d]
        n_patches_h = self.config.model.tokenizer.image_size // self.config.model.tokenizer.patch_size
        n_patches_w = n_patches_h
        n_patches = n_patches_h * n_patches_w
        # prepare rope
        estimator_pos = None
        encoder_pos = None
        decoder_pos = None
        if self.rope is not None:
            estimator_pos = self.position_getter(B*(v_input + v_target), n_patches_h, n_patches_w, input.image.device)
            estimator_pos = estimator_pos + 1

            encoder_latent_pos = torch.zeros(
                B, n_latents, 2,
                device=input.image.device,
                dtype=estimator_pos.dtype
            )
            encoder_pos = self.position_getter(B*v_input, n_patches_h, n_patches_w, input.image.device) # [b*v_input, n_patches, 2]
            encoder_pos = encoder_pos + 1
            encoder_pos = rearrange(encoder_pos, '(b v) n d -> b (v n) d', v=v_input, n=n_patches) # [b, v_input*n_patches, 2]
            encoder_pos = torch.cat([encoder_pos, encoder_latent_pos], dim=1) # [b, v_input*n_patches + n_latents, 2]


            decoder_latent_pos = torch.zeros(
                B*v_target, n_latents, 2,
                device=input.image.device,
                dtype=estimator_pos.dtype
            )
            decoder_pos = self.position_getter(B*v_target, n_patches_h, n_patches_w, input.image.device) # [b*v_target, n_patches, 2]
            decoder_pos = decoder_pos + 1
            decoder_pos = torch.cat([decoder_pos, decoder_latent_pos], dim=1) # [b*v_target, n_latents + n_patches, 2]


        all_images = torch.cat([input.image, target.image], dim=1) # [b, v_input + v_target, c, h, w]
        pixel_feat = self.image_tokenizer(all_images) # [b*(v_input + v_target), n_patches, d]
        pixel_out_feat = self.pass_layers(pixel_feat, transformer_type='estimator', pos=estimator_pos, v_all=(v_input+v_target)) # [b*(v_input + v_target), n_patches, d]
        pixel_out_feat = rearrange(pixel_out_feat, '(b v) n d -> b v n d', v=v_input + v_target) # [b, v_input + v_target, n_patches, d]

        estimated_extrinsics, estimated_fov = self.pose_wrapper(pixel_out_feat) # [b, v_input + v_target, 4, 4], [b, 1]
        
        # print(f"estimated_fov shape: {estimated_fov.shape}")
        if self.config.get("unposed", {}).get("predict_fov", True):
            estimated_intrinsics = self.fov_to_intrinsics(estimated_fov, h, v_input + v_target) # [b, v_input + v_target, 4]
        else:
            estimated_intrinsics = torch.cat([input.fxfycxcy, target.fxfycxcy], dim=1) # [b, v_input + v_target, 4]

        # print(f"estimated_intrinsics shape: {estimated_intrinsics.shape}")
        # print(f"estimated_extrinsics shape: {estimated_extrinsics.shape}")
        estimated_pluckers = self.pose_unwrapper(estimated_extrinsics, fxfycxcy=estimated_intrinsics) # [b, v_input + v_target, 6, h, w]

        input_pixels = input.image # [b, v_input, c, h, w]
        input_pluckers, target_pluckers = estimated_pluckers.split([v_input, v_target], dim=1) # [b, v_input, 6, h, w], [b, v_target, 6, h, w]

        if self.config.get("unposed", {}).get("fusion_strategy", "pixel_fuse") == "pixel_fuse":
            pixel_pluckers = torch.cat([input_pixels, input_pluckers], dim=2) # [b, v_input, c=3+6, h, w]
            pixel_pluckers_feat = self.encoder_tokenizer(pixel_pluckers) # [b*v_input, n_patches, d]

        elif self.config.get("unposed", {}).get("fusion_strategy", "pixel_fuse") == "feature_fuse":
            pixel_feat = self.image_tokenizer(input_pixels) # [b*v_input, n_patches, d]
            plucker_feat = self.pose_tokenizer(input_pluckers) # [b*v_input, n_patches, d]
            pixel_pluckers_feat = self.encoder_fuse(pixel_feat, plucker_feat) # [b*v_input, n_patches, d]

        # _, n_patches, _ = pixel_pluckers_feat.shape
        pixel_pluckers_feat = rearrange(pixel_pluckers_feat, '(b v) n d -> b (v n) d', v=v_input) # [b, v_input*n_patches, d]

        into_encoder = torch.cat([pixel_pluckers_feat, latent_scene_tokens], dim=1) # [b, v_input*n_patches + latent_scene_tokens, d]
        outof_encoder = self.pass_layers(into_encoder, transformer_type='encoder', pos=encoder_pos) # [b, v_input*n_patches + latent_scene_tokens, d]
        _, latent_scene_tokens = outof_encoder.split([v_input*n_patches, n_latents], dim=1) # [b, v_input*n_patches, d], [b, latent_scene_tokens, d]

        # target_pixels = target.image # [b, v_target, c, h, w]
        target_pluckers = target_pluckers # [b, v_target, 6, h, w]

        target_pluckers_feat = self.target_pose_tokenizer(target_pluckers) # [b*v_target, n_patches, d]
        repeated_latent_tokens = repeat(latent_scene_tokens, 'b n d -> (b v_target) n d', v_target=v_target) # [b*v_target, latent_scene_tokens, d]

        into_decoder = torch.cat([target_pluckers_feat, repeated_latent_tokens], dim=1) # [b*v_target, n_latent_vectors + n_patches, d]
        outof_decoder = self.pass_layers(into_decoder, transformer_type='decoder', pos=decoder_pos) # [b*v_target, n_latent_vectors + n_patches, d]
        target_image_tokens, _ = outof_decoder.split([n_patches, n_latents], dim=1) # [b*v_target, n_patches, d], [b*v_target, latent_scene_tokens, d]

        target_image_tokens = rearrange(target_image_tokens, '(b v) n d -> b v n d', v=v_target) # [b, v_target, n_patches, d]
        target_pixels = self.image_unpatchifier(target_image_tokens) # [b, v_target, c, h, w]

        loss_metrics = None
        if has_target_image:
            loss_metrics = self.loss_computer(
                target_pixels,
                target.image
            )

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=target_pixels        
            )
        
        return result
