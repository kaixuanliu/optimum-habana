# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch VideoLlava model."""

import torch
from transformers.models.video_llava.modeling_video_llava import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaCausalLMOutputWithPast
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

class GaudiVideoLlavaForConditionalGeneration(VideoLlavaForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values_images: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, VideoLlavaCausalLMOutputWithPast]:
        r"""
        
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (pixel_values_images is not None or pixel_values_videos is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image/video tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            img_token_not_enough = (input_ids == self.config.image_token_index).sum(
                1
            ).max() < self.config.image_seq_length
            video_token_not_enough = (input_ids == self.config.video_token_index).sum(
                1
            ).max() < self.config.video_seq_length
            inputs_not_expanded = (img_token_not_enough and pixel_values_images is not None) or (
                video_token_not_enough and pixel_values_videos is not None
            )
            pixels_present = input_ids.shape[-1] == 1 and (
                pixel_values_images is not None or pixel_values_videos is not None
            )
            legacy_processing = inputs_not_expanded or pixels_present

        if pixel_values_images is not None or pixel_values_videos is not None:
            image_outputs, video_outputs, num_frames = self._get_vision_features(
                pixel_values_images=pixel_values_images,
                pixel_values_videos=pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            image_features = video_features = None
            if image_outputs is not None:
                image_features = self.multi_modal_projector(image_outputs)
            if video_outputs is not None:
                video_features = self.multi_modal_projector(video_outputs)

            if legacy_processing:
                logger.warning_once(
                    "Expanding inputs for image tokens in Video-LLaVa should be done in processing. "
                    "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                    "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                    "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                )
                if input_ids.shape[1] != 1:
                    for features, frames in ((image_features, 1), (video_features, num_frames)):
                        if features is not None:
                            (
                                inputs_embeds,
                                attention_mask,
                                labels,
                                position_ids,
                                input_ids,
                            ) = self._merge_input_ids_with_visual_features(
                                features,
                                inputs_embeds,
                                input_ids,
                                attention_mask,
                                labels,
                                num_frames=frames,
                            )
                    cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
                else:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    target_length = input_ids.shape[1]
                    past_length = first_layer_past_key_value.shape[-1]

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], past_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                    cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[
                        -target_length:
                    ]

            # TODO: @raushan retain only the new behavior after v4.47
            else:
                if image_outputs is not None:
                    special_image_mask = (
                        (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    )
                    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

                if video_outputs is not None:
                    special_image_mask = (
                        (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    )
                    video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, video_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            token_idx=token_idx,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VideoLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values_images is not None else None,
            video_hidden_states=video_features if pixel_values_videos is not None else None,
        )